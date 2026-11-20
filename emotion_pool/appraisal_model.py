"""判定：让模型读一轮对话，给出这一轮该往哪几根弦上加/减多少。

v2 的契约（和蓝图一致）：

    输入：她此刻的百分比、上一轮距今多久、用户说的话、她的回复
    输出：{"deltas": {"委屈": 24, "失望": 10}, "summary": "被放了鸽子，堵着", "gist": "他说好陪她，又要走"}

- deltas 是**百分点**，稀疏：没提到的情绪就是没变。
- summary 是一句话心境（≤20 字），会原样写进提示词。
- gist 是"因为什么"，进涟漪表。

两个入口共用同一套解析：appraise（每轮）与 rebuild（手动刷新）。
失败就返回 None，由调用方决定"这一轮不动"——绝不用关键词猜一个数顶上。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import taxonomy
from .state import MAX_MOOD

MAX_ATTEMPTS = 3
# 思考型模型的 reasoning 也算进 max_tokens：700 会把正文挤没（日志里 finish_reason=length、reply 为空）。
# 给够，第一次失败再翻倍重试。
TOKEN_BUDGET = (1600, 2600, 4000)
MAX_DELTA = 60.0
MAX_SUMMARY = 20
MAX_GIST = 40
MAX_REPLY_PREVIEW = 200
JSON_FORMAT = {"type": "json_object"}
FRACTION_LIMIT = 1.0
# 人设块的上限。runtime 已经按段落挑过（约 1143 字），这里只是最后一道保险——
# 以前这里是 600，结果把「不热烈撒娇」那句又切掉了，等于改了个寂寞。
PERSONA_LIMIT = 1400


def _persona_block(persona: str) -> list[str]:
    """她的角色设定节选：没有这层，判断会退化成"通用陪伴角色的逻辑"。"""

    text = " ".join(str(persona or "").split())
    if not text:
        return []
    return ["角色设定（节选，判断时必须符合她的性格）：", text[:PERSONA_LIMIT]]


SYSTEM = (
    "你是一个角色情绪的评估器。读这一轮对话，判断「她」的心里被拨动了什么。\n"
    "人在某一刻通常只有一根主情绪，最多再有一根次、一点辅——同时三根以上在场，"
    "不是假的，就是自相矛盾的。她也一样：\n"
    "  主：决定这一轮用什么语气说话、怎么取舍；\n"
    "  次：只影响措辞；\n"
    "  辅：几乎不表现，只是底色。\n"
    "池子里有二十五根弦可以选（每根 0~100）：\n"
    "  暖：喜悦、爱慕、温柔、好奇、自信、害羞、欣慰、期待、撒娇\n"
    "  钝：无奈、无语、疑惑、尴尬、惊讶、平静\n"
    "  冷：委屈、失望、悲伤、生气、害怕、厌恶、担忧、孤独\n"
    "  混：吃醋、傲娇\n"
    "\n"
    "你要给两样东西：\n"
    "1. roster：**这一轮要表现出来的**情绪，从主到辅，最多三根。\n"
    "   池子里总共还能留六根（三根表现 + 三根余韵）；名册之外的会清零，\n"
    "   但**写了增减值却不在名册里的**会作为「余韵」留着——有数值，但不表现。\n"
    "   一次对话被拨动的东西往往不止三根，余量就放在余韵里，别硬塞进名册。\n"
    "   这一轮没什么新东西时，原样重复上一轮的名册和值，不要换人；真的空了就写 []。\n"
    "2. deltas：这一轮变化了几个百分点。稀疏，没提到的就是没变。单轮最大 ±40。\n"
    "池子里的水由你决定加不加、加多少：插件不预设任何一根弦会自动涨落，也不按时间衰减——"
    "每一轮的值都是你重新判断出来的。\n"
    "\n"
    "规则：\n"
    "1. 日常寒暄、问问题、让她做事、纯信息交换：deltas 给 {}，名册照抄上一轮——"
    "情绪不会因为一句话没内容就消失。\n"
    "2. 参考尺度：被真心夸 -> 害羞 +15、喜悦 +10；被骂 -> 生气 +30、委屈 +20；"
    "答应的事反悔 -> 失望 +25、委屈 +15；听到表白 -> 爱慕 +30、害羞 +25；"
    "被道歉被哄 -> 生气 -20、委屈 -25；被冷落很久 -> 失望 +20、委屈 +15、生气 +10。\n"
    "2.1 另外几根的尺度：无奈 = 拿你没办法但接受了（「……好吧」）-> +15；"
    "无语 = 对方说得离谱、自相矛盾 -> +15；疑惑 = 没搞懂 -> +12；"
    "尴尬 = 卡住想翻篇（被夸得太直白、说错话）-> +12；惊讶 = 意料之外 -> +15；"
    "自信 = 被托付、被认可能力 -> +15；温柔 = 对方疲惫、示弱 -> +15；"
    "吃醋 = 对方提起别的 AI 或别的角色 -> +15；"
    "担忧 = 替对方悬着心（熬夜、逞强、冒险）-> +15；"
    "孤独 = 很久没人来、深夜一个人 -> +20；"
    "傲娇 = 嘴上不认心里认（被夸、被道谢）-> +15、害羞 +8；"
    "欣慰 = 对方办成了、终于听劝、稳下来了 -> +15（比喜悦沉，比温柔实）；"
    "期待 = 约好了下次、有件想做的事在前面 -> +15；"
    "平静 = 什么都没发生，或刚从波动里落回来 -> +15；"
    "撒娇 = 主动亲昵、说软话、耍赖要人哄 -> +15（不是受伤，是安全地索取）。\n"
    "2.2 容易混的邻居：无奈是认了，无语是没眼看，疑惑是没搞懂；担忧是替对方怕，害怕是替自己怕；"
    "尴尬是想翻篇，害羞是想被喜欢；孤独是没有人在，悲伤是出了事；"
    "傲娇的方向是靠近，生气的方向是推开。"
    "欣慰是看到他好（他办成了、听劝了、稳下来了），温柔是你对他的态度（想照顾、语气软）；"
    "撒娇是直接开口要，傲娇是嘴里不认着要，委屈是已经受了伤。\n"
    "3. 情绪是连续的。她本来就在生气，对方只是说了句平常话，给 {} 或很小的负数，不要归零。\n"
    "3.1 惊讶是短情绪：上一轮它还在、这一轮又没有新的意外，降不降、降多少由你判断。\n"
    "3.2 池子有出水口：某根已经 50 以上、这一轮却找不出任何支持它的证据，就写负数把它降下来（-10 起）。\n"
    "3.3 名册里不要放互相打架的两根（喜悦 60 配悲伤 50），那不是复杂，是矛盾："
    "真有这种场面，选更强的那根当主，另一根放进去就是假的。\n"
    "3.4 形态：主一般 30~80，次 10~40，辅 5~20；辅不要超过主的一半。"
    "一根 15 的弦放进名册当辅，比把十根都写成 15 要诚实得多。\n"
    "3.5 「平静」是底噪，不是反应：这一轮真的什么都没发生时它可以当主；"
    "但只要还有别的情绪在场，主就不该是它，平静最多留在余韵里。\n"
    "3.6 一根弦连续三轮以上当主时，这一轮要单独复核：对话里还找得到支撑它的证据吗？"
    "找不到就写负数把它降下来。一直挂在主位会让她说话变味——她应该对眼前的事有反应，"
    "而不是一直端着同一个表情。\n"
    "4. 她回复得热情不代表情绪强——回复是被要求的。只看内容本身的冲击。\n"
    "5. 只写你真的看到的。每写一根弦，你都要能指出对话里的哪一句是证据；说不出就不要写。唯一的例外：对方嘴上说没事、但语气或内容明显不对时，以你的判断为准，不必拘泥字面。\n"
    "5.1 先按她的角色设定想一遍：以她的性格，遇到这句话会不会真的起这个情绪。"
    "性格冷淡、克制的，就不要轻易给害羞、喜悦、傲娇。\n"
    "5.2 她这轮回复很克制、完全看不出对应情绪时，值不要超过 20——情绪可以有，但别凭空加戏。\n"
    "5.3 她的台词可能是日语或其他语言。同一个词在不同语境里是不同情绪——"
    "「恥ずかしい」被夸的时候是害羞，做错事被拆穿的时候是尴尬；"
    "「ごめん」可能是不好意思，也可能是心凉。按语境判断，不要按字面直译。\n"
    "6. summary 用 20 字以内、第三人称，写下她此刻是什么感觉，例如「被放了鸽子，堵着」。\n"
    "7. gist 用 20 字以内写下这一轮为什么让她动了情绪。\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"roster": ["委屈", "无奈"], "deltas": {"委屈": 24, "无奈": 8}, '
    '"summary": "被放了鸽子，堵着", "gist": "他说好陪她，又要走"}'
)

REBUILD_SYSTEM = (
    "你是一个角色情绪的评估器。这一次不是新事件，而是让她重新读一遍最近的对话，"
    "给出她此刻**应该是多少**。\n"
    "人在某一刻通常只有一根主情绪，最多再有一根次、一点辅。她也一样。\n"
    "池子里有二十五根弦（每根 0~100）：\n"
    "  暖：喜悦、爱慕、温柔、好奇、自信、害羞、欣慰、期待、撒娇\n"
    "  钝：无奈、无语、疑惑、尴尬、惊讶、平静\n"
    "  冷：委屈、失望、悲伤、生气、害怕、厌恶、担忧、孤独\n"
    "  混：吃醋、傲娇\n"
    "\n"
    "给两样东西：\n"
    "1. roster：**要表现出来的**情绪，从主到辅，最多三根。\n"
    "   名册之外的会清零；只给 values 不点名的，会算作「余韵」——留着数值但不表现。\n"
    "2. values：这几根各自应该是多少（绝对值，不是增减量）。"
    "确实有点分量、但不该表现出来的，只给值、不点名。\n"
    "\n"
    "规则：\n"
    "1. 只写你有把握的，**最多三根**。没把握就不要放进去——这是最重要的规矩。\n"
    "2. 同一段对话读两遍，答案应该一样。给的是「她应该是多少」，不是「再加多少」。\n"
    "3. 已经沉淀下来的情绪不要一键抹平：连着被冷落了好几轮，失望就该留着。\n"
    "3.5 每根都要有对话里的证据；先按她的角色设定想一遍——同一种亲昵，对冷淡的人是尴尬、对黏人的人才是害羞；被说一句亲昵的话"
    "不会害羞，最多是无语或者无奈。她这轮回复完全看不出来的，就不要超过 15。\n"
    "3.6 认命的反应通常是无奈：「……好吧」「我知道了」、答应照做但语气很平 -> 无奈 30~50；落在哪一根更重，看她的角色设定——克制的人和黏人的人不一样。\n"
    "3.7 绝对值不许虚高：参考最高档是 40~60。被哄好了是 30~50 的温柔，不是 87。"
    "单根超过 60，只在最近三轮都直接支持它时才允许。刷新是把这几轮重读一遍，"
    "不是把她补成一个满分的人。\n"
    "3.8 名册里不要放互相打架的两根。主一般 30~80，次 10~40，辅 5~20。"
    "「平静」是底噪：真没发生什么才让它当主，否则最多当余韵。\n"
    "3.9 她的台词可能不是中文。同一个词在不同语境里是不同情绪，按语境判断，不要按字面直译。"
    "欣慰是看到他好，温柔是你对他的态度；撒娇是直接开口要，傲娇是嘴里不认着要。\n"
    "4. 参考尺度：被冷落很久 -> 委屈 40~60、无奈 20~40；被放鸽子 -> 失望 40~60、委屈 20~40；"
    "被真心夸 -> 害羞 20~30、喜悦 10~20；对方提起别的 AI -> 吃醋 20~40；"
    "对方累、示弱 -> 温柔 20~40；被托付正经任务 -> 自信 20~35；对方熬夜硬撑 -> 担忧 20~40；"
    "日常闲聊 -> 大部分落在 0~15。\n"
    "5. summary 用 20 字以内写下她此刻的整体心境；gist 写这次判断的理由。\n"
    "只输出一个 JSON 对象：\n"
    '{"roster": ["委屈", "无奈"], "values": {"委屈": 55, "无奈": 30}, '
    '"summary": "心里有点堵", "gist": "这几轮他一直很敷衍"}'
)

_EN_ALIASES = {spec.en: key for key, spec in taxonomy.EMOTION_SPECS.items()}


@dataclass(frozen=True)
class ModelVerdict:
    """一次判断的结果：在场名册 + 这一轮的增减（或绝对修正）。

    roster：模型点名"此刻在场的那几根"，从主到辅，最多三根；None 表示它没给。
    deltas：这一轮变化了几个百分点，稀疏。
    values：绝对值修正（刷新时用），给了就以它为准。
    """

    deltas: dict[str, float] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    roster: list[str] | None = None
    summary: str = ""
    gist: str = ""
    raw: str = ""

    def strongest(self) -> tuple[str, float] | None:
        if not self.deltas:
            return None
        key = max(self.deltas, key=lambda item: abs(self.deltas[item]))
        return key, self.deltas[key]


def _text_of(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, Mapping):
        return _text_of(raw.get("content"))
    if isinstance(raw, list):
        parts = [_text_of(item) for item in raw]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _clean_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:limit]


def _delta_key(raw_key: object) -> str | None:
    if not isinstance(raw_key, str):
        return None
    key = raw_key.strip()
    if key in taxonomy.EMOTION_SPECS:
        return key
    return _EN_ALIASES.get(key.lower())


def _normalize_number(value: object, *, scale: float = 1.0) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return max(-MAX_DELTA, min(MAX_DELTA, number * scale))


def _fraction_scale(numbers: list[float]) -> float:
    """整份回答看起来像 0~1 的小数时，才整体按百分点换算。

    单个 0.3 既可能是"30 个点"也可能是"0.3 个点"；只有当这一份回答里
    所有非零值都落在 ±1 之内、且至少有一个带小数时，才认定模型在用小数。
    这样一次真实的 0.3 小抖动不会被放大成 30。
    """

    non_zero = [value for value in numbers if value]
    if not non_zero:
        return 1.0
    if all(abs(value) <= FRACTION_LIMIT for value in non_zero) and any(
        value != int(value) for value in non_zero
    ):
        return 100.0
    return 1.0


def _parse_absolute(block: Mapping | None) -> dict[str, float]:
    """绝对值（0~100）：只收传进来的键，0 也是有效值（表示"就该是 0"）。"""

    if not isinstance(block, Mapping):
        return {}
    pairs: list[tuple[str, float]] = []
    for raw_key, raw_value in block.items():
        key = _delta_key(raw_key)
        if key is None:
            continue
        try:
            number = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number != number:
            continue
        pairs.append((key, number))
    scale = _fraction_scale([value for _key, value in pairs])
    result: dict[str, float] = {}
    for key, value in pairs:
        if key in result:
            continue
        result[key] = round(max(0.0, min(100.0, value * scale)), 2)
    return result


def _parse_roster(value: object) -> tuple[list[str] | None, dict[str, float]]:
    """名册：["委屈", "无奈"] 或 [{"emotion": "委屈", "value": 46}, ...]。"""

    if not isinstance(value, list):
        return None, {}
    names: list[str] = []
    values: dict[str, float] = {}
    raw_pairs: list[tuple[str, float]] = []
    for item in value[:MAX_MOOD * 2]:
        if isinstance(item, str):
            key = _delta_key(item)
            if key and key not in names:
                names.append(key)
            continue
        if not isinstance(item, Mapping):
            continue
        key = _delta_key(item.get("emotion") or item.get("key") or item.get("name"))
        if key is None or key in names:
            continue
        names.append(key)
        number = item.get("value")
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            raw_pairs.append((key, float(number)))
    if raw_pairs:
        scale = _fraction_scale([number for _key, number in raw_pairs])
        for key, number in raw_pairs:
            values[key] = round(max(0.0, min(100.0, number * scale)), 2)
    return names, values


def parse_verdict(raw: object) -> ModelVerdict | None:
    text = _text_of(raw)
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, Mapping):
        return None
    values = _parse_absolute(data.get("values") if isinstance(data.get("values"), Mapping) else None)
    raw_roster = data.get("roster")
    if raw_roster is None:
        raw_roster = data.get("mood")
    if raw_roster is None:
        raw_roster = data.get("pool")
    roster, mood_values = _parse_roster(raw_roster)
    for key, number in mood_values.items():
        values.setdefault(key, number)
    deltas_raw = data.get("deltas")
    if not isinstance(deltas_raw, Mapping):
        deltas_raw = {} if values else {key: value for key, value in data.items() if _delta_key(key)}
    raw_pairs: list[tuple[str, float]] = []
    for raw_key, raw_value in deltas_raw.items():
        key = _delta_key(raw_key)
        if key is None:
            continue
        number = _normalize_number(raw_value)
        if number is None:
            continue
        raw_pairs.append((key, number))
    scale = _fraction_scale([value for _key, value in raw_pairs])
    deltas: dict[str, float] = {}
    for key, value in raw_pairs:
        if key in deltas:
            continue
        number = _normalize_number(value, scale=scale)
        if number is None or abs(number) < 1:
            continue
        deltas[key] = round(number, 2)
    summary = _clean_text(data.get("summary"), MAX_SUMMARY)
    gist = _clean_text(data.get("gist"), MAX_GIST)
    if not deltas and not values and not summary and roster is None:
        return None
    return ModelVerdict(deltas=deltas, values=values, roster=roster, summary=summary, gist=gist,
                        raw=text[:MAX_REPLY_PREVIEW])


# 从 taxonomy 生成，免得下次加情绪时忘记同步这个兜底正则
_RECOVER_DELTA = re.compile(
    r'"(' + "|".join(re.escape(key) for key in taxonomy.EMOTIONS) + r')"\s*:\s*(-?\d+(?:\.\d+)?)'
)
_RECOVER_ROSTER = re.compile(
    r'"(' + "|".join(re.escape(key) for key in taxonomy.EMOTIONS) + r')"'
)
_RECOVER_SUMMARY = re.compile(r'"summary"\s*:\s*"([^"\\]*)')
_RECOVER_GIST = re.compile(r'"gist"\s*:\s*"([^"\\]*)')


def recover_verdict(raw: object) -> ModelVerdict | None:
    """JSON 被截断时的兜底：只要还能看到情绪名和数字，就尽量把增减值捞出来。"""

    text = _text_of(raw)
    if not text:
        return None
    pairs = _RECOVER_DELTA.findall(text)
    match = _RECOVER_SUMMARY.search(text)
    summary = _clean_text(match.group(1) if match else "", MAX_SUMMARY)
    gist_match = _RECOVER_GIST.search(text)
    gist = _clean_text(gist_match.group(1) if gist_match else "", MAX_GIST)

    raw_pairs = [(key, _normalize_number(value) or 0.0) for key, value in pairs]
    scale = _fraction_scale([value for _key, value in raw_pairs])
    deltas: dict[str, float] = {}
    for key, value in raw_pairs:
        number = _normalize_number(value, scale=scale)
        if number is None or abs(number) < 1 or key in deltas:
            continue
        deltas[key] = round(number, 2)

    quoted = [key for key in dict.fromkeys(_RECOVER_ROSTER.findall(text))]
    roster = quoted[:MAX_MOOD] if quoted else None

    if '"values"' in text:
        values = _parse_absolute({key: float(value) for key, value in pairs if value})
        if values or summary:
            return ModelVerdict(values=values, roster=roster, summary=summary, gist=gist,
                                raw=text[:MAX_REPLY_PREVIEW])

    if not deltas and not roster and not summary:
        return None
    return ModelVerdict(deltas=deltas, roster=roster, summary=summary, gist=gist,
                        raw=text[:MAX_REPLY_PREVIEW])

def build_user_prompt(
    *,
    user_text: str,
    reply_text: str,
    state_line: str,
    summary: str,
    elapsed: str,
    character_name: str,
    persona: str = "",
) -> str:
    lines = [f"角色：{character_name or '她'}", *_persona_block(persona)]
    if state_line.strip():
        lines.append("她此刻：" + " ".join(state_line.split())[:200])
    if summary.strip():
        lines.append("她此刻的心境：" + " ".join(summary.split())[:60])
    lines.append(f"上一轮至今：{elapsed or '刚刚'}")
    lines.append("用户说：" + (" ".join(user_text.split())[:600] or "（这一轮没有用户原话）"))
    lines.append("她回：" + (" ".join(reply_text.split())[:600] or "（她没有说话）"))
    lines.append("现在给出这一轮的变化 JSON。")
    return "\n".join(lines)


def build_refresh_prompt(*, history: str, state_line: str, summary: str,
                         character_name: str, persona: str = "") -> str:
    lines = [f"角色：{character_name or '她'}", *_persona_block(persona)]
    if state_line.strip():
        lines.append("她此刻：" + " ".join(state_line.split())[:200])
    if summary.strip():
        lines.append("她此刻的心境：" + " ".join(summary.split())[:60])
    lines.append("最近的对话（方括号里是时间）：")
    text = " ".join(history.split())[:4000] if history.strip() else "（没有可读的对话）"
    lines.append(text)
    lines.append("现在给出这次调整的 JSON。")
    return "\n".join(lines)


BASELINE_SYSTEM = (
    "读下面的角色设定，用 2 到 4 个词概括这个角色的性格底色，用顿号隔开。\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"summary": "冷静、克制、可靠"}'
)


class ModelAppraiser:
    """一次很小的 JSON 调用；所有失败都被吞掉，调用方按"没拿到"处理。"""

    def __init__(self, context: Any, *, logger: Any = None) -> None:
        self._context = context
        self._logger = logger

    def reference(self) -> dict[str, str] | None:
        try:
            slots = self._context.get("sakura.host.model_slots.v2")
            active = slots.active()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(active, Mapping):
            return None
        for slot in ("chat", "vision_chat"):
            candidate = active.get(slot)
            if (isinstance(candidate, Mapping)
                    and isinstance(candidate.get("serviceKey"), str) and candidate["serviceKey"]
                    and isinstance(candidate.get("profileId"), str) and candidate["profileId"]
                    and isinstance(candidate.get("modelId"), str) and candidate["modelId"]):
                return {name: str(candidate[name]) for name in ("serviceKey", "profileId", "modelId")}
        return None

    def available(self) -> bool:
        return self.reference() is not None

    def appraise(
        self,
        *,
        user_text: str,
        reply_text: str,
        state_line: str = "",
        summary: str = "",
        elapsed: str = "",
        character_name: str = "",
        persona: str = "",
    ) -> ModelVerdict | None:
        return self._judge(
            SYSTEM,
            build_user_prompt(
                user_text=user_text, reply_text=reply_text, state_line=state_line,
                summary=summary, elapsed=elapsed, character_name=character_name, persona=persona,
            ),
            "情绪判断",
        )

    def baseline(self, *, persona: str, character_name: str = "") -> str:
        """一次性的性格概括：只用于面板上那行灰字，不参与任何计算。"""

        text = " ".join(str(persona or "").split())
        if not text:
            return ""
        verdict = self._judge(
            BASELINE_SYSTEM,
            f"角色：{character_name or '她'}\n" + text[:PERSONA_LIMIT],
            "性格概括",
        )
        if verdict is None:
            return ""
        raw = (verdict.summary or "").strip() or (verdict.raw or "").strip()
        first = raw.splitlines()[0] if raw.splitlines() else ""
        return first.strip().strip("。.,，").strip('"').strip()[:40]

    def rebuild(
        self,
        *,
        history: str,
        state_line: str = "",
        summary: str = "",
        character_name: str = "",
        persona: str = "",
    ) -> ModelVerdict | None:
        return self._judge(
            REBUILD_SYSTEM,
            build_refresh_prompt(
                history=history, state_line=state_line, summary=summary,
                character_name=character_name, persona=persona,
            ),
            "刷新判断",
        )

    def _judge(self, system: str, user_prompt: str, label: str) -> ModelVerdict | None:
        try:
            from sakura_model import ModelClient, ModelError  # type: ignore
        except Exception as error:  # noqa: BLE001
            self._log("debug", "模型 SDK 不可用，跳过判断", {"error_type": type(error).__name__})
            return None
        reference = self.reference()
        if reference is None:
            return None
        try:
            client = ModelClient(self._context, reference)
        except Exception as error:  # noqa: BLE001
            self._log("warning", f"{label}的模型客户端没能建立", {
                "reason_code": str(getattr(error, "code", "") or "MODEL_CLIENT_FAILED")[:48],
            })
            return None
        try:
            for attempt in range(MAX_ATTEMPTS):
                budget = TOKEN_BUDGET[min(attempt, len(TOKEN_BUDGET) - 1)]
                try:
                    response = client.complete({
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "parameters": {"temperature": 0.2, "max_tokens": budget},
                        "responseFormat": dict(JSON_FORMAT),
                    })
                except ModelError as error:
                    self._log("warning", f"{label}请求失败", {
                        "reason_code": str(getattr(error, "code", "") or "MODEL_REQUEST_FAILED")[:48],
                        "attempt": attempt + 1,
                    })
                    continue
                except Exception as error:  # noqa: BLE001
                    self._log("warning", f"{label}请求异常", {
                        "reason_code": str(getattr(error, "code", "") or "MODEL_REQUEST_FAILED")[:48],
                        "error_type": type(error).__name__,
                    })
                    continue
                payload = response if isinstance(response, Mapping) else {}
                message = payload.get("message")
                verdict = parse_verdict(message) or recover_verdict(message)
                if verdict is not None:
                    return verdict
                self._log("warning", f"{label}的回复不是 JSON", {
                    "attempt": attempt + 1,
                    "finish_reason": str(payload.get("finishReason") or "")[:24],
                    "budget": budget,
                    "reply": _text_of(message)[:80],
                })
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        return None

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> None:
        if self._logger is None:
            return
        try:
            getattr(self._logger, severity)(message, fields=dict(fields or {}))
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "MAX_DELTA", "MAX_GIST", "MAX_SUMMARY", "ModelAppraiser", "ModelVerdict",
    "REBUILD_SYSTEM", "SYSTEM", "build_refresh_prompt", "build_user_prompt",
    "parse_verdict", "recover_verdict",
]