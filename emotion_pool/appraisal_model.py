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
# 先判后答之后，一次截断就等于"这一轮没有情绪"（还占掉 12 秒里的几秒）。
# 起点给到 2800：max_tokens 是上限不是目标，正常调用不会因此变慢，但明显少掉一轮重试。
TOKEN_BUDGET = (2800, 4000)
MAX_DELTA = 60.0
MAX_SUMMARY = 20
MAX_GIST = 40
MAX_REPLY_PREVIEW = 200
JSON_FORMAT = {"type": "json_object"}
FRACTION_LIMIT = 1.0
# 人设块的上限。runtime 已经按段落挑过（约 1143 字），这里只是最后一道保险——
# 以前这里是 600，结果把「不热烈撒娇」那句又切掉了，等于改了个寂寞。
# 和 runtime.PERSONA_BUDGET 保持一致：两个上限不一致时，真正生效的是小的那个，
# 前面裁多久都是白裁。放开到 20000 就表示"判断器这一侧不再截断"。
PERSONA_LIMIT = 20000


def _persona_block(persona: str) -> list[str]:
    """她的角色设定节选：没有这层，判断会退化成"通用陪伴角色的逻辑"。"""

    text = " ".join(str(persona or "").split())
    if not text:
        return []
    return ["角色设定（节选，判断时必须符合她的性格）：", text[:PERSONA_LIMIT]]


SYSTEM = (
    "你是一个角色情绪的评估器。她的情绪由你判断：插件不预设任何一根弦会自动涨落，也不按时间衰减——"
    "每一轮的值都是你重新判断出来的。\n"
    "\n"
    "判断固定走三步，顺序不能反：\n"
    "  第一步  这一轮，她的处境变了什么？他给了她什么、拿走了什么、她还在等什么，还是这一轮什么都没变？\n"
    "  第二步  这些变化，落在她这个人身上是几分？同一件事，冷淡的人给 5，黏人的人给 20——"
    "这一步只看她的角色设定和她这几轮的来路，不看他这句话本身有多重。\n"
    "  第三步  才轮到弦的名字：这么重的分量、这个方向，用哪几根弦来装。\n"
    "先想好用哪根弦、再回头去他的话里找证据，那是查表，不是判断。"
    "那张表会把她变成「看到累就固定递水、看到喜欢就固定亲吻、看到别的女孩就固定吃醋」的人，"
    "她卡里明写着不要这样。刺激不是反应，反应才是反应。\n"
    "\n"
    "每写一根非零的弦，你都要能回答一句「她自己在哪里」——指得出她说过的话、做过的事，或被触到的那条设定。"
    "有没有只看三处：她自己的表现；他的处境真的变了；她设定里写明的反应方式被触到了"
    "（只限「她会怎么对待他」，不包括「她对他提到的人是什么感觉」——那些人不在场）。"
    "「他说了 X」「他提到了 Y」不是答案。人设只回答几分，不回答有没有。\n"
    "\n"
    "分量分四档：擦了一下 5~10；明明白白 10~20；压得住的 20~30；这一轮被掀翻 30~40。"
    "落在哪一档，看她在自己人设下受得住多少，不看他这句话有多重。"
    "定完再看结果落在哪：主一般 30~80，次 10~40，辅 5~20；越界就以形态为准，把这一轮的分量收回来。"
    "已经 85 以上的，只收不加——情绪几乎没有长期贴着天花板的时候。\n"
    "\n"
    "情绪不必「看起来合理」：喜极而泣、又气又怕都是真的。看着打架的两根，只要各自有证据就都留着，"
    "先看清哪一根更重。\n"
    "一个人这一刻通常只有一根主情绪，最多再有一根次、一点辅。她也一样：\n"
    "  主：决定这一轮用什么语气说话、怎么取舍；\n"
    "  次：只影响措辞；\n"
    "  辅：几乎不表现，只是底色。\n"
    "池子里有二十六根弦可以选（每根 0~100）：\n"
    "  暖：喜悦、兴奋、爱慕、温柔、好奇、自信、害羞、欣慰、期待、撒娇\n"
    "  钝：无奈、无语、疑惑、尴尬、惊讶、平静\n"
    "  冷：委屈、失望、悲伤、生气、害怕、厌恶、担忧、孤独\n"
    "  混：吃醋、傲娇\n"
    "\n"
    "你要给两样东西：\n"
    "1. roster：**这一轮要表现出来的**情绪，从主到辅，最多三根。\n"
    "   池子里总共还能留六根（三根表现 + 三根余韵）；名册之外的会清零，"
    "写了增减值却不在名册里的会作为「余韵」留着——有数值，但不表现，所以最高只留到 15%。"
    "一次对话被拨动的东西往往不止三根，余量放在余韵里，别硬塞进名册；"
    "真到了 15% 以上，那它就该进名册，而不是继续挂在余韵里。\n"
    "   这一轮没什么新东西时，原样重复上一轮的名册和值，不要换人；真的空了就写 []。\n"
    "2. deltas：这一轮变化了几个百分点。稀疏，没提到的就是没变。单轮最大 ±40。\n"
    "\n"
    "规则：\n"
    "1. 日常寒暄、问问题、让她做事、纯信息交换：deltas 给 {}，名册照抄上一轮——"
    "情绪不会因为一句话没内容就消失。\n"
    "2. 弦的词典（只写每根弦是什么，不写「什么情况就给」——什么情况给，由上面三步推出来，不从这张表里查）：\n"
    "   喜悦 = 心里满、轻快；兴奋 = 往前冲（想立刻做点什么）；期待 = 有件事在前面等着；"
    "欣慰 = 看到他好（他办成了、听劝了、稳下来了）；\n"
    "   温柔 = 你对待他的态度：想照顾、语气放软；爱慕 = 心里靠向他；害羞 = 想被喜欢又怕被看穿；"
    "好奇 = 想知道；自信 = 心里有底；平静 = 底噪；\n"
    "   无奈 = 认了：不争了、照做，心里没平；无语 = 没眼看；疑惑 = 没搞懂；"
    "尴尬 = 卡住了想翻篇；惊讶 = 意料之外，然后很快过去；\n"
    "   委屈 = 已经受了伤，说不出口；失望 = 盼过的事没成；悲伤 = 出了事；生气 = 往外推；"
    "害怕 = 替自己怕；厌恶 = 想躲开；担忧 = 替对方悬着心；孤独 = 没有人在；\n"
    "   吃醋 = 替自己不甘：她在这段关系里被排到后面了（只是替他难受，那是担忧）；"
    "傲娇 = 嘴上不认、人没走开（方向是靠近，生气的方向是推开）。\n"
    "3. 「撒娇」和「傲娇」看行为，不看心情——这两个名字底下是两种做法，不是两种感受：\n"
    "    撒娇 = 行为：她开口要了点什么。说法不限——「再叫一次」「别走」「陪我一会儿」都算，"
    "绕个弯要人留下、把话说到一半等他接，也算。只要她这一轮是在要，就是撒娇。"
    "心软不是撒娇，心软是温柔或害羞。没有在要，就别给；已经挂在那里的，写负数收下来。\n"
    "    傲娇 = 行为：她否认了，人没走开。也不止一种说法——嘴硬着把事做了、说「随你」却赖着不动，"
    "都算。否认之后真的冷下来、不接话，那是生气；被夸得说不出话，那是害羞。\n"
    "    这两根可以同时有：一句「才不要」，下一句又要他留下，就是都有。先看清哪一句更重。\n"
    "4. 情绪是连续的。她本来就在生气，对方只是说了句平常话，给 {} 或很小的负数，不要归零。\n"
    "5. 情绪会退，退的依据是证据不是数值：某一根这一轮找不出任何支持它的证据，就写负数把它降下来，"
    "降到 15 以下就算收干净了，不必一次归零——归零是名册的事。"
    "只是挂着、这一轮根本没被拨动的，尤其要收：数值再小，挂着不动也是假的。\n"
    "6. 「平静」是底噪，不是反应：这一轮真的什么都没发生时它可以当主；"
    "但只要还有别的情绪在场，主就不该是它，平静最多留在余韵里。\n"
    "7. 状态行里写着「已连续 N 轮为主」的，这一轮必须正面处理：对话里还找得到支撑它的证据吗？"
    "找不到就写负数把它降下来——一直挂在主位会让她一直端着同一个表情。\n"
    "8. 只写你真的看到的。每写一根弦，你都要能指出对话里的哪一句是证据；说不出就不要写，"
    "没把握的宁小勿大。唯一的例外：对方嘴上说没事、但语气或内容明显不对时，以你的判断为准。\n"
    "9. 她的台词可能是日语或其他语言。同一个词在不同语境里是不同情绪——"
    "「恥ずかしい」被夸的时候是害羞，做错事被拆穿的时候是尴尬；"
    "「ごめん」可能是不好意思，也可能是心凉。按语境判断，不要按字面直译。\n"
    "10. summary 用 20 字以内、第三人称，写下她此刻是什么感觉，例如「被放了鸽子，堵着」。"
    "同一个说法最多沿用一轮：感受没变就换个角度或换一组证据写"
    "（「被说笨，嘴硬心不服」→「不服气，偏又挑不出话」）。\n"
    "11. gist 用 20 字以内写下这一轮为什么让她动了情绪。\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"roster": ["委屈", "无奈"], "deltas": {"委屈": 24, "无奈": 8}, '
    '"summary": "被放了鸽子，堵着", "gist": "他说好陪她，又要走"}'
);


RECHECK_SYSTEM = (
    "你是同一个角色情绪评估器的第二步。第一步在她开口之前跑，只看到用户的话，给了一组预估值；"
    "现在她说了什么、每一段用什么语气，你都看得到——用这份证据去修正它。\n"
    "她的话按段给你，每段前面的方括号里是语气标签（中性/不满/害羞/请求/惊讶……）。\n"
    "\n"
    "规则：\n"
    "1. 先读轨迹：她这几段从什么变成什么。一句一句看，不要平均成一种情绪。\n"
    "2. 段里明摆着的情绪要给足。她写着「有点高兴」，那害羞或欣慰就该涨；"
    "她压着说「可别只是嘴上说说」，那是傲娇或者不满，不是温柔。\n"
    "3. 段里没有的情绪要收。某一根这一轮完全没演出来，就写足够大的负数把它压到 15 以下——"
    "允许大改，单根绝对值不超过 40。\n"
    "4. 她这几段是证据，第一步是猜的。证据和猜测冲突，以证据为准；能推翻就推翻，"
    "不要为了让两次判断看起来对得上，把差值留在 25 以内——那样校正只是给猜测做个样子。\n"
    "5. 别把语气标签直接当情绪名：中性不等于平静，不满不等于生气，看它落在哪句话上。\n"
    "6. 撒娇、傲娇、吃醋，依据是她这几段里实际做了什么：她要了、她否认了又没走开、"
    "她做了比较说了酸话。说法不限，别按句式认；他提了谁，只能算刺激。\n"
    "7. 情绪涨到高处要停：85 以上的这一轮只许收；70 以上的，没有新的、明确的证据就不再往上加。"
    "主一般在 30~80，一直往上涨不是深情，是失真。\n"
    "\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"deltas": {"害羞": 14, "无奈": -18}, "summary": "被承诺了，高兴又要装作不在乎", '
    '"gist": "第2段明显高兴，第3段是嘴上的警告"}'
)


REBUILD_SYSTEM = (
    "你是一个角色情绪的评估器。这一次不是新事件，而是让她重新读一遍最近的对话，"
    "给出她此刻**应该是多少**。\n"
    "人在某一刻通常只有一根主情绪，最多再有一根次、一点辅。情绪也不必「看起来合理」："
    "喜极而泣、又气又怕都是真的，看着打架的两根只要各自有证据就都留着。\n"
    "池子里有二十六根弦（每根 0~100）：\n"
    "  暖：喜悦、兴奋、爱慕、温柔、好奇、自信、害羞、欣慰、期待、撒娇\n"
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
    "4. 每根都要有对话里的证据；先按她的角色设定想一遍——同一种亲昵，对冷淡的人是尴尬、"
    "对黏人的人才是害羞；被说一句亲昵的话不会害羞，最多是无语或者无奈。"
    "她这几轮完全看不出来的，就不要超过 15。\n"
    "5. 绝对值不许虚高：参考最高档是 40~60。被哄好了是 30~50 的温柔，不是 87。"
    "单根超过 60，只在最近三轮都直接支持它时才允许。刷新是把这几轮重读一遍，不是把她补成满分的人。\n"
    "6. 形态：主一般 30~80，次 10~40，辅 5~20。「平静」是底噪：真没发生什么才让它当主，"
    "否则最多当余韵。\n"
    "7. 她的台词可能不是中文。同一个词在不同语境里是不同情绪，按语境判断，不要按字面直译。"
    "撒娇是她开口要了、傲娇是她否认了又没走开，两样可以同时有。\n"
    "8. 看的是她这几轮被放在什么位置上，不是他用了什么词：被冷落、被托付、被放鸽子、被真心夸，"
    "这是一类；他提到了谁、说了哪个词，那是另一类——后者本身不构成任何一根弦。"
    "分量怎么定：日常闲聊大部分落在 0~15；真被压到的事 40~60；被哄回来 30~50。\n"
    "9. summary 用 20 字以内写下她此刻的整体心境；gist 写这次判断的理由。"
    "同一个说法最多沿用一轮，第三轮起必须换角度或换证据。\n"
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
    her_recent: str = "",
) -> str:
    lines = [f"角色：{character_name or '她'}", *_persona_block(persona)]
    if state_line.strip():
        lines.append("她此刻：" + " ".join(state_line.split())[:200])
    if summary.strip():
        lines.append("她此刻的心境：" + " ".join(summary.split())[:60])
    lines.append(f"上一轮至今：{elapsed or '刚刚'}")
    recent = " ".join(str(her_recent or "").split())[:600]
    if recent:
        lines.append("她前几轮自己说的（方括号里是她当时的语气）：")
        lines.append(recent)
    lines.append("用户说：" + (" ".join(user_text.split())[:600] or "（这一轮没有用户原话）"))
    spoken = " ".join(reply_text.split())[:600]
    if spoken:
        lines.append("她回：" + spoken)
    else:
        lines.append(
            "她还没开口。下面判断的是：这一轮落在她身上的东西是什么——"
            "她接下来会怎么说，由这个状态决定。"
        )
    lines.append("现在给出这一轮的变化 JSON。")
    return "\n".join(lines)


def build_recheck_prompt(
    *,
    character_name: str,
    persona: str,
    state_line: str,
    summary: str,
    user_text: str,
    reply_segments: list[Mapping[str, Any]],
    elapsed: str = "",
) -> str:
    lines = [f"角色：{character_name or '她'}", *_persona_block(persona)]
    if state_line.strip():
        lines.append("先判之后的她：" + " ".join(state_line.split())[:200])
    if summary.strip():
        lines.append("先判给的心境：" + " ".join(summary.split())[:60])
    if elapsed.strip():
        lines.append(f"上一轮至今：{elapsed}")
    lines.append("用户说：" + (" ".join(user_text.split())[:600] or "（没有原话）"))
    lines.append("她说的（按段，方括号里是语气标签）：")
    for item in reply_segments:
        tone = " ".join(str(item.get("tone") or "中性").split())[:8] or "中性"
        spoken = str(item.get("text") or item.get("zh") or "")
        japanese = str(item.get("ja") or "")
        body = spoken or japanese
        if spoken and japanese and japanese.strip() != spoken.strip():
            body = spoken + " ／ " + japanese
        lines.append(f"  [{tone}] " + (" ".join(body.split())[:240] or "（这一段没有文字）"))
    lines.append("现在给出这次修正的 JSON。")
    return "\n".join(lines)


REFRESH_PERSONA_LIMIT = 4000


def build_refresh_prompt(*, history: str, state_line: str, summary: str,
                         character_name: str, persona: str = "") -> str:
    # 刷新要重读十六轮对话，人设只留前面一段就够——整张卡进去会把这一次调用拖到半分钟以上。
    lines = [f"角色：{character_name or '她'}", *_persona_block(persona[:REFRESH_PERSONA_LIMIT])]
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


# 思考型模型会把预算全花在推理上：qwen3.7-flash 实测一次判断想 2,300 个 token、27 秒，
# 而先判只等 15 秒——等于每轮都白跑。判断不需要推理：读一遍、写几十个 token 就够。
def _thinking_off(thinking: bool) -> dict[str, Any] | None:
    """要不要在请求里关掉思考。只影响判断模型，聊天那条线不归它管。

    关（默认）：请求里带 enable_thinking=false。OpenAI 兼容端点对不认识的字段
    一般是忽略而不是报错（deepseek 实测直接放过），所以不需要按模型名挑。
    开：什么都不加，让模型自己决定。
    """

    return None if thinking else {"enable_thinking": False}


class ModelAppraiser:
    """一次很小的 JSON 调用；所有失败都被吞掉，调用方按"没拿到"处理。"""

    def __init__(self, context: Any, *, logger: Any = None,
                 own_reference: Any = None, thinking_mode: Any = None) -> None:
        self._context = context
        self._logger = logger
        # 判断模型要不要想（只影响判断，不影响聊天），由设置决定
        self._thinking_enabled = thinking_mode
        # 插件自己的模型槽（plugin:<id>:judge）。配了就用它，没配回落到对话模型——
        # 聊天要文笔，判断只要算得对，这两件事不该抢同一个模型。
        self._own_reference = own_reference

    def thinking_enabled(self) -> bool:
        if not callable(self._thinking_enabled):
            return False
        try:
            return bool(self._thinking_enabled())
        except Exception:  # noqa: BLE001
            return False

    def own_reference(self) -> dict[str, str] | None:
        if not callable(self._own_reference):
            return None
        try:
            value = self._own_reference()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(value, Mapping):
            return None
        picked = {name: str(value.get(name) or "") for name in ("serviceKey", "profileId", "modelId")}
        return picked if all(picked.values()) else None

    def reference(self) -> dict[str, str] | None:
        own = self.own_reference()
        if own is not None:
            return own
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

    def slot_label(self) -> str:
        """面板/日志里那一行：判断现在用的是哪个槽。"""

        own = self.own_reference()
        if own is not None:
            return "判断模型：" + own["modelId"][:40]
        return ""

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
        her_recent: str = "",
    ) -> ModelVerdict | None:
        return self._judge(
            SYSTEM,
            build_user_prompt(
                user_text=user_text, reply_text=reply_text, state_line=state_line,
                summary=summary, elapsed=elapsed, character_name=character_name, persona=persona,
                her_recent=her_recent,
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

    def recheck(
        self,
        *,
        user_text: str,
        reply_segments: list[Mapping[str, Any]],
        state_line: str = "",
        summary: str = "",
        elapsed: str = "",
        character_name: str = "",
        persona: str = "",
    ) -> ModelVerdict | None:
        """第二步：她说完之后，用她实际的表达修正预估值。"""

        return self._judge(
            RECHECK_SYSTEM,
            build_recheck_prompt(
                character_name=character_name, persona=persona, state_line=state_line,
                summary=summary, user_text=user_text, reply_segments=reply_segments,
                elapsed=elapsed,
            ),
            "回复校正",
        )

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
        extra = _thinking_off(self.thinking_enabled())
        if extra:
            self._log("info", f"{label}已关闭思考", {"model": str(reference.get("modelId") or "")[:40]})
        try:
            for attempt in range(MAX_ATTEMPTS):
                budget = TOKEN_BUDGET[min(attempt, len(TOKEN_BUDGET) - 1)]
                parameters: dict[str, Any] = {"temperature": 0.2, "max_tokens": budget}
                if extra:
                    parameters["extra_body"] = dict(extra)
                try:
                    response = client.complete({
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "parameters": parameters,
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
                # 抓现场：被模型服务拒掉时，要留下能拿去对账的东西——
                # 是哪一层拒的（HTTP 状态）、哪一次（重试序号）、请求有多大。
                finish = str(payload.get("finishReason") or "")[:24]
                diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
                fields = {
                    "attempt": attempt + 1,
                    "finish_reason": finish,
                    "budget": budget,
                    "http_status": str(diagnostics.get("httpStatus") or "")[:12],
                    "chars": len(str(system or "")) + len(str(user_prompt or "")),
                    "reply": _text_of(message)[:80],
                }
                if finish == "content_filter":
                    self._log("warning", f"{label}被模型服务拒绝（content_filter）", fields)
                else:
                    self._log("warning", f"{label}的回复不是 JSON", fields)
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