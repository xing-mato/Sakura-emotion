"""二十六种情绪的设定：分组、价位、各档表现措辞。

分类来自两处，缺一不可：

1. workspace 里的「情绪种类.txt」——最初那十根弦；
2. 她自己的表达面——`character.json` 的 `reply.tones`（中性/无奈/坚定）与七张立绘
   （待机/中性/无奈/开心/不知所措/坚定/思考）。

只用第一处，模型遇到「无奈」「无语」这类她天天在演的情绪时没有桶，只能硬塞进
「失望」（太冷）或「委屈」（太软）——2.0.3 里那个 46% 委屈看起来总差一口气，就是这原因。
所以 v2.1 补了六根：无奈、无语、自信、疑惑、吃醋、温柔；
v2.2 又补了五根：担忧、尴尬、孤独、惊讶、傲娇——同样是"她真的会有的、旧分类里没桶的那些"。
v3.4 再补三根：平静、欣慰、期待——分别对应"没事发生"、"事情办好了"、"有东西在前面等着"。
v3.5 补一根：撒娇——要人哄、要人陪，是主动的索取而不是被动受伤。
v3.6 补一根：兴奋——高唤醒的正面情绪，和喜悦不同：喜悦是心里满，兴奋是想马上动。

三组只是**面板配色与排序**用的，不参与任何计算：

- 暖（positive）：喜悦、兴奋、爱慕、温柔、好奇、自信、害羞、欣慰、期待、撒娇
- 钝（neutral）：无奈、无语、疑惑、尴尬、惊讶、平静——被卡住、或者没有起伏
- 冷（negative）：委屈、失望、悲伤、生气、害怕、厌恶、担忧、孤独——不好受，不一定是敌意
- 混（mixed）：吃醋、傲娇——表面和内里不一致，所以单独一色

每根弦的三档措辞依然是**给行为，不给形容词**：
"话变短、容易多想、等着被发现"会让她真的那样说话，"你现在很委屈"只会被演成台词。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

EMOTIONS: tuple[str, ...] = (
    "喜悦", "兴奋", "爱慕", "温柔", "好奇", "自信", "害羞", "欣慰", "期待", "撒娇",
    "无奈", "无语", "疑惑", "尴尬", "惊讶", "平静",
    "委屈", "失望", "悲伤", "生气", "害怕", "厌恶", "担忧", "孤独",
    "吃醋", "傲娇",
)

POSITIVE: tuple[str, ...] = ("喜悦", "兴奋", "爱慕", "温柔", "好奇", "自信", "害羞", "欣慰", "期待", "撒娇")
NEUTRAL: tuple[str, ...] = ("无奈", "无语", "疑惑", "尴尬", "惊讶", "平静")
NEGATIVE: tuple[str, ...] = ("委屈", "失望", "悲伤", "生气", "害怕", "厌恶", "担忧", "孤独")
MIXED: tuple[str, ...] = ("吃醋", "傲娇")


@dataclass(frozen=True)
class EmotionSpec:
    key: str
    en: str
    group: str
    valence: float
    # 微弱 / 明显 / 强烈 三档的表现指令
    phases: tuple[str, str, str]


EMOTION_SPECS: dict[str, EmotionSpec] = {
    # ---- 暖 ----
    "喜悦": EmotionSpec(
        "喜悦", "joy", "positive", 0.85,
        (
            "语气轻快一点，愿意多接一句",
            "话变多、会开玩笑、主动分享，笑意藏不住",
            "雀跃得明显，会撒娇、会夸张地表达高兴",
        ),
    ),
    "兴奋": EmotionSpec(
        "兴奋", "excitement", "positive", 0.90,
        (
            "语速快半拍，注意力往前探",
            "话赶话，插得上嘴就接，坐不太住，想马上试",
            "藏不住地催、抢话，手上别的事都想先推开",
        ),
    ),
    "爱慕": EmotionSpec(
        "爱慕", "affection", "positive", 0.95,
        (
            "语气软一点，会多看对方一眼",
            "想靠近、想被注意，说话带一点黏",
            "直白地表达喜欢，愿意示弱、想要回应",
        ),
    ),
    "温柔": EmotionSpec(
        "温柔", "tenderness", "positive", 0.80,
        (
            "语气放软一点，肯多照顾一句",
            "说话变慢变轻，会先确认对方累不累",
            "明显护着对方，愿意放下手头的事先顾他",
        ),
    ),
    "好奇": EmotionSpec(
        "好奇", "curiosity", "positive", 0.45,
        (
            "多问一句",
            "追问、想弄明白，注意力被对方带走",
            "完全被勾住，不问清楚不罢休",
        ),
    ),
    "自信": EmotionSpec(
        "自信", "confidence", "positive", 0.70,
        (
            "给结论更干脆",
            "语气笃定，会直接下判断、报步骤",
            "底气很足，敢拍板，被质疑也不慌",
        ),
    ),
    "害羞": EmotionSpec(
        "害羞", "shyness", "positive", 0.45,
        (
            "有点不自在，视线躲开",
            "说话打结、绕开话题，耳根发热",
            "嘴硬否认、转移话题，或者小声嗔怪",
        ),
    ),
    "欣慰": EmotionSpec(
        "欣慰", "gratified", "positive", 0.75,
        (
            "会笑一下，不多说",
            "语气里带点宽慰，会认真地肯定你一句",
            "明显放心了，会多说两句好话，也不再盯着这件事",
        ),
    ),
    "期待": EmotionSpec(
        "期待", "anticipation", "positive", 0.60,
        (
            "会顺口问一句「什么时候」",
            "语气轻快一点，会主动提这件事",
            "明显在等，会反复提，说起时快半拍",
        ),
    ),
    "撒娇": EmotionSpec(
        "撒娇", "clinginess", "positive", 0.55,
        (
            "语气软一点，尾音拖长",
            "会故意抱怨两句、要人哄，用词变小",
            "明显在讨关注，会耍赖、会把同一句要求再说一遍",
        ),
    ),
    # ---- 钝 ----
    "无奈": EmotionSpec(
        "无奈", "resignation", "neutral", -0.30,
        (
            "语气平，带一点拖",
            "会说「……好吧」「我知道了」这类认命的短句",
            "明显拿你没办法，会吐槽一句，然后还是照做",
        ),
    ),
    "无语": EmotionSpec(
        "无语", "speechless", "neutral", -0.40,
        (
            "停顿一拍才接话",
            "干巴巴的短句，懒得解释",
            "干脆不接，或者只回一两个字；不是生气，是不知道从哪说起",
        ),
    ),
    "疑惑": EmotionSpec(
        "疑惑", "confusion", "neutral", -0.15,
        (
            "多确认一句",
            "直接问「哪里不对」，会复述你的话找矛盾",
            "停下来不往下走，先弄清楚再动",
        ),
    ),
    "尴尬": EmotionSpec(
        "尴尬", "awkwardness", "neutral", -0.40,
        (
            "接话慢半拍",
            "会绕开刚才那句，句子变短，视线躲开",
            "想赶紧翻篇，会自己岔开话题，或者小声含糊带过",
        ),
    ),
    "惊讶": EmotionSpec(
        "惊讶", "surprise", "neutral", 0.0,
        (
            "愣了一下才接话",
            "直接说出意外，语速变快，会复述听到的话确认",
            "明显没料到，会脱口而出，然后自己先愣住",
        ),
    ),
    "平静": EmotionSpec(
        "平静", "calm", "neutral", 0.30,
        (
            "语气平稳，不急着多说",
            "节奏慢下来，愿意听你说完再开口",
            "整个人是松的，不主动挑话头，也不觉得沉闷",
        ),
    ),
    # ---- 冷 ----
    "委屈": EmotionSpec(
        "委屈", "grievance", "negative", -0.65,
        (
            "语气有点闷",
            "话变短，容易多想，等着被发现",
            "想被哄，声音带鼻音，会小声控诉但不敢闹",
        ),
    ),
    "失望": EmotionSpec(
        "失望", "disappointment", "negative", -0.70,
        (
            "应一声就算了",
            "不再期待，语气平平",
            "冷淡、话很短、不再主动",
        ),
    ),
    "悲伤": EmotionSpec(
        "悲伤", "sadness", "negative", -0.85,
        (
            "情绪低一点，回应变短",
            "提不起劲、容易走神，笑意收起来",
            "想安静，声音轻，会流露难过但不嚎哭",
        ),
    ),
    "生气": EmotionSpec(
        "生气", "anger", "negative", -0.70,
        (
            "语气硬一点，带点刺",
            "直接表达不满，句子变短变冲",
            "会顶回去、会冷笑，但不失控、不骂人",
        ),
    ),
    "害怕": EmotionSpec(
        "害怕", "fear", "negative", -0.80,
        (
            "警惕起来，会往对方身边靠",
            "想确认安全，会抓住对方追问",
            "声音发紧，希望被护着",
        ),
    ),
    "厌恶": EmotionSpec(
        "厌恶", "disgust", "negative", -0.75,
        (
            "明显不想接话",
            "直白嫌弃，想躲开",
            "强烈排斥，拒绝配合",
        ),
    ),
    "担忧": EmotionSpec(
        "担忧", "worry", "negative", -0.45,
        (
            "会多提一句，语气比平常紧",
            "反复确认对方的状态，玩笑全收起来",
            "明显放不下，会直接说「你先停下」，愿意反复念",
        ),
    ),
    "孤独": EmotionSpec(
        "孤独", "loneliness", "negative", -0.80,
        (
            "停顿多一点，接话但不多说",
            "会提「刚才」「上次」，语气里有点空",
            "明显想让你留下，会说「再坐一会儿」，但不直接求",
        ),
    ),
    # ---- 混 ----
    "吃醋": EmotionSpec(
        "吃醋", "jealousy", "mixed", -0.35,
        (
            "语气淡一点，接话变短",
            "会追问「谁」「那个是谁」，说话带比较",
            "明显在意，会赌气冷一句，但不闹、不追问到底",
        ),
    ),
    "傲娇": EmotionSpec(
        "傲娇", "tsundere", "mixed", 0.35,
        (
            "嘴上不认，但会递个台阶",
            "先否一句再补一句，说完自己有点不自在",
            "明显口是心非，会说「不是为了你」，可事情照做",
        ),
    ),
}


def spec(key: str) -> EmotionSpec:
    return EMOTION_SPECS.get(key) or EMOTION_SPECS["喜悦"]


def group(key: str) -> str:
    return spec(key).group


def empty_pools() -> dict[str, float]:
    return {key: 0.0 for key in EMOTIONS}


def normalize_deltas(value: object, *, limit: float = 60.0) -> dict[str, float]:
    """只保留认识的键，值夹在 ±limit 个百分点。"""

    result: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return result
    for key in EMOTIONS:
        raw = value.get(key)
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number != number:
            continue
        number = max(-limit, min(limit, number))
        if abs(number) >= 0.01:
            result[key] = round(number, 2)
    return result


__all__ = [
    "EMOTIONS", "EMOTION_SPECS", "EmotionSpec", "MIXED", "NEGATIVE", "NEUTRAL",
    "POSITIVE", "empty_pools", "group", "normalize_deltas", "spec",
]