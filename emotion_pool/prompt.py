"""把情绪写成模型看得懂的一段话（v4：主 / 次 / 辅）。

人在某一刻只有一根主情绪，最多再有一根次、一点辅。所以注入的不再是一张百分比清单，
而是三个带角色的位置：

    【情绪】主·委屈 46%，次·无奈 22%，心境：被放了鸽子，堵着。

角色本身是有用信息：主决定语气与取舍，次只影响措辞，辅几乎不表现。
写作原则没变：**给行为，不给形容词**。
"""

from __future__ import annotations

from typing import Any, Mapping

from . import state as state_module
from . import taxonomy

RULE_BASE = (
    "（这是你此刻的底色，不是台词：主决定这一轮的语气和取舍，次只影响措辞，辅几乎不表现；"
    "「余韵」那些更不要表现，它们只是让底色厚一点。别把清单念出来，让它从语气里透出来。"
    "强度低的时候宁可不表现，也别为了演而演。）"
)

RULE_DETAILED = (
    "（细则：这轮回复如果分成多段，tone 与表情标签跟着此刻的主情绪走，"
    "但只能从角色允许的那些里选。对方的话如果真的强烈触动你，可以当场反应得比上面更明显——"
    "那是被说中了，不是切换人格。情绪的变化会由下一轮的判断记下来。）"
)

CALM_LINE = "此刻没有哪根弦在被拨动"

BUDGET_HINT = {"brief": 110, "standard": 200, "detailed": 430}
MAX_CHARS = {"brief": 110, "standard": 240, "detailed": 600}


def ago(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60:
        return "刚刚"
    if value < 3600:
        return f"{int(value // 60)} 分钟前"
    if value < 86400:
        return f"{int(value // 3600)} 小时前"
    return f"{int(value // 86400)} 天前"


def percent(value: float) -> int:
    return int(round(max(0.0, min(1.0, float(value))) * 100))


def _slots_text(slots: list[Mapping[str, Any]]) -> str:
    return "，".join(
        f"{item['role']}·{item['key']} {percent(item['value'])}%" for item in slots
    )


def _episode_line(episode: state_module.Episode, now: float) -> str:
    head = f"{ago(float(now) - float(episode.at))} {episode.emotion}"
    if abs(episode.delta) >= 1:
        head += f" {episode.delta:+.0f}"
    gist = (episode.gist or episode.trigger or "").strip()
    return f"{head}（{' '.join(gist.split())[:40]}）" if gist else head


def build(
    character_state: state_module.CharacterState,
    settings: Any,
    now: float,
    *,
    memory_line: str = "",
) -> str:
    detail = getattr(settings, "detail", "standard")
    if detail not in BUDGET_HINT:
        detail = "standard"
    described = state_module.describe(character_state)
    summary = str(described["summary"] or "").strip()

    if described["calm"]:
        line = f"【情绪】{CALM_LINE}，心境：{summary or '平稳'}。"
        if detail == "brief":
            return line
        lines = [line]
        if detail == "detailed":
            episodes = character_state.episodes_recent(limit=2, within_seconds=24 * 3600, now=now)
            if episodes:
                lines.append("最近的涟漪：" + "；".join(_episode_line(item, now) for item in episodes) + "。")
            if memory_line.strip():
                lines.append("相关往事：" + " ".join(memory_line.split())[:120] + "。")
            lines.append(RULE_DETAILED)
        else:
            lines.append("（平静不等于冷淡，也不是没有脾气。）")
        return _fit("\n".join(lines), detail)

    slots = described["slots"]
    first = "【情绪】" + _slots_text(slots)
    if summary:
        first += f"，心境：{summary}"
    first += "。"

    if detail == "brief":
        return _fit(first, detail)

    lines = [first]
    # 余韵：池子里还有值，但这一轮不该表现出来——写清楚，免得她拿它当第二根主情绪
    residual = described.get("residual") or []
    if residual:
        lines.append(
            "余韵：" + "、".join(f"{item['key']} {percent(item['value'])}%" for item in residual)
            + "（有，但不表现，只当底色。）"
        )
    if detail == "detailed":
        episodes = character_state.episodes_recent(limit=2, within_seconds=12 * 3600, now=now)
        if episodes:
            lines.append("最近的涟漪：" + "；".join(_episode_line(item, now) for item in episodes) + "。")
        if memory_line.strip():
            lines.append("相关往事：" + " ".join(memory_line.split())[:120] + "。")
        lines.append(RULE_DETAILED)
    else:
        lines.append(RULE_BASE)
    return _fit("\n".join(lines), detail)


def _fit(text: str, detail: str) -> str:
    limit = MAX_CHARS.get(detail, 240)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def status_now(character_state: state_module.CharacterState) -> dict[str, str]:
    """插件设置页里「此刻的她」。"""

    described = state_module.describe(character_state)
    summary = str(described["summary"] or "").strip()
    if described["calm"]:
        return {
            "state": "neutral",
            "label": "平静",
            "message": (summary or "没有哪根弦在被拨动")[:240],
        }
    slots = described["slots"]
    head = slots[0]
    message = " · ".join(filter(None, [_slots_text(slots), summary]))[:240]
    return {
        "state": "warning" if head["value"] >= 0.55 else "ready",
        "label": f"{head['role']}·{head['key']}"[:120],
        "message": message,
    }


def status_last(character_state: state_module.CharacterState, now: float) -> dict[str, str]:
    """插件设置页里「最近一次判断」。"""

    verdict = character_state.last_judge
    if not isinstance(verdict, dict):
        return {"state": "neutral", "label": "还没有", "message": "在下一轮回复之后会出现"}
    deltas = verdict.get("deltas") if isinstance(verdict.get("deltas"), Mapping) else {}
    has_applied = isinstance(verdict.get("applied"), Mapping)
    applied = verdict["applied"] if has_applied else {}
    evicted = verdict.get("evicted") if isinstance(verdict.get("evicted"), list) else []
    # 显示"真正落地"的变化，不是模型写的原始增减：
    # 低于 5% 在场线的增减会被清零（幽灵值），写进这一行就会出现"说了 +4 面板却看不见"。
    # 被挤出名额的那几根也不算情绪变化。
    source_row = applied if has_applied else deltas
    changed = sorted(
        ((key, float(value)) for key, value in source_row.items()
         if key in taxonomy.EMOTIONS and abs(float(value)) >= 1 and key not in evicted),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    parts = [f"{key} {value:+.0f}" for key, value in changed[:3]]
    ghost = [
        key for key, value in deltas.items()
        if key in taxonomy.EMOTIONS and abs(float(value)) >= 1 and key not in applied
    ]
    at = float(verdict.get("at") or 0.0)
    age = ago(now - at) if at else ""
    if parts:
        label = "、".join(parts)
    elif ghost:
        # 有增减，但都低于 5% 在场线，落地时被清零了——说清楚，别让人以为是面板没刷新
        label = f"没有变化（{len(ghost)} 根低于 5% 在场线）"
    else:
        label = "没有变化"
    gist = str(verdict.get("summary") or verdict.get("gist") or "").strip()
    source = "刷新" if verdict.get("source") == "refresh" else "这一轮回复"
    message = "；".join(part for part in (age, gist[:140], f"来自{source}") if part)[:240]
    return {"state": "ready", "label": label[:120], "message": message or "只记录了变化量"}


__all__ = [
    "BUDGET_HINT", "CALM_LINE", "MAX_CHARS", "RULE_BASE", "RULE_DETAILED",
    "ago", "build", "percent", "status_last", "status_now",
]
