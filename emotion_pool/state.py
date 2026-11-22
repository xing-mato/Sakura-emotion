"""情绪数据与演化规则（v2：轮驱动，模型给增减值）。

和 v1 的区别：

- **没有时间衰减**。状态只在模型判断时变，半衰期、显著度、互斥表、惯性全部删掉。
- **百分比**。对外一律 0-100 的整数；内部存 0.05 这种浮点只是为了精度。
- **稀疏增减**。模型只说"委屈 +24、失望 +10"，没说的就是没变。
- **心境是一句话**，不是第二个物理层。

这一层仍然只有数据与纯函数，不碰宿主，便于离线自检。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from . import taxonomy

PRESENCE = 0.05          # 低于 5% 视为"不在场"
# 人的情绪结构：一根主、最多一根次、偶尔一点辅。
# 这三根是**表现层**——她会真的用它们说话。
#
# 表现层之外还可以有"余韵"：数值留着，但不表现。
# 理由很实在：一次对话里被拨动的东西往往不止三根，但超过三根一起往外冒，
# 她连一个清楚的反应都给不出来（倒反天罡）。所以：
#   表现层  3 根，决定语气
#   余韵    3 根，只当底色，永不进注入的主句
#   其余    清零
MAX_MOOD = 3        # 表现层名额
MAX_TRACKED = 6     # 池子总容量 = 表现 3 + 余韵 3
MOOD_ROLES = ("主", "次", "辅")
LEVEL_THRESHOLDS = (0.25, 0.55)   # 微弱 / 明显 / 强烈

MAX_PERCENT = 100.0
EPISODE_MIN_DELTA = 20.0  # 一次增减 ≥ 20 个百分点才算"被触动"
PENDING_LIMIT = 12
PENDING_TTL = 1800.0


def _epoch(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def clamp_value(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def percent_of(value: float) -> int:
    return int(round(clamp_value(value) * 100))


@dataclass
class Episode:
    """一次真正被她记住的触动。"""

    at: float
    emotion: str
    delta: float = 0.0        # 这次增减了几个百分点（正数向上）
    intensity: float = 0.0    # 记录当时的深浅（0~1），给记录表用
    source: str = "model"
    gist: str = ""
    trigger: str = ""
    turn_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": round(float(self.at), 3),
            "emotion": self.emotion,
            "delta": round(float(self.delta), 2),
            "intensity": round(clamp_value(self.intensity), 4),
            "source": self.source,
            "gist": self.gist[:160],
            "trigger": self.trigger[:160],
            "turn_id": self.turn_id[:64],
        }

    @classmethod
    def from_dict(cls, raw: object) -> "Episode":
        data = raw if isinstance(raw, Mapping) else {}
        emotion = data.get("emotion")
        intensity = data.get("intensity")
        if isinstance(intensity, (int, float)) and float(intensity) > 1.0:
            intensity = float(intensity) / 100.0     # v1 迁移：旧值是 0~100 的百分比
        return cls(
            at=_epoch(data.get("at")),
            emotion=emotion if emotion in taxonomy.EMOTIONS else "喜悦",
            delta=float(data.get("delta") or 0.0),
            intensity=clamp_value(intensity),
            source=str(data.get("source") or "model")[:16],
            gist=str(data.get("gist") or "")[:160],
            trigger=str(data.get("trigger") or "")[:160],
            turn_id=str(data.get("turn_id") or "")[:64],
        )


class Applied(dict):
    """apply_mood 的结果。

    本体是这个 dict（每根真实变化了几个百分点），另外挂一个 evicted：
    被名册挤出名额、整根清零的那几根。它们看着像暴跌，其实不是情绪变化——
    面板和涟漪都不该把它当情绪事件。
    """

    def __init__(self, values: object = None, *, evicted: object = ()) -> None:
        super().__init__(values or {})
        self.evicted = list(evicted or ())


@dataclass
class CharacterState:
    """一个角色的二十六根弦。"""

    values: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in taxonomy.EMOTIONS})
    summary: str = ""
    # 性格底色：模型从角色卡里提炼的两三个词，只给面板显示，不参与任何计算
    baseline: str = ""
    # 当前主情绪连续当了几轮：{"害羞": 6}
    main_streak: dict[str, int] = field(default_factory=dict)
    updated_at: float = 0.0
    turn_count: int = 0
    episodes: list[Episode] = field(default_factory=list)
    # 待判断的轮次：turn_id -> {"text": str, "at": float}
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 最近一次判断：{"at": float, "deltas": {情绪: 百分点}, "summary": str, "gist": str, "source": str}
    last_judge: dict[str, Any] | None = None
    # 上一轮后从长期记忆里捞到的往事
    memory_ctx: dict[str, Any] | None = None

    # ---- 变化 ----

    def apply_deltas(self, deltas: Mapping[str, float]) -> dict[str, float]:
        """按百分点增减；返回真正生效的那部分（夹紧到 0~100 之后）。"""

        applied: dict[str, float] = {}
        for key in taxonomy.EMOTIONS:
            raw = deltas.get(key)
            if raw is None:
                continue
            try:
                delta = float(raw)
            except (TypeError, ValueError):
                continue
            if delta != delta or abs(delta) < 0.01:
                continue
            before = clamp_value(self.values.get(key, 0.0))
            after = clamp_value(before + delta / 100.0)
            self.values[key] = after
            real = round((after - before) * 100.0, 2)
            if abs(real) >= 0.01:
                applied[key] = real
        return applied

    def apply_mood(
        self,
        roster: list[str] | None,
        deltas: Mapping[str, float] | None = None,
        values: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """换一张情绪谱：先在旧值上应用增减，再决定谁在场，最后把名册外的清零。

        roster 是模型点名"此刻在场的那几根"（从主到辅，最多三根）：
        - None  = 模型没说名册，沿用现在在场的（不清场）；
        - []    = 模型明确说"没有哪根在场"，全部归零；
        - [...] = 名册。名册外的情绪一律清零；写了增减值却不在名册里的会被补进来。

        清零是**结构性**的（人的情绪只有三个名额），不是衰减：谁进谁出由模型点名。
        返回每一根真正变化了多少个百分点。
        """

        before = {key: clamp_value(self.values.get(key, 0.0)) for key in taxonomy.EMOTIONS}
        after = dict(before)
        sparse = taxonomy.normalize_deltas(deltas)
        for key, delta in sparse.items():
            after[key] = clamp_value(before.get(key, 0.0) + delta / 100.0)
        # 绝对值修正（刷新时用）：给了就以它为准，不再叠加增减
        absolute = dict(values or {})
        for key, value in absolute.items():
            if key in taxonomy.EMOTION_SPECS:
                after[key] = clamp_value(float(value) / 100.0)

        if roster is None:
            names = [key for key in taxonomy.EMOTIONS if before.get(key, 0.0) >= PRESENCE]
        else:
            names = [key for key in dict.fromkeys(roster) if key in taxonomy.EMOTION_SPECS]
        for key in (*sparse, *absolute):
            if key not in names:
                names.append(key)
        names = sorted(dict.fromkeys(names), key=lambda key: after.get(key, 0.0), reverse=True)
        keep = set(names[:MAX_TRACKED])

        evicted = [
            key for key in taxonomy.EMOTIONS
            if before.get(key, 0.0) >= PRESENCE and key not in keep
        ]
        applied: dict[str, float] = {}
        for key in taxonomy.EMOTIONS:
            value = after.get(key, 0.0) if key in keep else 0.0
            # 低于在场线的值既不进面板也不进提示词，留着只会占掉三个名额之一，
            # 形成「状态里有、界面没有、判断器也看不到」的幽灵值——直接清掉。
            if value < PRESENCE:
                value = 0.0
            self.values[key] = value
            diff = round((value - before.get(key, 0.0)) * 100.0, 2)
            if abs(diff) >= 0.01:
                applied[key] = diff
        return Applied(applied, evicted=evicted)

    def set_values(self, values: Mapping[str, float]) -> dict[str, float]:
        """绝对值写入（刷新用）：只改传进来的那几根，返回实际变化了多少个点。

        和 apply_deltas 的区别是它幂等——同一份判断算两遍，结果一样。
        """

        applied: dict[str, float] = {}
        for key in taxonomy.EMOTIONS:
            if key not in values:
                continue
            raw = values.get(key)
            try:
                target = clamp_value(float(raw) / 100.0)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            before = clamp_value(self.values.get(key, 0.0))
            self.values[key] = target
            diff = round((target - before) * 100.0, 2)
            if abs(diff) >= 0.01:
                applied[key] = diff
        return applied

    def note_main(self, key: str | None) -> None:
        """记下这一轮的主情绪。换人就从 1 重新数。"""

        if not key:
            self.main_streak = {}
            return
        self.main_streak = {key: int(self.main_streak.get(key, 0)) + 1}

    def streak_of(self, key: str) -> int:
        return int(self.main_streak.get(key, 0))

    def set_baseline(self, text: object) -> None:
        cleaned = " ".join(str(text or "").split())[:40]
        if cleaned:
            self.baseline = cleaned

    def set_summary(self, text: object) -> None:
        cleaned = " ".join(str(text or "").split())[:40]
        if cleaned:
            self.summary = cleaned

    # ---- 读数 ----

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(
            ((key, clamp_value(self.values.get(key, 0.0))) for key in taxonomy.EMOTIONS),
            key=lambda item: item[1],
            reverse=True,
        )

    def top(self, limit: int = 3, *, floor: float = PRESENCE) -> list[tuple[str, float]]:
        picked = [(key, value) for key, value in self.ranked() if value >= floor]
        return picked[:max(0, int(limit))]

    def calm(self) -> bool:
        return not self.top(1, floor=PRESENCE)

    def level_of(self, value: float) -> str:
        if value < PRESENCE:
            return "—"
        if value >= LEVEL_THRESHOLDS[1]:
            return "强烈"
        if value >= LEVEL_THRESHOLDS[0]:
            return "明显"
        return "微弱"

    # ---- 记录 ----

    def add_episode(self, episode: Episode, *, keep: int) -> int:
        self.episodes.append(episode)
        limit = max(0, int(keep))
        if limit == 0:
            self.episodes = []
            return -1
        if len(self.episodes) > limit:
            self.episodes = self.episodes[-limit:]
        return len(self.episodes) - 1

    def trim_pending(self) -> None:
        if len(self.pending) <= PENDING_LIMIT:
            return
        ordered = sorted(self.pending.items(), key=lambda item: float(item[1].get("at", 0.0)))
        for turn_id, _record in ordered[:-PENDING_LIMIT]:
            self.pending.pop(turn_id, None)

    def prune_pending(self, now: float, *, max_age_seconds: float = PENDING_TTL) -> None:
        """待判断的轮次只留半小时：原话片段不该在文件里越积越久。"""

        cutoff = float(now) - max(60.0, float(max_age_seconds))
        stale = [
            turn_id for turn_id, record in self.pending.items()
            if float(record.get("at", 0.0) or 0.0) < cutoff
        ]
        for turn_id in stale:
            self.pending.pop(turn_id, None)

    def last_episode(self, *, within_seconds: float, now: float) -> Episode | None:
        for episode in reversed(self.episodes):
            age = float(now) - float(episode.at)
            if 0 <= age <= within_seconds:
                return episode
        return None

    def episodes_recent(self, *, limit: int, within_seconds: float, now: float) -> list[Episode]:
        picked = []
        for episode in reversed(self.episodes):
            age = float(now) - float(episode.at)
            if 0 <= age <= within_seconds:
                picked.append(episode)
                if len(picked) >= limit:
                    break
        return list(reversed(picked))

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {key: round(clamp_value(value), 4) for key, value in self.values.items()},
            "summary": self.summary,
            "baseline": self.baseline,
            "main_streak": {key: int(value) for key, value in self.main_streak.items()},
            "updated_at": round(float(self.updated_at), 3),
            "turn": int(self.turn_count),
            "episodes": [item.to_dict() for item in self.episodes],
            "pending": {
                turn_id: {
                    "text": str(record.get("text") or "")[:600],
                    "at": round(float(record.get("at", 0.0)), 3),
                }
                for turn_id, record in self.pending.items()
            },
            "last_judge": self.last_judge,
            "memory_ctx": self.memory_ctx,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "CharacterState":
        data = raw if isinstance(raw, Mapping) else {}
        raw_values = data.get("values")
        if not isinstance(raw_values, Mapping):
            raw_values = data.get("pools") if isinstance(data.get("pools"), Mapping) else {}
        values = taxonomy.empty_pools()
        for key in taxonomy.EMOTIONS:
            values[key] = clamp_value(raw_values.get(key))
        # 旧存档可能同时挂着十几根弦：读进来先按容量收口（表现三根 + 余韵三根）
        for key, _value in sorted(values.items(), key=lambda item: item[1], reverse=True)[MAX_TRACKED:]:
            values[key] = 0.0
        episodes = []
        if isinstance(data.get("episodes"), list):
            episodes = [Episode.from_dict(item) for item in data["episodes"]]
        pending: dict[str, dict[str, Any]] = {}
        raw_pending = data.get("pending")
        if isinstance(raw_pending, Mapping):
            for turn_id, record in list(raw_pending.items())[-PENDING_LIMIT:]:
                if not isinstance(record, Mapping):
                    continue
                pending[str(turn_id)[:64]] = {
                    "text": str(record.get("text") or "")[:600],
                    "at": _epoch(record.get("at")),
                }
        summary = data.get("summary")
        if not isinstance(summary, str):
            summary = ""
        baseline = data.get("baseline")
        if not isinstance(baseline, str):
            baseline = ""
        streak: dict[str, int] = {}
        raw_streak = data.get("main_streak")
        if isinstance(raw_streak, Mapping):
            for key, value in list(raw_streak.items())[:4]:
                if key in taxonomy.EMOTION_SPECS and isinstance(value, (int, float)):
                    streak[str(key)] = max(0, int(value))
        last_judge = data.get("last_judge") or data.get("last_verdict")
        memory_ctx = data.get("memory_ctx")
        return cls(
            values=values,
            summary=summary[:40],
            baseline=baseline[:40],
            main_streak=streak,
            updated_at=_epoch(data.get("updated_at") or data.get("last_update")),
            turn_count=max(0, int(data.get("turn") or data.get("turn_count") or 0)),
            episodes=episodes,
            pending=pending,
            last_judge=dict(last_judge) if isinstance(last_judge, Mapping) else None,
            memory_ctx=dict(memory_ctx) if isinstance(memory_ctx, Mapping) else None,
        )


def describe(state: CharacterState) -> dict[str, Any]:
    """主 / 次 / 辅 三个位置。人在某一刻通常只有一根主情绪，最多再有一根次、一点辅。"""

    ranked = [(key, value) for key, value in state.ranked() if value >= PRESENCE][:MAX_TRACKED]
    top = ranked[:MAX_MOOD]
    return {
        "calm": not top,
        "slots": [
            {
                "role": MOOD_ROLES[index],
                "key": key,
                "value": value,
                "level": state.level_of(value),
            }
            for index, (key, value) in enumerate(top)
        ],
        # 余韵：有数值、但不会成为主句里的情绪，只当底色
        "residual": [
            {"key": key, "value": value, "level": state.level_of(value)}
            for key, value in ranked[MAX_MOOD:]
        ],
        "summary": state.summary,
    }


__all__ = [
    "CharacterState", "EPISODE_MIN_DELTA", "Episode", "LEVEL_THRESHOLDS", "MAX_MOOD",
    "MAX_PERCENT", "MAX_TRACKED", "MOOD_ROLES", "PENDING_LIMIT", "PENDING_TTL", "PRESENCE",
    "clamp_value", "describe", "percent_of",
]