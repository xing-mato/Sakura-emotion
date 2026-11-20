"""运行时：以「轮」为节拍，让模型决定她的情绪怎么变。

一次对话的经过（v2）：

    用户说话
      │
      ├─ 注入（同步、只读）
      │     把此刻的百分比写成一段话塞进提示词；记下这一轮的原话，等回复后判断
      │
      └─ 她回复完成（sakura.host.chat.completed）
            后台线程读回原话 → 一次模型判断 → 稀疏增减值落到池子 → 下一轮生效
            间距很久没说话时，判断的输入里会多一行「上一轮至今：72 小时」，由模型决定怎么办

没有时间衰减，没有关键词，没有惯性阈值。状态只在判断时变；判断失败就这一轮不动。
所有入口都是"失败即静默"：异常只写日志，绝不抛回提示词组装路径。
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Mapping

from . import prompt as prompt_module
from . import state as state_module
from . import taxonomy

SOURCE_LABELS = {"model": "模型", "refresh": "刷新", "manual": "手动"}
MEMORY_MIN_DELTA = 40.0
MEMORY_WRITE_MIN_GAP = 300.0
MEMORY_WRITE_MIN_TURNS = 3
MEMORY_CTX_TTL = 480.0
MAX_REPLY_CHARS = 600
HISTORY_MAX_BYTES = 2 * 1024 * 1024
HISTORY_RECORD_CHARS = 4000
PLUGIN_VERSION = "1.0.0-beta"
# 人设：整张卡都交给判断器，只做总长保护。
# 挑段落等于替角色决定「哪段不重要」，换个卡（段落标题不一样）就会漏掉关键信息。
PERSONA_BUDGET = 2000
_SECTION = re.compile(r"【([^】]{1,20})】")


class EmotionRuntime:
    def __init__(
        self,
        *,
        settings_provider: Callable[[], Any],
        store: Any,
        logger: Any = None,
        timeline: Any = None,
        appraiser: Any = None,
        memory: Any = None,
        character_provider: Callable[[], str | None] | None = None,
        persona_provider: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings_provider = settings_provider
        self._store = store
        self._logger = logger
        self._timeline = timeline
        self._appraiser = appraiser
        self._memory = memory
        self._character_provider = character_provider
        self._persona_provider = persona_provider
        self._clock = clock

        self._lock = threading.RLock()
        self._characters: dict[str, state_module.CharacterState] | None = None
        self._names: dict[str, str] = {}
        self._persona_cache: tuple[str, float] = ("", 0.0)
        self._baseline_tried: dict[str, float] = {}
        self._last_character_id = ""
        self._last_memory_write = 0.0
        self._last_memory_turn = -999
        self._last_recall_query = ""

        self._history_path = _history_path(getattr(store, "path", None))
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._closed = False

    # ---- 设置 / 状态 ----

    def _settings(self) -> Any:
        try:
            return self._settings_provider()
        except Exception:  # noqa: BLE001
            from .settings import parse
            return parse({})

    def _states(self) -> dict[str, state_module.CharacterState]:
        if self._characters is None:
            try:
                self._characters = self._store.load()
            except Exception:  # noqa: BLE001
                self._characters = {}
        return self._characters

    def _state(self, character_id: str) -> state_module.CharacterState:
        states = self._states()
        key = character_id or "default"
        if key not in states:
            states[key] = state_module.CharacterState()
        return states[key]

    def _save(self) -> None:
        if self._characters is None:
            return
        try:
            self._store.save(self._characters)
        except Exception:  # noqa: BLE001
            pass

    def _persona(self) -> str:
        """她的角色设定节选：判断必须按她的性格，而不是通用陪伴逻辑。"""

        text, at = self._persona_cache
        if text and (self._clock() - at) < 600:
            return text
        excerpt = ""
        provider = self._persona_provider
        if provider is not None:
            try:
                value = provider()
            except Exception:  # noqa: BLE001
                value = ""
            if isinstance(value, str):
                excerpt = _persona_excerpt(value)
        self._persona_cache = (excerpt, float(self._clock()))
        return excerpt

    def _current_character_id(self) -> str:
        if self._character_provider is not None:
            try:
                value = self._character_provider()
            except Exception:  # noqa: BLE001
                value = None
            if isinstance(value, str) and value:
                return value
        return self._last_character_id or "default"

    # ---- 注入 ----

    def fragment_for(self, request: object) -> list[dict[str, Any]]:
        try:
            return self._fragment_for(request)
        except Exception as error:  # noqa: BLE001
            self._log("warning", "情绪注入失败，这一轮按没有情绪处理", {
                "error_type": type(error).__name__,
            })
            return []

    def _fragment_for(self, request: object) -> list[dict[str, Any]]:
        if self._closed or not isinstance(request, Mapping):
            return []
        settings = self._settings()
        if not settings.enabled:
            return []
        now = float(self._clock())
        character_id = _text(request.get("character_id")) or "default"
        character_name = _text(request.get("character_name"))
        text = _text(request.get("current_input"))
        turn_id = _text(request.get("current_turn_id"))
        source = _text(request.get("source")) or "chat"

        with self._lock:
            state = self._state(character_id)
            self._last_character_id = character_id
            if character_name:
                self._names[character_id] = character_name[:64]
            state.prune_pending(now)
            if source == "chat" and text and turn_id and turn_id not in state.pending:
                state.pending[turn_id] = {"text": " ".join(text.split())[:600], "at": now}
                state.trim_pending()
                self._log("debug", "记下这一轮，等她回复后判断", {"turn": state.turn_count, "pending": len(state.pending)})
            content = prompt_module.build(
                state, settings, now,
                memory_line=self._memory_line(state, now, settings),
            )
            self._save()

        return [{
            "id": "emotion_state",
            "content": content,
            "priority": 70,
            "budgetHint": prompt_module.BUDGET_HINT.get(settings.detail, 200),
            "sensitivity": "private",
            "required": False,
        }]

    # ---- 完成事件 ----

    def on_chat_completed(self, payload: object) -> None:
        try:
            self._enqueue_completed(payload)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "完成的对话没能进入判断队列", {"error_type": type(error).__name__})

    def _enqueue_completed(self, payload: object) -> None:
        if self._closed or not isinstance(payload, Mapping):
            return
        settings = self._settings()
        if not settings.enabled:
            return
        turn_id = _text(payload.get("turnId"))
        character_id = _text(payload.get("characterId")) or "default"
        if not turn_id:
            return
        with self._lock:
            state = self._state(character_id)
            if turn_id not in state.pending:
                return
        self._ensure_worker()
        try:
            self._queue.put_nowait(("judge", character_id, turn_id, float(self._clock())))
        except queue.Full:
            self._log("debug", "判断队列已满，这一轮被跳过")

    # ---- 后台线程 ----

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            worker = threading.Thread(target=self._worker_loop, name="emotion-pool-judge", daemon=True)
            self._worker = worker
            worker.start()

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if item is None:
                return
            try:
                self._handle(item)
            except Exception as error:  # noqa: BLE001
                self._log("warning", "情绪判断线程出错", {"error_type": type(error).__name__})
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    def _handle(self, item: tuple) -> None:
        if self._closed or len(item) < 4:
            return
        _kind, character_id, turn_id, started_at = item
        settings = self._settings()
        if not settings.enabled:
            return
        with self._lock:
            state = self._state(character_id)
            record = state.pending.get(turn_id)
            if not isinstance(record, Mapping):
                return
            user_text = str(record.get("text") or "")
            spoken_at = float(record.get("at") or started_at)
            # 「上一轮至今」要量的是「距上一轮结束多久」，不是「这次生成了几秒」。
            # 旧写法用 spoken_at（本轮注入的时刻）做起点，于是永远算成几秒、
            # 模型看到的永远是「刚刚」——「很久没来」这条路从来没生效过。
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            previous_at = float(last.get("at") or 0.0)
            gap_seconds = spoken_at - previous_at if previous_at > 0 else 0.0
            state_line, summary = _state_lines(state)
            character_name = self._names.get(character_id, "")
        appraiser = self._appraiser
        reply_text = self._reply_text(turn_id)
        verdict = None
        if appraiser is not None:
            try:
                if appraiser.available():
                    verdict = appraiser.appraise(
                        user_text=user_text,
                        reply_text=reply_text,
                        state_line=state_line,
                        summary=summary,
                        elapsed=_elapsed(max(0.0, gap_seconds)),
                        character_name=character_name,
                        persona=self._persona(),
                    )
            except Exception as error:  # noqa: BLE001
                self._log("warning", "情绪判断失败", {"error_type": type(error).__name__})
                verdict = None

        now = float(self._clock())
        episode_snapshot = None
        with self._lock:
            state = self._state(character_id)
            state.pending.pop(turn_id, None)
            state.turn_count += 1
            before_values = {
                key: round(value, 4) for key, value in state.values.items() if value > 0
            }
            if verdict is not None:
                applied = state.apply_mood(
                    verdict.roster, verdict.deltas, verdict.values or None,
                )
                state.set_summary(verdict.summary)
                state.updated_at = now
                top = state.top(1)
                state.note_main(top[0][0] if top else None)
                state.last_judge = {
                    "at": now,
                    "roster": list(verdict.roster) if verdict.roster is not None else None,
                    "before": before_values,   # 判断前的快照，下一轮给判断器看趋势
                    "deltas": dict(verdict.deltas),
                    "values": dict(verdict.values),
                    "applied": {key: value for key, value in applied.items()},
                    "evicted": list(getattr(applied, "evicted", ())),
                    "summary": verdict.summary,
                    "gist": verdict.gist,
                    "source": "turn",
                }
                episode_snapshot = self._record_episode(
                    state, applied, verdict, settings, now, user_text, turn_id,
                )
                self._append_history({
                    "at": now,
                    "source": "turn",
                    "character": character_id,
                    "turn": state.turn_count,
                    "roster": list(verdict.roster) if verdict.roster is not None else None,
                    "deltas": {key: round(value, 1) for key, value in verdict.deltas.items()},
                    "applied": {key: round(value, 1) for key, value in applied.items()},
                    "evicted": list(getattr(applied, "evicted", ())),
                    "after": {
                        key: prompt_module.percent(value)
                        for key, value in state.top(3, floor=0.0) if value > 0
                    },
                    "summary": verdict.summary,
                    "gist": verdict.gist,
                    "user": user_text,
                    "reply": reply_text,
                })
                self._log("info", "这一轮的情绪判断完成", {
                    "event": "emotion_pool.turn.judged",
                    "turn": state.turn_count,
                    # 中文键的字典会被宿主的日志净化器整条丢掉（键名必须匹配
                    # [A-Za-z][A-Za-z0-9_.-]*），所以拼成文本再打，否则日志永远显示 deltas={}。
                    "deltas": _pairs_text(verdict.deltas) or "none",
                    "applied": _pairs_text(applied) or "none",
                    "evicted": ",".join(getattr(applied, "evicted", ())) or "none",
                    "summary": verdict.summary[:40],
                })
            else:
                self._log("debug", "这一轮没有拿到判断，保持原样", {"turn": state.turn_count})
            self._save()

        self._ensure_baseline(character_id, character_name, appraiser)

        if settings.memory_write and episode_snapshot is not None:
            self._remember(character_id, episode_snapshot, now)
        if settings.memory_read and len(user_text) >= 6 and user_text != self._last_recall_query:
            self._recall(character_id, user_text, now)

    def _append_history(self, record: Mapping[str, Any]) -> None:
        """逐轮追加一行 jsonl，给「导出运行日志」用。超过 2 MB 就轮转一份。"""

        path = getattr(self, "_history_path", None)
        if path is None:
            return
        try:
            if path.is_file() and path.stat().st_size > HISTORY_MAX_BYTES:
                path.replace(path.with_name(path.name + ".1"))
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line[:HISTORY_RECORD_CHARS] + "\n")
        except OSError:
            pass

    def _read_history(self, limit: int) -> list[dict[str, Any]]:
        path = getattr(self, "_history_path", None)
        if path is None or not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, Mapping):
                        rows.append(dict(item))
        except OSError:
            return []
        return rows[-max(1, int(limit)):]

    def export_log(self, limit: int = 120) -> str:
        """把当前值、逐轮判定、涟漪和聊天原文导成一个文本文件，返回路径。"""

        now = float(self._clock())
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        path = _history_path(getattr(self._store, "path", None))
        if path is None:
            raise RuntimeError("没有可写的数据目录")
        target = path.with_name(f"emotion-pool-log-{stamp}.txt")
        with self._lock:
            state = self._state(self._current_character_id())
            described = state_module.describe(state)
            lines = [
                "情绪池 · 运行日志导出",
                "=" * 40,
                f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
                f"插件版本：{PLUGIN_VERSION}",
                f"角色：{self._current_character_id()}",
                "",
                "【一】当前情绪",
            ]
            if described["slots"]:
                lines.append("，".join(
                    f"{item['role']}·{item['key']} {prompt_module.percent(item['value'])}%"
                    for item in described["slots"]
                ))
            else:
                lines.append("（此刻没有哪根弦被拨动）")
            if described.get("residual"):
                lines.append("余韵：" + "、".join(
                    f"{item['key']} {prompt_module.percent(item['value'])}%"
                    for item in described["residual"]
                ))
            if state.summary:
                lines.append("心境：" + state.summary)
            if state.baseline:
                lines.append("底色：" + state.baseline)
            lines.append(f"判定轮次：{state.turn_count}")
            lines.extend(["", "【二】逐轮判定记录"])
            history = self._read_history(limit)
            if not history:
                lines.append("（还没有记录）")
            for item in history:
                when = time.strftime("%m-%d %H:%M:%S", time.localtime(float(item.get("at") or 0)))
                kind = "刷新重判" if item.get("source") == "refresh" else "逐轮判断"
                lines.append("")
                lines.append(f"{when}  {kind}  第 {item.get('turn')} 轮")
                deltas = item.get("deltas") or {}
                applied = item.get("applied") or {}
                if deltas:
                    lines.append("  模型给的增减：" + "、".join(
                        f"{key} {float(value):+.0f}" for key, value in deltas.items()))
                if applied:
                    lines.append("  实际落地：" + "、".join(
                        f"{key} {float(value):+.0f}" for key, value in applied.items()))
                if item.get("evicted"):
                    lines.append("  被挤出名额：" + "、".join(str(x) for x in item["evicted"]))
                after = item.get("after") or {}
                if after:
                    lines.append("  判断后：" + "、".join(f"{k} {v}%" for k, v in after.items()))
                if item.get("summary"):
                    lines.append("  心境：" + str(item["summary"]))
                if item.get("gist"):
                    lines.append("  因为：" + str(item["gist"]))
                if item.get("user"):
                    lines.append("  用户说：" + str(item["user"])[:400])
                if item.get("reply"):
                    lines.append("  她回：  " + str(item["reply"])[:400])
            lines.extend(["", "【三】情绪涟漪"])
            episodes = list(state.episodes[-20:])
            if not episodes:
                lines.append("（还没有）")
            for episode in episodes:
                when = time.strftime("%m-%d %H:%M", time.localtime(float(episode.at)))
                gist = (episode.gist or episode.trigger or "").strip()
                lines.append(f"{when}  {episode.emotion} {episode.delta:+.0f}  {gist}")
        lines.extend(["", "【四】聊天原文"])
        for entry in self._entries(limit):
            kind = str(entry.get("kind") or "")
            if kind == "human":
                text = _payload_text(entry.get("payload"))
                who = "用户"
            elif kind == "assistant":
                text = _segments_text(entry.get("payload"))
                who = "她　"
            else:
                continue
            if not text:
                continue
            lines.append(f"{_clock_of(entry)}  {who}：{text[:300]}")
        lines.append("")
        lines.append("（本文件包含聊天原文，可能含隐私内容，请自行保管。）")
        target.write_text("\n".join(lines), encoding="utf-8")
        return str(target)

    def _ensure_baseline(self, character_id: str, character_name: str, appraiser: Any) -> None:
        """每个角色提炼一次性格底色，只给面板用。

        放在判断线程里做，不阻塞对话；失败就下次再说，但同一个角色一小时内只试一次。
        """

        fetch = getattr(appraiser, "baseline", None)
        if not callable(fetch):
            return
        now = float(self._clock())
        with self._lock:
            state = self._state(character_id)
            if state.baseline:
                return
            if (now - self._baseline_tried.get(character_id, 0.0)) < 3600:
                return
            self._baseline_tried[character_id] = now
        try:
            line = fetch(persona=self._persona(), character_name=character_name)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "性格底色没能提炼", {"error_type": type(error).__name__})
            return
        if not line:
            return
        with self._lock:
            self._state(character_id).set_baseline(line)
            self._save()
        self._log("info", "性格底色已提炼", {"baseline": str(line)[:40]})

    def _record_episode(
        self,
        state: state_module.CharacterState,
        applied: Mapping[str, float],
        verdict: Any,
        settings: Any,
        now: float,
        user_text: str,
        turn_id: str,
    ) -> dict[str, Any] | None:
        if not applied or settings.episode_keep <= 0:
            return None
        # 被名额挤出去的那几根不算情绪事件——它们看着像暴跌，其实只是没位置了。
        evicted = set(getattr(applied, "evicted", ()))
        real = {key: value for key, value in applied.items() if key not in evicted}
        if not real:
            return None
        # 正负都记：情绪回落同样是「发生过的事」，
        # 只记上涨会让涟漪表看起来只涨不跌，回查历史时失真。
        key, delta = max(real.items(), key=lambda item: abs(item[1]))
        if abs(delta) < state_module.EPISODE_MIN_DELTA:
            return None
        episode = state_module.Episode(
            at=now,
            emotion=key,
            delta=delta,
            intensity=state.values.get(key, 0.0),
            source="model",
            gist=getattr(verdict, "gist", "") or "",
            trigger=" ".join(user_text.split())[:160],
            turn_id=turn_id,
        )
        state.add_episode(episode, keep=settings.episode_keep)
        return {
            "emotion": episode.emotion,
            "intensity": float(episode.intensity),
            "delta": float(episode.delta),
            "gist": episode.gist,
            "trigger": episode.trigger,
        }

    # ---- 手动刷新 ----

    def refresh_with_model(self) -> dict[str, Any]:
        settings = self._settings()
        if not settings.enabled:
            return {"ok": False, "message": "情绪池已关闭，先打开总开关。"}
        appraiser = self._appraiser
        if appraiser is None or not appraiser.available():
            return {"ok": False, "message": "没有可用的聊天模型，暂时没法重新判断。"}
        character_id = self._current_character_id()
        history = self._history_text(limit=16)
        if not history.strip():
            return {"ok": False, "message": "还没读到对话，等她和你聊过几句再刷新。"}
        with self._lock:
            state = self._state(character_id)
            state_line, summary = _state_lines(state)
            character_name = self._names.get(character_id, "")
        verdict = None
        try:
            verdict = appraiser.rebuild(
                history=history, state_line=state_line, summary=summary,
                character_name=character_name, persona=self._persona(),
            )
        except Exception as error:  # noqa: BLE001
            self._log("warning", "刷新时的情绪判断失败", {"error_type": type(error).__name__})
        if verdict is None:
            return {"ok": False, "message": "模型这次没给出可用的判断，先不动。"}
        now = float(self._clock())
        with self._lock:
            state = self._state(character_id)
            # 刷新是"重判"，不是"再加"：优先用绝对值覆盖，只有模型没按契约给 values 时
            # 才退回增量——否则同一段对话每点一次就往上叠一层，越刷越委屈。
            before_snapshot = dict(state.values)
            applied = state.apply_mood(verdict.roster, verdict.deltas, verdict.values or None)
            mode = "set" if verdict.values else "add"
            evicted_now = set(getattr(applied, "evicted", ()))
            strongest_set = max(
                ((key, abs(value)) for key, value in applied.items() if key not in evicted_now),
                key=lambda item: item[1], default=("", 0.0))
            if strongest_set[1] >= state_module.EPISODE_MIN_DELTA and settings.episode_keep > 0:
                key = strongest_set[0]
                state.add_episode(state_module.Episode(
                    at=now,
                    emotion=key,
                    delta=applied.get(key, 0.0),
                    intensity=state.values.get(key, 0.0),
                    source="refresh",
                    gist=verdict.gist,
                    trigger="（刷新时重新判断）",
                    turn_id="",
                ), keep=settings.episode_keep)
            top = state.top(1)
            state.note_main(top[0][0] if top else None)
            state.set_summary(verdict.summary)
            state.updated_at = now
            state.last_judge = {
                "at": now,
                "roster": list(verdict.roster) if verdict.roster is not None else None,
                "before": {key: round(value, 4) for key, value in before_snapshot.items() if value > 0},
                "values": dict(verdict.values),
                "deltas": dict(verdict.deltas),
                "applied": {key: value for key, value in applied.items()},
                "evicted": list(getattr(applied, "evicted", ())),
                "summary": verdict.summary,
                "gist": verdict.gist,
                "mode": mode,
                "source": "refresh",
            }
            self._save()
            self._append_history({
                "at": now,
                "source": "refresh",
                "character": character_id,
                "turn": state.turn_count,
                "roster": list(verdict.roster) if verdict.roster is not None else None,
                "deltas": {key: round(value, 1) for key, value in verdict.deltas.items()},
                "applied": {key: round(value, 1) for key, value in applied.items()},
                "evicted": list(getattr(applied, "evicted", ())),
                "after": {
                    key: prompt_module.percent(value)
                    for key, value in state.top(3, floor=0.0) if value > 0
                },
                "summary": verdict.summary,
                "gist": verdict.gist,
                "user": "",
                "reply": "",
            })
            self._log("info", "刷新时的情绪判断完成", {
                "event": "emotion_pool.refresh.judged",
                "turn": state.turn_count,
                "mode": mode,
                "roster": ",".join(verdict.roster or []) or "none",
                # 宿主日志有 1800 字节上限，字段太长会被整段截掉，所以绝对值只留前三根
                "values": _pairs_text(verdict.values, limit=3) or "none",
                "applied": _pairs_text(applied) or "none",
                "summary": verdict.summary[:40],
            })
            described = state_module.describe(state)
        if not described["slots"]:
            return {"ok": True, "message": "她想了想，此刻没有哪根弦在被拨动。"}
        parts = [
            f"{item['role']}·{item['key']} {prompt_module.percent(item['value'])}%"
            for item in described["slots"]
        ]
        return {"ok": True, "message": ("她重新判断：" + "，".join(parts))[:200]}
    # ---- 长期记忆 ----

    def _remember(self, character_id: str, episode: Mapping[str, Any], now: float) -> None:
        if self._memory is None:
            return
        if float(episode.get("delta") or 0.0) < MEMORY_MIN_DELTA:
            return
        with self._lock:
            state = self._state(character_id)
            if (now - self._last_memory_write) < MEMORY_WRITE_MIN_GAP:
                return
            if state.turn_count - self._last_memory_turn < MEMORY_WRITE_MIN_TURNS:
                return
            self._last_memory_write = now
            self._last_memory_turn = state.turn_count
        gist = str(episode.get("gist") or "").strip()
        trigger = " ".join(str(episode.get("trigger") or "").split())[:40]
        emotion = str(episode.get("emotion") or "")
        delta = float(episode.get("delta") or 0.0)
        stamp = time.strftime("%m-%d %H:%M", time.localtime(now))
        content = f"（情绪记忆）{stamp} 他说「{trigger}」，她{emotion} {delta:+.0f}"
        if gist:
            content += f"：{gist}"
        try:
            self._memory.remember(content, importance=max(0.5, min(1.0, delta / 100.0 + 0.4)))
        except Exception as error:  # noqa: BLE001
            self._log("debug", "长期记忆写入失败", {"error_type": type(error).__name__})

    def _recall(self, character_id: str, query: str, now: float) -> None:
        if self._memory is None:
            return
        self._last_recall_query = query
        lines: list[str] = []
        try:
            lines = self._memory.recall(query, limit=3)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "长期记忆检索失败", {"error_type": type(error).__name__})
            return
        if not lines:
            return
        line = "；".join(item.strip() for item in lines if item.strip())[:160]
        with self._lock:
            state = self._state(character_id)
            state.memory_ctx = {"line": line, "at": now}
            self._save()

    def _memory_line(self, state: state_module.CharacterState, now: float, settings: Any) -> str:
        if not settings.memory_read:
            return ""
        ctx = state.memory_ctx
        if not isinstance(ctx, Mapping):
            return ""
        at = float(ctx.get("at") or 0.0)
        line = str(ctx.get("line") or "").strip()
        if not line or (now - at) > MEMORY_CTX_TTL:
            return ""
        return line

    # ---- 时间线 ----

    def _entries(self, limit: int) -> list[Mapping[str, Any]]:
        timeline = self._timeline
        if timeline is None:
            return []
        try:
            response = timeline.read_recent({"limit": max(1, int(limit))})
        except Exception as error:  # noqa: BLE001
            self._log("debug", "读取最近对话失败", {"error_type": type(error).__name__})
            return []
        entries = response.get("entries") if isinstance(response, Mapping) else None
        if not isinstance(entries, list):
            return []
        return [item for item in entries if isinstance(item, Mapping)]

    def _history_text(self, limit: int = 16) -> str:
        lines: list[str] = []
        for entry in self._entries(limit):
            kind = str(entry.get("kind") or "")
            if kind == "human":
                text = _payload_text(entry.get("payload"))
                if text:
                    lines.append(f"[{_clock_of(entry)}] 用户：{text[:200]}")
            elif kind == "assistant":
                text = _segments_text(entry.get("payload"))
                if text:
                    lines.append(f"[{_clock_of(entry)}] 她：{text[:200]}")
        return "\n".join(lines[-max(1, int(limit)):])

    def _reply_text(self, turn_id: str) -> str:
        entries = self._entries(8)
        matched = ""
        fallback = ""
        for entry in reversed(entries):
            if str(entry.get("kind") or "") != "assistant":
                continue
            text = _segments_text(entry.get("payload"))
            if not text:
                continue
            if str(entry.get("turnId") or "") == turn_id:
                matched = text
                break
            if not fallback:
                fallback = text
        return (matched or fallback)[:MAX_REPLY_CHARS]

    # ---- 设置页与仪表盘 ----

    def dashboard_payload(self) -> dict[str, Any]:
        """仪表盘只拿数字：此刻在场的那两三根，和"上一次判断是什么时候"。

        人同时只有主·次·辅三个位置，所以面板也不再是二十一根条的清单——
        栏里那几根就是她此刻的全部。
        """

        with self._lock:
            state = self._state(self._current_character_id())
            described = state_module.describe(state)
            slots = [
                {
                    "role": item["role"],
                    "key": item["key"],
                    "value": prompt_module.percent(item["value"]),
                }
                for item in described["slots"]
            ]
            # 当前没有的情绪收进抽屉：面板默认只摆在场的那三根，其余的点箭头才展开。
            present = {item["key"] for item in described["slots"]}
            others = sorted(
                (
                    {"key": key, "value": prompt_module.percent(state.values.get(key, 0.0))}
                    for key in taxonomy.EMOTIONS
                    if key not in present
                ),
                key=lambda item: -item["value"],
            )
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            at = float(last.get("at") or 0.0)
            updated = (
                f"上一次判断：{prompt_module.ago(self._clock() - at)} · "
                f"{prompt_module.status_last(state, self._clock())['label']}"
                if at > 0 else "还没有判断过：等她回一轮话，或点一次刷新"
            )
            payload = {
                "slots": slots,
                "others": others,
                "baseline": (state.baseline or "")[:40],
                "updated": updated[:80],
                "turns": int(state.turn_count),
            }
            self._save()
        return payload

    def preview(self) -> str:
        settings = self._settings()
        now = float(self._clock())
        with self._lock:
            character_id = self._current_character_id()
            state = self._state(character_id)
            text = prompt_module.build(
                state, settings, now, memory_line=self._memory_line(state, now, settings),
            )
        head = "这是本轮会原样塞进提示词的内容：" if settings.enabled else "插件已关闭，正常不会注入。下面是关掉之前的内容："
        budget = prompt_module.BUDGET_HINT.get(settings.detail, 200)
        return f"{head}\n\n{text}\n\n（共 {len(text)} 字，预算提示 {budget} tokens，详略档位：{settings.detail}）"

    def reset(self) -> str:
        now = float(self._clock())
        with self._lock:
            character_id = self._current_character_id()
            state = self._state(character_id)
            state.values = taxonomy.empty_pools()
            state.summary = ""
            state.updated_at = now
            state.last_judge = None
            state.memory_ctx = None
            self._save()
        return f"已经把她归零（{character_id}）。涟漪记录还在，共 {len(state.episodes)} 条。"[:240]

    def memory_test(self) -> str:
        if self._memory is None:
            return "没有可用的记忆桥。"
        try:
            return str(self._memory.test())[:600]
        except Exception as error:  # noqa: BLE001
            return f"测试失败：{type(error).__name__}"

    def status(self) -> dict[str, dict[str, str]]:
        now = float(self._clock())
        with self._lock:
            state = self._state(self._current_character_id())
            return {
                "now": prompt_module.status_now(state),
                "last": prompt_module.status_last(state, now),
            }

    def episode_rows(self, limit: int = 20) -> list[dict[str, Any]]:
        count = max(1, min(100, int(limit)))
        with self._lock:
            state = self._state(self._current_character_id())
            episodes = list(state.episodes[-count:])
        rows: list[dict[str, Any]] = []
        for index, episode in enumerate(reversed(episodes)):
            gist = (episode.gist or episode.trigger or "").strip()
            rows.append({
                "itemId": f"{episode.turn_id or int(episode.at)}-{index}",
                "values": {
                    "at": _iso(episode.at),
                    "emotion": episode.emotion[:16],
                    "level": f"{episode.delta:+.0f}"[:8],
                    "source": SOURCE_LABELS.get(episode.source, episode.source)[:16],
                    "gist": " ".join(gist.split())[:120],
                },
            })
        return rows

    def saved_text(self) -> str:
        try:
            return self._store.path_text()
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._save()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    # ---- 日志 ----

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> None:
        if self._logger is None:
            return
        try:
            getattr(self._logger, severity)(message, fields=dict(fields or {}))
        except Exception:  # noqa: BLE001
            pass


def _history_path(state_path: object):
    """历史与导出都放在 state.json 旁边；拿不到路径就返回 None。"""

    try:
        return Path(str(state_path)).with_name("history.jsonl")
    except (TypeError, ValueError):
        return None


def _persona_excerpt(text: object, budget: int = PERSONA_BUDGET) -> str:
    """整张卡都给，只在超过预算时按段落等比例收窄。

    以前按 600 字硬切，正好切掉角色卡里最关键的那句性格描述；
    后来又按段落白名单挑，那是拿一张卡的形状去套所有角色——
    换一个段落标题不同的角色就会退化。现在两样都不做：
    卡不长就整张给，太长就每段按比例留一点，保证没有一段被整段丢掉。
    """

    flat = " ".join(str(text or "").split())
    if len(flat) <= budget:
        return flat
    marks = list(_SECTION.finditer(flat))
    if len(marks) < 2:
        return flat[:budget]
    total = float(len(flat))
    ratio = budget / total if total else 0.0
    pieces = []
    for index, match in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(flat)
        body = flat[match.start():end].strip()
        keep = max(80, int(len(body) * ratio))
        pieces.append(body[:keep])
    return " ".join(pieces)[:budget]


def _pairs_text(values: object, limit: int = 6) -> str:
    """把 {"委屈": 24.0} 拼成 "委屈=24,失望=10"。

    宿主的日志净化器只接受 ASCII 键名（[A-Za-z][A-Za-z0-9_.-]*），
    直接传中文字典会被整条吃掉，日志里永远是 {}——排查时等于没有。
    """

    if not isinstance(values, Mapping):
        return ""
    parts = []
    for key, value in values.items():
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if abs(number) < 0.01:
            continue
        parts.append(f"{key}={number:+.0f}")
        if len(parts) >= max(1, int(limit)):
            break
    return ",".join(parts)[:200]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _elapsed(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60:
        return "刚刚"
    if value < 3600:
        return f"{int(value // 60)} 分钟前"
    if value < 86400:
        return f"{int(value // 3600)} 小时前"
    return f"{int(value // 86400)} 天前"


def _state_lines(state: state_module.CharacterState) -> tuple[str, str]:
    """给判断器看的状态行。带上上一轮的数值，让它能看出这根弦在涨还是在落。"""

    described = state_module.describe(state)
    if not described["slots"]:
        return "", described["summary"]
    last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
    before = last.get("before") if isinstance(last.get("before"), Mapping) else {}
    parts = []
    for item in described["slots"]:
        now_pct = prompt_module.percent(item["value"])
        text = f"{item['role']}·{item['key']} {now_pct}%"
        notes = []
        streak = state.streak_of(item["key"])
        if streak >= 3:
            notes.append(f"已连续 {streak} 轮为主")
        previous = before.get(item["key"])
        if isinstance(previous, (int, float)) and not isinstance(previous, bool):
            prev_pct = prompt_module.percent(float(previous))
            if abs(prev_pct - now_pct) >= 1:
                notes.append(f"上一轮 {prev_pct}%")
        if notes:
            text += "（" + "，".join(notes) + "）"
        parts.append(text)
    return "，".join(parts), described["summary"]


def _iso(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return datetime.now().astimezone().isoformat(timespec="seconds")


def _payload_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    text = payload.get("text")
    return " ".join(text.split()) if isinstance(text, str) else ""


def _clock_of(entry: Mapping[str, Any]) -> str:
    created = entry.get("createdAt")
    if isinstance(created, str) and created:
        try:
            return datetime.fromisoformat(created).astimezone().strftime("%H:%M")
        except ValueError:
            pass
    return "--:--"


def _segments_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    segments = payload.get("segments")
    if isinstance(segments, str):
        return " ".join(segments.split())
    if not isinstance(segments, list):
        text = payload.get("text")
        return " ".join(text.split()) if isinstance(text, str) else ""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, Mapping):
            piece = segment.get("text")
            if isinstance(piece, str) and piece.strip():
                parts.append(" ".join(piece.split()))
    return " ".join(parts).strip()


__all__ = ["EmotionRuntime"]