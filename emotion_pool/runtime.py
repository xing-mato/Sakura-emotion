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
import re
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Mapping

from . import appraisal_model
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
PLUGIN_VERSION = "1.0.0-beta.8"

# 先判后答：判断在她说之前跑，所以要给一个硬上限。超时就用当前状态回答，
# 不能因为判断慢而卡住对话——超时后那次调用自己跑完也没关系，结果直接丢掉。
JUDGE_TIMEOUT = 15.0

# 同一次判断并发采三份再取平均：单份采样的抖动是主要噪声，平均之后稳得多。
# 三份是并发跑，等待时间约等于最慢的那一份，不会变成三倍。
JUDGE_SAMPLES = 3

# 判断"聊天记录还在不在"时，多久以内的状态不去碰：她可能正在这一轮说话，
# 而一次瞬时的读取失败不该把状态抹掉。
CHAT_SYNC_GRACE = 60.0

# 后校在她说完之后跑，不占用户的等待时间，所以可以给得宽一点。
RECHECK_TIMEOUT = 25.0
# 人设：整张卡都交给判断器，只做总长保护。
# 挑段落等于替角色决定「哪段不重要」，换个卡（段落标题不一样）就会漏掉关键信息。
# 先别截：上限放开到 20000，整张卡都能进判断器（Sakura 的卡 12k）。
# 截断只是权宜，等跑一段时间看判断质量和延迟，再决定要不要做句子级抽取。
PERSONA_BUDGET = 20000
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
        samples: int = JUDGE_SAMPLES,
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
        # 先判采样份数：默认 3（并发取平均）；离线测试用 1 保证确定性
        self._samples = max(1, int(samples))
        self._clock = clock

        self._lock = threading.RLock()
        self._characters: dict[str, state_module.CharacterState] | None = None
        self._names: dict[str, str] = {}
        # 人设按角色分开缓存：换角色必须立刻换成新角色的卡。
        self._persona_cache: dict[str, tuple[str, float]] = {}
        self._baseline_tried: dict[str, float] = {}
        self._last_character_id = ""
        self._last_memory_write = 0.0
        self._last_memory_turn = -999
        self._last_recall_query = ""
        # 面板是轮询的，一次读空可能只是宿主还没加载完：要两次都对得上才动手。
        self._empty_chat_seen: dict[str, float] = {}

        self._history_path = _history_path(getattr(store, "path", None))
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

    def _persona(self, character_id: str = "") -> str:
        """她的角色设定节选：判断必须按她的性格，而不是通用陪伴逻辑。

        缓存按角色分开。原来是一个全局槽位、10 分钟有效——切角色后的头十分钟，
        新角色会拿着旧角色的卡被判断，那就是人设串台。
        """

        key = character_id or "default"
        text, at = self._persona_cache.get(key, ("", 0.0))
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
        self._persona_cache[key] = (excerpt, float(self._clock()))
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

        # 聊天记录被清空时，情绪一起归零——没有那段对话，就没有那段情绪留下的理由。
        self._sync_cleared_chat(character_id)

        should_judge = False
        with self._lock:
            state = self._state(character_id)
            self._last_character_id = character_id
            if character_name:
                self._names[character_id] = character_name[:64]
            state.prune_pending(now)
            if source == "chat" and text and turn_id and turn_id not in state.pending:
                # 同一个 turn_id 只判一次：先记下来，判断期间宿主再来要一次注入也不会重复跑模型。
                state.pending[turn_id] = {"text": " ".join(text.split())[:600], "at": now}
                state.trim_pending()
                should_judge = True

        # 先判后答：她说之前就把情绪定下来，然后再把此刻的状态交给提示词。
        if should_judge:
            self._judge_ahead(character_id, text, turn_id, now)

        with self._lock:
            state = self._state(character_id)
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

    # ---- 判断 ----

    def _call_with_timeout(
        self, call: Any, kwargs: dict[str, Any], timeout: float, slow_message: str,
    ) -> Any:
        """模型调用放进线程里跑，超时就不等了。超时那次自己跑完也无所谓，结果直接丢。"""

        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["verdict"] = call(**kwargs)
            except Exception as error:  # noqa: BLE001
                box["error"] = type(error).__name__
                box["verdict"] = None

        thread = threading.Thread(target=run, name="emotion-pool-model", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._log("warning", slow_message, {"timeout_s": timeout})
            return None
        if box.get("error"):
            self._log("warning", "情绪判断失败", {"error_type": str(box["error"])})
        return box.get("verdict")

    def _appraise_with_timeout(self, appraiser: Any, **kwargs: Any) -> Any:
        return self._call_with_timeout(
            appraiser.appraise, kwargs, JUDGE_TIMEOUT, "先判超时，这一轮按当前情绪回答",
        )

    def _sample_verdicts(self, appraiser: Any, kwargs: dict[str, Any], samples: int) -> Any:
        """同一次判断并发采 N 份，取平均。比单份稳，等待时间约等于最慢的一份。"""

        results: list[Any] = [None] * max(1, samples)

        def run(index: int) -> None:
            try:
                results[index] = appraiser.appraise(**kwargs)
            except Exception as error:  # noqa: BLE001
                self._log("debug", "采样失败", {"error_type": type(error).__name__})
                results[index] = None

        threads = [
            threading.Thread(target=run, args=(index,), name=f"emotion-pool-sample-{index}", daemon=True)
            for index in range(len(results))
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + JUDGE_TIMEOUT
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        got = [item for item in results if item is not None]
        if not got:
            self._log("warning", "先判没有拿到结果，这一轮按当前情绪回答", {"samples": len(results)})
            return None
        if len(got) < len(results):
            self._log("warning", "先判只拿到部分采样", {"got": len(got), "expected": len(results)})
        return _average_verdicts(got)

    # ---- 完成事件 ----

    def on_chat_completed(self, payload: object) -> None:
        """她说完了：补记回复，然后按她实际的表达做一次校正。"""

        try:
            self._schedule_recheck(payload)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "回复校正没能安排", {"error_type": type(error).__name__})
        try:
            self._attach_reply(payload)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "回复没能补进判断记录", {"error_type": type(error).__name__})

    def _recent_her(self, *, rounds: int = 2, per_round: int = 3) -> str:
        """她前几轮自己说的话（带语气）。

        先判跑在她开口之前。如果只给她一个数字状态行和他这一句话，判断器手上就没有"她"——
        能拿来做依据的只剩他的用词，判断必然退化成关键词表（他说了别的女孩，就该吃醋）。
        把她说过的原话递过去，证据才有地方落。
        """

        blocks: list[str] = []
        for entry in reversed(self._entries(12)):
            if len(blocks) >= max(1, rounds):
                break
            if str(entry.get("kind") or "") != "assistant":
                continue
            payload = entry.get("payload")
            items = payload.get("segments") if isinstance(payload, Mapping) else None
            if not isinstance(items, list):
                continue
            parts: list[str] = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                # 宿主时间线里的段落文本在 "text" 上，不是 ja/zh（和回复校正同一条坑）。
                spoken = item.get("zh") or item.get("text") or item.get("ja") or ""
                body = " ".join(str(spoken).split())[:60]
                if not body:
                    continue
                tone = " ".join(str(item.get("tone") or "").split())[:6]
                parts.append(f"[{tone}] {body}" if tone else body)
            if parts:
                blocks.append(" / ".join(parts[-max(1, per_round):]))
        return "\n".join(reversed(blocks))[:600]

    def _reply_segments(self, turn_id: str) -> list[dict[str, Any]]:
        """把她这一轮的回复拆成段（每段带语气标签）。"""

        for entry in reversed(self._entries(8)):
            if str(entry.get("kind") or "") != "assistant":
                continue
            if str(entry.get("turnId") or "") != turn_id:
                continue
            payload = entry.get("payload")
            items = payload.get("segments") if isinstance(payload, Mapping) else None
            if isinstance(items, list):
                out: list[dict[str, Any]] = []
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    # 宿主时间线里的段落文本在 "text" 上，不是 ja/zh——之前只读 ja/zh，
                    # 校正拿到的每一段都是空的，于是模型只能看见语气标签。
                    spoken = item.get("zh") or item.get("text") or ""
                    out.append({
                        "tone": str(item.get("tone") or "中性"),
                        "text": " ".join(str(spoken).split()),
                        "ja": " ".join(str(item.get("ja") or "").split()),
                    })
                return out
        return []

    def _schedule_recheck(self, payload: object) -> None:
        """她说完之后，用她实际的表达做一次校正（异步，不占用户的等待时间）。"""

        if self._closed or not isinstance(payload, Mapping):
            return
        if not self._settings().enabled:
            return
        appraiser = self._appraiser
        if appraiser is None or not callable(getattr(appraiser, "recheck", None)):
            return
        turn_id = _text(payload.get("turnId"))
        character_id = _text(payload.get("characterId")) or "default"
        if not turn_id:
            return
        segments = self._reply_segments(turn_id)
        if len(segments) < 2:
            self._log("debug", "回复只有一段，跳过校正", {"segments": len(segments)})
            self._settle_recheck(character_id, turn_id)
            return
        with self._lock:
            state = self._state(character_id)
            record = state.pending.get(turn_id)
            user_text = str(record.get("text") or "") if isinstance(record, Mapping) else ""
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            judged_at = float(last.get("at") or 0.0)
        if judged_at <= 0:
            return
        threading.Thread(
            target=self._recheck,
            args=(character_id, turn_id, user_text, judged_at, segments),
            name="emotion-pool-recheck",
            daemon=True,
        ).start()

    def _settle_recheck(self, character_id: str, turn_id: str, *, failed: bool = False) -> None:
        """没有校正这一步时收尾：别让面板一直写着"等她说完再校正"。"""

        with self._lock:
            state = self._state(character_id)
            last = state.last_judge
            if not isinstance(last, Mapping):
                return
            if str(last.get("turn_id") or "") not in ("", turn_id) and turn_id:
                return
            updated = dict(last)
            updated["rechecked"] = True
            updated["recheck_failed"] = bool(failed)
            updated.setdefault("track", [])
            # 结算快照：面板只显示这个。没校正成时就用先判的值兜底。
            updated["settled"] = _snapshot(state)
            state.last_judge = updated
            self._save()

    def _recheck(
        self, character_id: str, turn_id: str, user_text: str,
        judged_at: float, segments: list[dict[str, Any]],
    ) -> None:
        try:
            self._recheck_inner(character_id, turn_id, user_text, judged_at, segments)
        except Exception as error:  # noqa: BLE001
            self._log("warning", "回复校正出错，保留先判的值", {"error_type": type(error).__name__})

    def _recheck_inner(
        self, character_id: str, turn_id: str, user_text: str,
        judged_at: float, segments: list[dict[str, Any]],
    ) -> None:
        appraiser = self._appraiser
        if appraiser is None or self._closed:
            return
        with self._lock:
            state = self._state(character_id)
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            if float(last.get("at") or 0.0) != judged_at:
                # 已经有更新的一轮判过了，别拿上一轮的证据去改新状态
                return
            state_line, summary = _state_lines(state)
            character_name = self._names.get(character_id, "")
            gap = float(self._clock()) - judged_at
        verdict = self._call_with_timeout(
            appraiser.recheck,
            dict(
                user_text=user_text,
                reply_segments=segments,
                state_line=state_line,
                summary=summary,
                elapsed=_elapsed(max(0.0, gap)),
                character_name=character_name,
                persona=self._persona(character_id),
            ),
            RECHECK_TIMEOUT,
            "校正超时，保留先判的值",
        )
        if verdict is None:
            self._log("info", "回复校正没有结果，保留先判的值", {"turn": state.turn_count})
            self._settle_recheck(character_id, turn_id, failed=True)
            return
        now = float(self._clock())
        track = [" ".join(str(item.get("tone") or "中性").split())[:8] or "中性" for item in segments][:4]
        with self._lock:
            state = self._state(character_id)
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            if float(last.get("at") or 0.0) != judged_at:
                return
            applied = state.apply_mood(verdict.roster, verdict.deltas, verdict.values or None)
            if verdict.summary:
                state.set_summary(verdict.summary)
            state.updated_at = now
            top = state.top(1)
            state.note_main(top[0][0] if top else None)
            last = dict(state.last_judge or {})
            last.update({
                "rechecked": True,
                "settled": _snapshot(state),
                "track": track,
                "recheck": {
                    "at": now,
                    "deltas": dict(verdict.deltas),
                    "applied": {key: round(value, 1) for key, value in applied.items()},
                    "summary": verdict.summary,
                    "gist": verdict.gist,
                },
                # 「上一次判断」那一行现在显示的是校正后的数字，不是先判的预测
                "deltas": {key: round(value, 1) for key, value in verdict.deltas.items()},
                "applied": {key: round(value, 1) for key, value in applied.items()},
                "evicted": list(getattr(applied, "evicted", ())),
                "summary": verdict.summary or last.get("summary") or "",
            })
            state.last_judge = last
            self._append_history({
                "at": now,
                "source": "recheck",
                "turn_id": turn_id,
                "character": character_id,
                "turn": state.turn_count,
                "deltas": {key: round(value, 1) for key, value in verdict.deltas.items()},
                "applied": {key: round(value, 1) for key, value in applied.items()},
                "evicted": list(getattr(applied, "evicted", ())),
                "track": track,
                "after": {
                    key: prompt_module.percent(value)
                    for key, value in state.top(3, floor=0.0) if value > 0
                },
                "summary": verdict.summary,
                "gist": verdict.gist,
                "user": "",
                "reply": "",
            })
            self._save()
        self._log("info", "回复校正完成", {
            "event": "emotion_pool.turn.rechecked",
            "turn": state.turn_count,
            "track": ",".join(track) or "none",
            "applied": _pairs_text(applied) or "none",
            "summary": (verdict.summary or "")[:40],
        })

    def _attach_reply(self, payload: object) -> None:
        """她说完了：把回复补进这一轮的判断记录（判断在她说之前已经做完，这里不再跑模型）。"""

        if self._closed or not isinstance(payload, Mapping):
            return
        turn_id = _text(payload.get("turnId"))
        if not turn_id:
            return
        with self._lock:
            for state in self._states().values():
                state.pending.pop(turn_id, None)
        reply = self._reply_text(turn_id)
        path = self._history_path
        if not reply or path is None or not path.is_file():
            return
        try:
            if path.stat().st_size > 512 * 1024:
                return
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        target = -1
        for index in range(len(lines) - 1, -1, -1):
            try:
                item = json.loads(lines[index])
            except ValueError:
                continue
            if isinstance(item, Mapping) and item.get("turn_id") == turn_id and not item.get("reply"):
                target = index
                break
        if target < 0:
            return
        try:
            record = json.loads(lines[target])
            record["reply"] = reply
            lines[target] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))[:HISTORY_RECORD_CHARS]
            temp = path.with_name(path.name + ".tmp")
            temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temp.replace(path)
        except (OSError, ValueError):
            return

    def _judge_ahead(self, character_id: str, user_text: str, turn_id: str, spoken_at: float) -> None:
        """先判后答：拿"她还没开口"的此刻去判断，判完才让提示词带上它。"""

        settings = self._settings()
        if not settings.enabled:
            return
        # 她前几轮的原话：先判手上不能只有"他这一句"，否则所谓判断只能是查表。
        her_recent = self._recent_her()
        with self._lock:
            state = self._state(character_id)
            # 「上一轮至今」要量的是「距上一轮结束多久」，不是「这次生成了几秒」。
            # 旧写法用 spoken_at（本轮注入的时刻）做起点，于是永远算成几秒、
            # 模型看到的永远是「刚刚」——「很久没来」这条路从来没生效过。
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            previous_at = float(last.get("at") or 0.0)
            gap_seconds = spoken_at - previous_at if previous_at > 0 else 0.0
            state_line, summary = _state_lines(state)
            character_name = self._names.get(character_id, "")
        appraiser = self._appraiser
        verdict = None
        if appraiser is not None:
            try:
                if appraiser.available():
                    verdict = self._sample_verdicts(
                        appraiser,
                        dict(
                        user_text=user_text,
                        # 先判后答：她还没说话，这里没有回复可读。
                        reply_text="",
                        state_line=state_line,
                        summary=summary,
                        elapsed=_elapsed(max(0.0, gap_seconds)),
                        character_name=character_name,
                        persona=self._persona(character_id),
                        her_recent=her_recent,
                        ),
                        self._samples,
                    )
            except Exception as error:  # noqa: BLE001
                self._log("warning", "情绪判断失败", {"error_type": type(error).__name__})
                verdict = None

        now = float(self._clock())
        episode_snapshot = None
        with self._lock:
            state = self._state(character_id)
            # 这里不清 pending：同一轮如果宿主再来要一次注入，要能认出来是同一轮。
            # 等她说完（chat.completed）再清。
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
                previous = state.last_judge if isinstance(state.last_judge, Mapping) else {}
                state.last_judge = {
                    "at": now,
                    "turn_id": turn_id,
                    "roster": list(verdict.roster) if verdict.roster is not None else None,
                    "before": before_values,   # 判断前的快照，下一轮给判断器看趋势
                    "deltas": dict(verdict.deltas),
                    "values": dict(verdict.values),
                    "applied": {key: value for key, value in applied.items()},
                    "evicted": list(getattr(applied, "evicted", ())),
                    "summary": verdict.summary,
                    "gist": verdict.gist,
                    "source": "turn",
                    # 先判只是预测：等她说完会有一次校正，面板据此显示"待校正/已校正"
                    "rechecked": False,
                    "track": [],
                    # 上一轮结算过的样子继续留着，面板在校正完成前显示它，不跳中间态
                    "settled": previous.get("settled"),
                }
                episode_snapshot = self._record_episode(
                    state, applied, verdict, settings, now, user_text, turn_id,
                )
                self._append_history({
                    "at": now,
                    "source": "turn",
                    "turn_id": turn_id,
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
                    # 她还没说话；等她说完，chat.completed 会把回复补进这一行。
                    "reply": "",
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

    def _export_records(self, character_id: str, limit: int) -> list[dict[str, Any]]:
        """只导当前角色的记录。两个角色的判定混在一份里，等于没法分析。"""

        rows = self._read_history(max(limit * 4, 400))
        picked = [item for item in rows if str(item.get("character") or "") == character_id]
        return picked[-max(1, int(limit)):]

    def export_text(self, fmt: str = "txt", limit: int = 120) -> tuple[str, str]:
        """生成导出内容（不落盘）：返回 (建议文件名, 文本)。fmt = txt | jsonl。"""

        now = float(self._clock())
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        character_id = self._current_character_id()
        records = self._export_records(character_id, limit)
        safe = "".join(ch for ch in (character_id or "default") if ch.isalnum() or ch in "._-")
        safe = safe or "default"
        if fmt == "jsonl":
            meta = {
                "_meta": {
                    "plugin": "sakuratools.emotion_pool",
                    "version": PLUGIN_VERSION,
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                    "character": character_id,
                    "records": len(records),
                }
            }
            rows = [json.dumps(meta, ensure_ascii=False)]
            rows.extend(json.dumps(item, ensure_ascii=False) for item in records)
            return f"emotion-pool-{safe}-{stamp}.jsonl", "\n".join(rows) + "\n"
        with self._lock:
            state = self._state(character_id)
            described = state_module.describe(state)
            lines = [
                "情绪池 · 运行日志导出",
                "=" * 40,
                f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
                f"插件版本：{PLUGIN_VERSION}",
                f"角色：{character_id}",
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
            history = records
            if not history:
                lines.append("（还没有记录）")
            for item in history:
                when = time.strftime("%m-%d %H:%M:%S", time.localtime(float(item.get("at") or 0)))
                kind = {"refresh": "刷新重判", "recheck": "回复校正"}.get(
                    str(item.get("source") or ""), "逐轮判断")
                lines.append("")
                tid = str(item.get("turn_id") or "")[:8]
                lines.append(f"{when}  {kind}  第 {item.get('turn')} 轮" + (f"（{tid}）" if tid else ""))
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
                track = item.get("track") or []
                if track:
                    lines.append("  本轮语气：" + " → ".join(str(x) for x in track))
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
        return f"emotion-pool-{safe}-{stamp}.txt", "\n".join(lines) + "\n"

    def export_log(self, limit: int = 120) -> str:
        """写到插件数据目录并返回路径——保存对话框不可用时的退路。只出 JSONL。"""

        name, text = self.export_text("jsonl", limit)
        base = _history_path(getattr(self._store, "path", None))
        if base is None:
            raise RuntimeError("没有可写的数据目录")
        target = base.with_name(name)
        target.write_text(text, encoding="utf-8")
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
            line = fetch(persona=self._persona(character_id), character_name=character_name)
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
                character_name=character_name, persona=self._persona(character_id),
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

    def _sync_cleared_chat(self, character_id: str, *, confirm: bool = False) -> None:
        """聊天记录被清空时，情绪一起归零。

        看的是"这段对话还在不在"，不是"过了多久"——没有那段对话，就没有那段情绪留下的理由。
        读不到对话（服务异常、返回形状不对）一律不动手：宁可留一条旧情绪，
        也不要在她正在说话的时候把状态抹掉。
        """

        timeline = self._timeline
        if timeline is None:
            return
        now = float(self._clock())
        with self._lock:
            state = self._state(character_id)
            if not state.turn_count and not state.summary and not any(state.values.values()):
                return
            if (now - float(state.updated_at or 0.0)) < CHAT_SYNC_GRACE:
                return
            before_turn = int(state.turn_count)
        try:
            response = timeline.read_recent({"limit": 1})
        except Exception as error:  # noqa: BLE001
            self._log("debug", "读取最近对话失败，情绪先不动", {"error_type": type(error).__name__})
            return
        entries = response.get("entries") if isinstance(response, Mapping) else None
        if not isinstance(entries, list):
            return
        if entries:
            self._empty_chat_seen.pop(character_id, None)
            return
        if confirm:
            # 面板路径：要连续两次读到空才动手，免得一次瞬时读空把状态抹了。
            seen = self._empty_chat_seen.get(character_id, 0.0)
            if not seen or (now - seen) < 5.0:
                self._empty_chat_seen[character_id] = now
                return
        self._empty_chat_seen.pop(character_id, None)
        with self._lock:
            state = self._state(character_id)
            state.values = taxonomy.empty_pools()
            state.summary = ""
            state.main_streak = {}
            state.last_judge = None
            state.memory_ctx = None
            state.pending = {}
            state.episodes = []
            state.turn_count = 0
            state.updated_at = now
            self._save()
        self._log("info", "聊天记录已清空，情绪一起归零", {
            "character": str(character_id)[:32], "turn": before_turn,
        })

    # ---- 设置页与仪表盘 ----

    def dashboard_payload(self) -> dict[str, Any]:
        """仪表盘只拿数字：此刻在场的那两三根，和"上一次判断是什么时候"。

        人同时只有主·次·辅三个位置，所以面板也不再是二十一根条的清单——
        栏里那几根就是她此刻的全部。
        """

        character_id = self._current_character_id()
        self._sync_cleared_chat(character_id, confirm=True)
        with self._lock:
            state = self._state(character_id)
            last = state.last_judge if isinstance(state.last_judge, Mapping) else {}
            # 面板只显示"结算之后"的值：先判是预测，校正才是结果。
            # 这一轮还没结算时，继续显示上一轮结算的样子，别让数值来回跳。
            settled = last.get("settled") if isinstance(last.get("settled"), Mapping) else None
            shown = state
            if settled and isinstance(settled.get("values"), Mapping):
                shown = state_module.CharacterState()
                shown.values = {
                    key: max(0.0, float(settled["values"].get(key, 0.0))) for key in taxonomy.EMOTIONS
                }
                shown.summary = str(settled.get("summary") or "")
            described = state_module.describe(shown)
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
                    {"key": key, "value": prompt_module.percent(shown.values.get(key, 0.0))}
                    for key in taxonomy.EMOTIONS
                    if key not in present
                ),
                key=lambda item: -item["value"],
            )
            at = float(last.get("at") or 0.0)
            if at > 0 and settled and not last.get("rechecked"):
                updated = f"校正中：{prompt_module.ago(self._clock() - at)}先判了一次，等她说完再结算"
            elif at > 0:
                updated = (
                    f"上一次判断：{prompt_module.ago(self._clock() - at)} · "
                    f"{prompt_module.status_last(state, self._clock())['label']}"
                )
            else:
                updated = "还没有判断过：等她回一轮话，或点一次刷新"
            payload = {
                "slots": slots,
                "others": others,
                "baseline": (state.baseline or "")[:40],
                "updated": updated[:80],
                "turns": int(state.turn_count),
                # 她这一轮实际用了哪几种语气（先判 → 校正之后才有）
                "track": [str(item)[:8] for item in (last.get("track") or [])][:4],
                "rechecked": bool(last.get("rechecked")),
                "recheck_failed": bool(last.get("recheck_failed")),
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


def _average_verdicts(verdicts: list[Any]) -> Any:
    """把几份采样平均成一份：增减取均值，心境取出现最多的那句。"""

    if not verdicts:
        return None
    if len(verdicts) == 1:
        return verdicts[0]
    keys: set[str] = set()
    for item in verdicts:
        keys.update(getattr(item, "deltas", {}) or {})
    deltas: dict[str, float] = {}
    for key in keys:
        values = [float((getattr(item, "deltas", {}) or {}).get(key, 0.0)) for item in verdicts]
        mean = sum(values) / len(values)
        if abs(mean) >= 1:
            deltas[key] = round(mean, 1)
    summaries = [str(getattr(item, "summary", "") or "") for item in verdicts]
    summaries = [item for item in summaries if item]
    summary = max(set(summaries), key=summaries.count) if summaries else ""
    gists = [str(getattr(item, "gist", "") or "") for item in verdicts]
    gist = next((item for item in gists if item), "")
    return appraisal_model.ModelVerdict(
        deltas=deltas, values={}, roster=None, summary=summary, gist=gist,
    )


def _snapshot(state: Any) -> dict[str, Any]:
    """结算快照：面板只显示校正之后的值，不显示先判的中间态。"""

    return {
        "values": {key: round(float(value), 4) for key, value in state.values.items() if value > 0},
        "summary": str(getattr(state, "summary", "") or ""),
        "at": float(getattr(state, "updated_at", 0.0) or 0.0),
    }


def _pairs_text(values: object, limit: int = 6) -> str:
    """把 {"委屈": 24.0} 拼成 "委屈+24,失望-10"。

    宿主的日志净化器只接受 ASCII 键名（[A-Za-z][A-Za-z0-9_.-]*），
    直接传中文字典会被整条吃掉，日志里永远是 {}——排查时等于没有。
    值里刻意不放 "="：这一行是 "applied=委屈+24,失望-10"，若值里再出现等号，
    碰上按 = 切分的日志查看器会被切成 委屈 → 键名，整条记录就成非法形状了。
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
        parts.append(f"{key}{number:+.0f}")
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