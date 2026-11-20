"""和 Mem0 长期记忆的桥。

先说实话：Mem0 插件（sakura.memory.mem0）的 manifest 里 provides 是空的——它没有
对外暴露任何 service key。所以"直接调用 Mem0 的接口"这条路不存在，硬读它的数据
目录更是越界。

它真正公开的只有一样东西：注册在 sakura.host.tools 里的四个工具
（memory_search / memory_remember / memory_update / memory_forget）。而
sakura.host.tools 的 catalog + execute 是宿主 SDK 里公开的调用面，于是这里就用它：

- recall()   -> memory_search(query)    拿回和这句话有关的往事
- remember() -> memory_remember(content) 把显著的情绪事件写成一条 episodic 记忆

三条硬约束，避免把别人的记忆系统搞乱或拖垮：

1. **永远不阻塞对话**。两个调用都只在后台线程里发生，并且都带 timeout_seconds；
   Mem0 没启用、没选整理模型、向量库还在加载，一律当"没拿到"。
2. **写入有闸门**：只有强度 >= 0.6 的情绪、间隔足够长、且用户明确开了"只写/读写"
   才写；一次最多一条，绝不刷屏。
3. **发现结果缓存**，工具目录每 5 分钟才重新读一次，别为每轮对话打一次服务。

如果哪天 Mem0 真的提供了 service key，这里换成 bind 即可，上层不用动。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

SEARCH_TOOL = "memory_search"
REMEMBER_TOOL = "memory_remember"
DISCOVERY_TTL = 300.0
CALL_TIMEOUT = 8.0
MAX_CONTENT = 180


class MemoryBridge:
    def __init__(self, context: Any, *, logger: Any = None) -> None:
        self._context = context
        self._logger = logger
        self._lock = threading.RLock()
        self._tools: Any = None
        self._discovered_at = 0.0
        self._entry: dict[str, str] | None = None
        self._reason = "还没检查"

    # ---- 发现 ----

    def discover(self, *, force: bool = False) -> dict[str, str] | None:
        with self._lock:
            fresh = (not force and self._entry is not None
                     and (time.monotonic() - self._discovered_at) < DISCOVERY_TTL)
            if fresh:
                return self._entry
            self._entry = self._probe()
            self._discovered_at = time.monotonic()
            return self._entry

    def available(self) -> bool:
        return self.discover() is not None

    def status_text(self) -> str:
        entry = self.discover()
        if entry is None:
            return f"没有找到可用的记忆工具（{self._reason}）"
        found = "、".join(sorted(entry))
        return f"找到记忆工具：{found}"

    def _probe(self) -> dict[str, str] | None:
        try:
            tools = self._context.get("sakura.host.tools")
            catalog = tools.catalog()
        except Exception as error:  # noqa: BLE001
            self._reason = f"工具目录读不到（{type(error).__name__}）"
            return None
        if not isinstance(catalog, list):
            self._reason = "工具目录格式不对"
            return None
        found: dict[str, str] = {}
        for item in catalog:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            registration = item.get("registrationId")
            if not isinstance(name, str) or not isinstance(registration, str) or not registration:
                continue
            if item.get("source") != "plugin":
                continue
            if name == SEARCH_TOOL and "search" not in found:
                found["search"] = registration
            elif name == REMEMBER_TOOL and "remember" not in found:
                found["remember"] = registration
        if not found:
            self._reason = "Mem0 插件可能没有启用"
            return None
        self._tools = tools
        self._reason = "可用"
        return found

    def _registration(self, role: str) -> tuple[Any, str] | None:
        entry = self.discover()
        if entry is None:
            return None
        registration = entry.get(role)
        if not registration:
            return None
        tools = self._tools
        if tools is None:
            try:
                tools = self._context.get("sakura.host.tools")
            except Exception:  # noqa: BLE001
                return None
        return tools, registration

    # ---- 读 ----

    def recall(self, query: str, *, limit: int = 3) -> list[str]:
        text = " ".join((query or "").split())[:120]
        if len(text) < 4:
            return []
        found = self._registration("search")
        if found is None:
            return []
        tools, registration = found
        try:
            response = tools.execute(registration, SEARCH_TOOL,
                                     {"query": text, "limit": max(1, min(5, int(limit)))},
                                     timeout_seconds=CALL_TIMEOUT)
        except Exception as error:  # noqa: BLE001
            self._log("debug", "长期记忆检索失败", {"reason_code": str(getattr(error, "code", "") or "RECALL_FAILED")[:48]})
            return []
        memories = response.get("memories") if isinstance(response, Mapping) else None
        if not isinstance(memories, list):
            return []
        lines: list[str] = []
        for item in memories:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(" ".join(content.split())[:80])
            if len(lines) >= 3:
                break
        return lines

    # ---- 写 ----

    def remember(
        self,
        content: str,
        *,
        importance: float = 0.6,
        category: str = "emotion",
    ) -> bool:
        body = " ".join((content or "").split())[:MAX_CONTENT]
        if len(body) < 6:
            return False
        found = self._registration("remember")
        if found is None:
            return False
        tools, registration = found
        payload = {
            "content": body,
            "layer": "episodic",
            "category": category[:32],
            "importance": max(0.1, min(1.0, float(importance))),
            "confidence": 0.7,
        }
        try:
            tools.execute(registration, REMEMBER_TOOL, payload, timeout_seconds=CALL_TIMEOUT)
        except Exception as error:  # noqa: BLE001
            self._log("warning", "情绪记忆写入失败", {
                "reason_code": str(getattr(error, "code", "") or "REMEMBER_FAILED")[:48],
            })
            return False
        return True

    # ---- 设置页动作 ----

    def test(self, *, probe: str = "他上次说过喜欢什么") -> str:
        entry = self.discover(force=True)
        if entry is None:
            return self.status_text()
        lines = [self.status_text()]
        try:
            hits = self.recall(probe, limit=3)
        except Exception as error:  # noqa: BLE001
            hits = []
            lines.append(f"检索时出错：{type(error).__name__}")
        if hits:
            lines.append("试检索「" + probe + "」拿到：" + " / ".join(hits))
        else:
            lines.append("试检索「" + probe + "」什么都没拿到——"
                         "可能是记忆库还是空的，或者向量模型没装好。写入功能不受影响。")
        if "remember" not in entry:
            lines.append("注意：没找到 memory_remember，写入会被跳过。")
        return "\n".join(lines)

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> None:
        if self._logger is None:
            return
        try:
            getattr(self._logger, severity)(message, fields=dict(fields or {}))
        except Exception:  # noqa: BLE001
            pass


__all__ = ["DISCOVERY_TTL", "MAX_CONTENT", "MemoryBridge", "REMEMBER_TOOL", "SEARCH_TOOL"]
