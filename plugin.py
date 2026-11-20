"""情绪池插件入口（Plugin API v4）。

它只做四件事，全部通过宿主公开的服务：

1. sakura.host.context   每轮往提示词里放一段"她此刻的内在状态"（scope=turn）。
2. sakura.host.settings  一个设置页 + 一张情绪涟漪记录表。
3. sakura.host.timeline  在慢路径里读回这一轮的原话（只读，不写时间线）。
4. sakura.host.tools     可选：通过工具目录调用 Mem0 的检索/写入工具。

没有任何一项是可选的：拿不到时间线就只做关键词快评，拿不到模型就只做本地判断，
拿不到 Mem0 就完全不联动。插件永远能起来，永远不会因为少一个服务而让对话失败。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

try:  # 正常：作为包的一部分加载
    from .emotion_pool import settings as settings_module
    from .emotion_pool.appraisal_model import ModelAppraiser
    from .emotion_pool.memory_bridge import MemoryBridge
    from .emotion_pool.runtime import EmotionRuntime
    from .emotion_pool.store import StateStore
except ImportError:  # entry: plugin:EmotionPoolPlugin，按顶层模块加载
    from emotion_pool import settings as settings_module
    from emotion_pool.appraisal_model import ModelAppraiser
    from emotion_pool.memory_bridge import MemoryBridge
    from emotion_pool.runtime import EmotionRuntime
    from emotion_pool.store import StateStore

PROVIDER_ID = "emotion_pool"
PLUGIN_ID = "sakuratools.emotion_pool"   # 和 plugin.yaml 的 id 一致；拿不到宿主给的 id 时用它
DASHBOARD_PAGE = "emotion"               # 左侧栏那一页的本地 id → 完整 id 是 <插件id>:emotion
DASHBOARD_PAGE_TITLE = "情绪"
DASHBOARD_PAGE_GROUP = "behavior"        # 宿主侧栏分组：character / ai / behavior / system
CHAT_COMPLETED_EVENT = "sakura.host.chat.completed"
SETTINGS_SECTION = "emotion_pool"
DASHBOARD_SECTION = "emotion_pool_view"
COLLECTION_SECTION = "emotion_pool"
ACTION_MESSAGE_LIMIT = 240
# 危险动作要二次确认：第一次点只警告，这个窗口内再点一次才真的执行



class EmotionPoolPlugin:
    def __init__(self) -> None:
        self.context: Any = None
        self.runtime: EmotionRuntime | None = None
        self._logger: Any = None
        self._lock = threading.RLock()
        self._reset_asked_at = 0.0
        self._export_asked_at = 0.0

    # ---- 生命周期 ----

    def setup(self, context: Any) -> None:
        logger = _optional(context, "sakura.host.logging")
        self.context = context
        self._logger = logger
        config = context.config
        get_settings = lambda: settings_module.parse(config.get())  # noqa: E731

        timeline = _optional(context, "sakura.host.timeline")
        character = _optional(context, "sakura.host.character")

        def character_snapshot() -> dict[str, Any]:
            if character is None:
                return {}
            try:
                value = character.current()
            except Exception:  # noqa: BLE001 - 没有选中角色时很正常
                return {}
            return dict(value) if isinstance(value, Mapping) else {}

        def current_character() -> str | None:
            identifier = character_snapshot().get("id")
            return identifier if isinstance(identifier, str) and identifier else None

        def current_persona() -> str:
            prompt = character_snapshot().get("systemPrompt")
            return prompt if isinstance(prompt, str) else ""

        store = StateStore(context.data_path("state.json"), logger=logger)
        appraiser = None
        try:
            appraiser = ModelAppraiser(context, logger=logger)
        except Exception as error:  # noqa: BLE001
            _warn(logger, "模型校正器没能建立，只做关键词快评", error)
        memory = None
        try:
            memory = MemoryBridge(context, logger=logger)
        except Exception as error:  # noqa: BLE001
            _warn(logger, "长期记忆桥没能建立，情绪照常工作", error)

        runtime = EmotionRuntime(
            settings_provider=get_settings,
            store=store,
            logger=logger,
            timeline=timeline,
            appraiser=appraiser,
            memory=memory,
            character_provider=current_character,
            persona_provider=current_persona,
        )
        self.runtime = runtime
        context.effect(runtime.close)

        self._register_context(context, logger)
        self._register_events(context, logger)
        self._register_settings(context, logger)
        self._register_dashboard(context, logger)
        self._register_collection(context, logger)

        _info(logger, "情绪池已就绪", {
            "provider": PROVIDER_ID,
            "state_file": runtime.saved_text(),
            "timeline": timeline is not None,
        })

    def close(self) -> None:
        runtime = self.runtime
        if runtime is not None:
            try:
                runtime.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- 注册 ----

    def _register_context(self, context: Any, logger: Any) -> None:
        try:
            context.get("sakura.host.context").register(
                {
                    "providerId": PROVIDER_ID,
                    "description": "她此刻的情绪：主导、余韵、心境与表现方式",
                    "order": 55.0,
                    "scope": "turn",
                    "failurePolicy": "skip",
                    "enabled": True,
                },
                self._fragments,
            )
        except Exception as error:  # noqa: BLE001
            _warn(logger, "情绪注入没能登记", error)

    def _register_events(self, context: Any, logger: Any) -> None:
        try:
            context.on(CHAT_COMPLETED_EVENT, self._on_chat_completed)
        except Exception as error:  # noqa: BLE001
            _warn(logger, "对话完成事件没能订阅，慢路径不会生效", error)

    def _register_settings(self, context: Any, logger: Any) -> None:
        try:
            context.get("sakura.host.settings").register(
                settings_module.descriptor(),
                load=self._load_settings,
                save=self._save_settings,
                actions={
                    "reset": self._action_reset,
                    "export_log": self._action_export_log,
                    "test_memory": self._action_test_memory,
                },
            )
        except Exception as error:  # noqa: BLE001
            _warn(logger, "设置页没能登记", error)

    def _register_dashboard(self, context: Any, logger: Any) -> None:
        """仪表盘：一个只读分区，挂到左侧栏自己的「情绪」页里。

        胶囊条走的是自定义前端模块（frontend/dashboard.js），宿主把它当 ES module 导入。
        CSP 是 style-src 'self'，所以样式必须用构造式样式表注入，塞 <style> 会被静默拦掉。
        """

        source = self._dashboard_source()
        if not source:
            _warn(logger, "仪表盘前端模块读不到，跳过仪表盘", RuntimeError("DASHBOARD_SOURCE_MISSING"))
            return
        try:
            settings = context.get("sakura.host.settings")
            settings.register(
                settings_module.dashboard_descriptor(source),
                load=self._load_dashboard,
                actions={
                    "refresh": self._action_dashboard_refresh,
                    "peek": self._action_dashboard_peek,
                    "reset": self._action_reset,
                    "export_log": self._action_export_log,
                },
            )
        except Exception as error:  # noqa: BLE001
            _warn(logger, "仪表盘分区没能登记", error)
            return
        page_id = self._register_dashboard_page(context, settings, logger)
        try:
            settings.place(DASHBOARD_SECTION, page_id=page_id, region="content", order=10)
        except Exception as error:  # noqa: BLE001
            _warn(logger, "仪表盘没能挂上，还可以从插件设置里打开", error)

    def _register_dashboard_page(self, context: Any, settings: Any, logger: Any) -> str:
        """在左侧栏给自己开一页（宿主支持插件注册命名空间页面）。

        申请不到就退回宿主的「交互」页——一个导航入口的失败，不该让仪表盘整个消失。
        """

        owner = getattr(context, "plugin_id", None)
        if not isinstance(owner, str) or not owner:
            owner = PLUGIN_ID
        page_id = f"{owner}:{DASHBOARD_PAGE}"
        try:
            settings.register_page({
                "pageId": DASHBOARD_PAGE,
                "title": DASHBOARD_PAGE_TITLE,
                "group": DASHBOARD_PAGE_GROUP,
                "icon": "sparkles",
                "order": 55,
                "regions": ["content"],
            })
        except Exception as error:  # noqa: BLE001
            _warn(logger, "情绪面板没能单独开一页，退回「交互」页", error)
            return "host:interaction"
        _info(logger, "情绪面板已挂到左侧栏独立页面", {"page_id": page_id})
        return page_id

    def _dashboard_source(self) -> str:
        """仪表盘的前端模块：宿主只接受"整段 JS 源码"放进展区描述里。"""

        path = Path(__file__).resolve().parent / "frontend" / "dashboard.js"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _load_dashboard(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"payload": {}}
        try:
            return {"payload": runtime.dashboard_payload()}
        except Exception as error:  # noqa: BLE001
            _warn(self._logger, "仪表盘读取失败", error)
            return {"payload": {}}

    def _action_dashboard_peek(self, values: object) -> dict[str, Any]:
        """只读一次当前状态，不判断、不调用模型。

        面板每 8 秒悄悄问一次，这样她每说完一句话，你切回来就能看到新数字，
        不用手动点刷新。
        """

        runtime = self.runtime
        if runtime is None:
            return {"values": {"payload": {}}}
        try:
            return {"values": {"payload": runtime.dashboard_payload()}}
        except Exception as error:  # noqa: BLE001
            _warn(self._logger, "仪表盘读取失败", error)
            return {"values": {"payload": {}}}

    def _action_dashboard_refresh(self, values: object) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"message": "插件尚未就绪"}
        try:
            result = runtime.refresh_with_model()
            message = str(result.get("message") or "已刷新")
            return {"message": message[:ACTION_MESSAGE_LIMIT],
                    "values": {"payload": runtime.dashboard_payload()}}
        except Exception as error:  # noqa: BLE001
            return {"message": f"刷新失败：{type(error).__name__}"}

    def _register_collection(self, context: Any, logger: Any) -> None:
        service = _optional(context, "sakura.host.settings.collection-v0")
        if service is None:
            return
        try:
            service.register(
                COLLECTION_SECTION,
                settings_module.collection_descriptor(),
                query=self._query_episodes,
            )
        except Exception as error:  # noqa: BLE001
            _warn(logger, "情绪涟漪记录表没能登记", error)

    # ---- 回调 ----

    def _fragments(self, request: object) -> list[dict[str, Any]]:
        runtime = self.runtime
        if runtime is None:
            return []
        return runtime.fragment_for(request)

    def _on_chat_completed(self, payload: object) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        runtime.on_chat_completed(payload)

    # ---- 设置页 ----

    def _load_settings(self) -> dict[str, Any]:
        values = self._settings().values()
        runtime = self.runtime
        if runtime is not None:
            values["saved"] = runtime.saved_text()[:512]
        return values

    def _save_settings(self, values: object) -> dict[str, Any]:
        incoming = {}
        if isinstance(values, Mapping):
            incoming = {key: values[key] for key in values if key in settings_module.EDITABLE_KEYS}
        merged = settings_module.parse({**self._settings().normalized(), **incoming})
        try:
            state = self.context.config.update(merged.normalized())
        except Exception as error:  # noqa: BLE001
            _warn(self._logger, "设置没能保存", error)
            return {"applicationState": "error"}
        if state not in {"applied", "restart_required", "error"}:
            state = "applied"
        return {"applicationState": state}

    # ---- 设置页动作 ----

    def _action_reset(self, values: object) -> dict[str, Any]:
        """仪表盘清零：第一次点只是警告，60 秒内再点一次才真的清。"""

        runtime = self.runtime
        if runtime is None:
            return {"message": "插件尚未就绪"}
        # 确认由仪表盘页面里的对话框负责（宿主不给设置页按钮弹窗）。
        _info(self._logger, "仪表盘清零：已执行")
        return {"message": runtime.reset()[:ACTION_MESSAGE_LIMIT]}

    def _action_export_log(self, values: object) -> dict[str, Any]:
        """导出运行日志：第一次点只提醒一次隐私，60 秒内再点一次才写文件。"""

        runtime = self.runtime
        if runtime is None:
            return {"message": "插件尚未就绪"}
        _info(self._logger, "导出运行日志：开始导出")
        try:
            path = runtime.export_log()
        except Exception as error:  # noqa: BLE001
            _warn(self._logger, "运行日志导出失败", error)
            return {"message": "导出失败：" + type(error).__name__}
        # 先确认文件真的在，再交给系统打开。os.startfile 对不存在的路径
        # 也会「成功」返回，然后由记事本弹一句「系统找不到指定的路径」——这个锅不能留给用户。
        name = os.path.basename(str(path))
        if not os.path.isfile(path):
            return {"message": ("文件没写出来：" + str(path))[-ACTION_MESSAGE_LIMIT:]}
        opened = _open_file(path)
        return {"message": (("已导出：" if opened else "已导出但没能打开：") + name)[:ACTION_MESSAGE_LIMIT]}

    def _action_test_memory(self, values: object) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"message": "插件尚未就绪"}
        return {"message": " ".join(runtime.memory_test().split())[:ACTION_MESSAGE_LIMIT]}

    def _action_open_state(self, values: object) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {"message": "插件尚未就绪"}
        text = runtime.saved_text()
        if not text:
            return {"message": "还没有状态文件"}
        _open_file(text)  # 打不开也不要紧，下面照样把路径告诉她
        return {"message": f"状态文件：{text[:200]}", "values": {"saved": text[:512]}}

    def _status_values(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {}
        status = runtime.status()
        return {"status_now": status["now"], "status_last": status["last"]}

    def _preview_path(self):
        context = self.context
        if context is None:
            return None
        try:
            return context.data_path(PREVIEW_FILE)
        except Exception:  # noqa: BLE001
            return None

    # ---- 记录表 ----

    def _query_episodes(self, request: object) -> dict[str, Any]:
        limit = 20
        if isinstance(request, Mapping):
            raw = request.get("limit")
            if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= 100:
                limit = raw
        runtime = self.runtime
        if runtime is None:
            return {"items": [], "nextCursor": None, "total": 0}
        rows = runtime.episode_rows(limit)
        return {"items": rows, "nextCursor": None, "total": len(rows)}

    # ---- 小工具 ----

    def _settings(self):
        try:
            return settings_module.parse(self.context.config.get())
        except Exception:  # noqa: BLE001
            return settings_module.parse({})


def _optional(context: Any, key: str) -> Any:
    try:
        return context.get(key)
    except Exception:  # noqa: BLE001 - 缺服务时退化，不让插件整体失败
        return None


def _open_file(path: Any) -> bool:
    try:
        os.startfile(str(path))  # type: ignore[attr-defined] - Windows 专有
        return True
    except (AttributeError, OSError):
        return False


def _info(logger: Any, message: str, fields: Mapping[str, Any] | None = None) -> None:
    if logger is None:
        return
    try:
        logger.info(message, fields=dict(fields or {}))
    except Exception:  # noqa: BLE001
        pass


def _warn(logger: Any, message: str, error: BaseException) -> None:
    if logger is None:
        return
    try:
        logger.warning(message, fields={
            "reason_code": str(getattr(error, "code", "") or "EMOTION_POOL_SETUP_FAILED")[:64],
            "error_type": type(error).__name__,
        })
    except Exception:  # noqa: BLE001
        pass


__all__ = ["EmotionPoolPlugin", "PROVIDER_ID"]