"""情绪池自己的模型槽：判断和聊天分开选模型。

判断和聊天是两种活。聊天要的是文笔、临场、读得懂话里有话；判断要的是按她的角色设定
把一件事的分量算对，输出只有几十个 token。绑在同一个模型上，换哪个都别扭——
聊天换强的，判断跟着变贵（一轮三次采样）；判断换便宜的，聊天跟着变笨。

宿主允许插件登记自己的模型槽（identity = plugin:<插件id>:<slotId>），
登记后会出现在 Sakura 设置 → 模型里，可以单独选服务、模型。

没配的时候一律回落到对话模型，所以不登记、不配置，插件照常工作。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

SLOT_ID = "judge"
SLOT_LABEL = "情绪判断模型"
SLOT_DESCRIPTION = "情绪池用它判断她此刻的情绪。留空则沿用对话模型。"
SLOT_ORDER = 30.0
EMPTY = {"serviceKey": "", "profileId": "", "modelId": ""}
KEYS = ("serviceKey", "profileId", "modelId")


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _selection(raw: object) -> dict[str, str]:
    data = raw if isinstance(raw, Mapping) else {}
    return {key: _text(data.get(key)) for key in KEYS}


class ModelSlot:
    """宿主里那个"情绪判断模型"槽：负责存取选择，并把选择和 host 对好。"""

    def __init__(self, path: Path, *, logger: Any = None) -> None:
        self._path = Path(path)
        self._logger = logger

    # ---- 宿主回调 ----

    def load(self) -> dict[str, str]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(EMPTY)
        return _selection(raw)

    def save(self, selection: object) -> dict[str, str]:
        data = _selection(selection)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._path.with_name(self._path.name + ".tmp")
            temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            temp.replace(self._path)
        except OSError as error:
            self._log("warning", "判断模型的选择没能存下来", {"error_type": type(error).__name__})
            return {"applicationState": "error"}
        self._log("info", "判断模型已更新", {
            "model": data["modelId"] or "（留空，沿用对话模型）",
        })
        return {"applicationState": "applied"}

    # ---- 给判断器用 ----

    def reference(self) -> dict[str, str] | None:
        """配了就用，没配返回 None（判断器回落到对话模型）。"""

        data = self.load()
        if not all(data[key] for key in KEYS):
            return None
        return data

    def register(self, context: Any) -> bool:
        try:
            slots = context.get("sakura.host.model_slots.v2")
            register = getattr(slots, "register", None)
            if not callable(register):
                return False
            register(
                {
                    "slotId": SLOT_ID,
                    "label": SLOT_LABEL,
                    "description": SLOT_DESCRIPTION,
                    "modelKind": "chat_completion",
                    "required": False,
                    "order": SLOT_ORDER,
                },
                load=self.load,
                save=self.save,
            )
            return True
        except Exception as error:  # noqa: BLE001 - 登记不上就沿用对话模型
            self._log("debug", "判断模型的槽没能登记", {"error_type": type(error).__name__})
            return False

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> None:
        if self._logger is None:
            return
        try:
            getattr(self._logger, severity)(message, fields=dict(fields or {}))
        except Exception:  # noqa: BLE001
            pass


__all__ = ["EMPTY", "ModelSlot", "SLOT_ID"]
