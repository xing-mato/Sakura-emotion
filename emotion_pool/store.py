"""情绪状态的落盘：一个 JSON 文件，原子写，坏了不连累插件。

v3 起文件头是 version: 3：二十一根弦的 values（0~1 的浮点）+ summary。
v2 的十根弦、v1 的 pools/mood 都能直接读进来（CharacterState.from_dict 认得旧键，
缺的键补 0），下一次保存时自动写成 v3。

文件长这样：

    {
      "version": 1,
      "characters": {
        "Sakura": { "pools": {...}, "mood": -0.21, "dominant": "委屈", ... }
      }
    }

为什么不用 SQLite：一份状态只有一个写者、一次最多几 KB、每轮写一次，
原子替换一个文件就够可靠；出问题时用户还能直接用记事本看懂并改回来。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from . import state as state_module

VERSION = 3


class StateStore:
    def __init__(self, path: Path, *, logger: Any = None) -> None:
        self._path = Path(path)
        self._logger = logger
        self._lock = threading.RLock()
        self._last_payload: str | None = None
        self._loaded = False
        self._doc: dict[str, Any] = {"version": VERSION, "characters": {}}

    @property
    def path(self) -> Path:
        return self._path

    def path_text(self) -> str:
        return str(self._path)

    def load(self) -> dict[str, state_module.CharacterState]:
        with self._lock:
            if not self._loaded:
                self._doc = self._read()
                self._loaded = True
            raw = self._doc.get("characters")
            characters: dict[str, state_module.CharacterState] = {}
            if isinstance(raw, Mapping):
                for character_id, record in raw.items():
                    key = str(character_id)[:128] or "default"
                    characters[key] = state_module.CharacterState.from_dict(record)
            return characters

    def save(self, characters: Mapping[str, state_module.CharacterState]) -> bool:
        with self._lock:
            payload = {
                "version": VERSION,
                "updated_at": int(time.time()),
                "characters": {
                    str(character_id): item.to_dict()
                    for character_id, item in characters.items()
                },
            }
            try:
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                self._log("warning", "情绪状态无法序列化，这次不落盘")
                return False
            if encoded == self._last_payload:
                return False
            try:
                self._write(encoded)
            except OSError as error:
                self._log("warning", "情绪状态写入失败", {"error_type": type(error).__name__})
                return False
            self._last_payload = encoded
            self._doc = payload
            return True

    # ---- 内部 ----

    def _read(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"version": VERSION, "characters": {}}
        try:
            text = self._path.read_text(encoding="utf-8")
            doc = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self._quarantine(error)
            return {"version": VERSION, "characters": {}}
        if not isinstance(doc, dict):
            self._quarantine(ValueError("根节点不是对象"))
            return {"version": VERSION, "characters": {}}
        return doc

    def _quarantine(self, error: BaseException) -> None:
        """坏文件挪到一边：下次还能人工看一眼，但不会反复把插件拖垮。"""

        target = self._path.with_name(f"{self._path.stem}.corrupt-{int(time.time())}{self._path.suffix}")
        try:
            os.replace(self._path, target)
        except OSError:
            target = Path("")
        self._log("warning", "情绪状态文件损坏，已另存并重新开始", {
            "error_type": type(error).__name__,
            "quarantined": str(target) if target else "",
        })

    def _write(self, encoded: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, self._path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> None:
        logger = self._logger
        if logger is None:
            return
        try:
            getattr(logger, severity)(message, fields=dict(fields or {}))
        except Exception:  # noqa: BLE001 - 日志失败不影响状态
            pass


__all__ = ["StateStore", "VERSION"]
