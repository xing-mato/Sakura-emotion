"""插件设置（v2：只留开关与偏好，没有任何一个数字能改她的情绪）。

v1 里的反应强度 / 消退速度 / 最短保持 / 切换门槛全部删掉——那四个数是把工程参数
甩给用户，而且互相耦合，没人调得动。现在情绪怎么变只由模型判断。

留下的是真正属于"你的偏好"的东西：开不开、写多长、要不要和 Mem0 联动、留多少条涟漪。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import taxonomy

DETAIL_OPTIONS = (
    ("brief", "极简 · 一两行，几乎不占上下文"),
    ("standard", "标准 · 几句百分比 + 一句心境"),
    ("detailed", "细致 · 加上涟漪与相关往事"),
)

MEMORY_OPTIONS = (
    ("off", "不联动 · 情绪只存在本插件里"),
    ("read", "只读 · 借用 Mem0 的相关往事"),
    ("write", "只写 · 把显著的情绪记进 Mem0"),
    ("read_write", "读写 · 两边都接上"),
)

EDITABLE_KEYS = ("enabled", "detail", "memory_mode", "episode_keep")

MAX_EPISODES = 40


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "on", "yes", "enabled"}:
            return True
        if text in {"0", "false", "off", "no", "disabled"}:
            return False
    return default


def _as_choice(value: object, default: str, options: tuple[tuple[str, str], ...]) -> str:
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if text in {key for key, _label in options} else default


def _as_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        try:
            value = int(float(value.strip()))
        except (TypeError, ValueError):
            return default
    if not isinstance(value, (int, float)):
        return default
    return max(minimum, min(maximum, int(value)))


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    detail: str = "standard"
    memory_mode: str = "off"
    episode_keep: int = 16

    @property
    def memory_read(self) -> bool:
        return self.memory_mode in {"read", "read_write"}

    @property
    def memory_write(self) -> bool:
        return self.memory_mode in {"write", "read_write"}

    def values(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "detail": self.detail,
            "memory_mode": self.memory_mode,
            "episode_keep": int(self.episode_keep),
        }

    def normalized(self) -> dict[str, Any]:
        return self.values()


def parse(values: object) -> Settings:
    raw: Mapping[str, Any] = values if isinstance(values, Mapping) else {}
    defaults = Settings()
    return Settings(
        enabled=_as_bool(raw.get("enabled"), defaults.enabled),
        detail=_as_choice(raw.get("detail"), defaults.detail, DETAIL_OPTIONS),
        memory_mode=_as_choice(raw.get("memory_mode"), defaults.memory_mode, MEMORY_OPTIONS),
        episode_keep=_as_int(raw.get("episode_keep"), defaults.episode_keep, 0, MAX_EPISODES),
    )


def descriptor() -> dict[str, Any]:
    enabled_when = {"field": "enabled", "equals": "true"}
    return {
        "sectionId": "emotion_pool",
        "title": "情绪池",
        "order": 58,
        "presentation": {"component": "form", "alignedUnits": True, "inlineDescriptions": True},
        "fields": [
            {
                "key": "enabled",
                "label": "让她有自己的情绪",
                "type": "boolean",
                "default": True,
                "description": "总开关。关掉后不再注入，也不再判断；已有的状态原样保留，下次打开还是她刚才的心情。",
            },
            {
                "key": "detail",
                "label": "写进提示词的详细程度",
                "type": "select",
                "default": "standard",
                "options": [{"value": value, "label": label} for value, label in DETAIL_OPTIONS],
                "description": "只影响长度，不影响情绪本身。标准档大约 60 字。",
                "enabledWhen": enabled_when,
            },
            {
                "key": "memory_mode",
                "label": "和 Mem0 长期记忆联动",
                "type": "select",
                "default": "off",
                "options": [{"value": value, "label": label} for value, label in MEMORY_OPTIONS],
                "description": "不联动也完全能用。只读 = 每轮后去 Mem0 里找和这句话有关的往事，下一轮接在情绪后面；只写 = 显著的情绪事件写进 Mem0 的 episodic 层。需要已启用 Mem0 插件。",
                "enabledWhen": enabled_when,
            },
            {
                "key": "episode_keep",
                "label": "保留多少条情绪涟漪",
                "type": "integer",
                "default": 16,
                "minimum": 0,
                "maximum": MAX_EPISODES,
                "unit": "条",
                "description": "「哪句话让她动了多少」的记录，喂给下一轮的连续感，也在下面的记录表里能看到。0 = 不记录。",
                "enabledWhen": enabled_when,
            },
            {
                "key": "saved",
                "label": "状态存档",
                "type": "readonly",
                "description": "所有角色的情绪都存在这个文件里，可以复制路径备份或手工改。",
                "default": "",
                "maxLength": 512,
                "copyable": True,
            },
        ],
        # 清零 / 导出都放到左侧栏那个「情绪」页里了——那边是我们自己的 UI，
        # 能给真正的确认对话框；设置页的按钮只能弹一行字，看不见就会被当成坏了。
        "actions": [
            {"actionId": "test_memory", "label": "测试长期记忆桥", "danger": False,
             "description": "检查能不能通过工具目录找到 Mem0，并试着检索一次"},
        ],
    }


def collection_descriptor() -> dict[str, Any]:
    return {
        "collectionId": "episodes",
        "title": "情绪涟漪 · 最近发生过什么",
        "description": "每一条是一次真正被触动的记录：因为哪句话、哪根弦、动了几个点。"
                       "这些也会挑两条写进提示词，让她记得自己刚才为什么那样。",
        "scope": "global",
        "pageSize": 20,
        "searchable": False,
        "deleteConfirmation": "",
        "filters": [],
        "fields": [],
        "columns": [
            {"key": "at", "label": "时间", "type": "datetime"},
            {"key": "emotion", "label": "情绪", "type": "string", "maxLength": 16},
            {"key": "level", "label": "变化", "type": "string", "maxLength": 8},
            {"key": "source", "label": "来源", "type": "string", "maxLength": 16},
            {"key": "gist", "label": "因为什么", "type": "string", "maxLength": 120},
        ],
    }


def dashboard_descriptor(source: str) -> dict[str, Any]:
    """左侧栏「情绪」页里那块仪表盘：一个自定义前端模块（胶囊条）。

    宿主给插件页面只开放 form / module / record-table / connection-status 四种画法，
    想要"圆角胶囊 + 渐变"只能走 module：把 JS 源码整段塞进 presentation.source，
    宿主把它当 ES module 导入并调用 mount()。数据通过一个只读 data 字段 payload 过去。
    """

    return {
        "sectionId": "emotion_pool_view",
        "title": "情绪",
        "order": 30,
        "presentation": {"component": "module", "source": source},
        "fields": [
            {
                "key": "payload",
                "label": "情绪数据",
                "type": "data",
                "readonly": True,
                "description": "仪表盘读的原始数据：此刻在场的主·次·辅，收进抽屉的其余情绪，和上一次判断的时间。",
                "default": {},
            },
        ],
        "actions": [
            {"actionId": "refresh", "label": "刷新", "danger": False,
             "description": "按最近的对话，请模型重新判断一次情绪"},
            # 面板每隔几秒悄悄调它一次，只读当前数值、不碰模型。
            # 宿主要求所有 action 都先声明，所以它也得在这里；module 类型不渲染宿主按钮。
            {"actionId": "peek", "label": "同步", "danger": False,
             "description": "只读取当前情绪值，不调用模型"},
            # 下面两个由面板自己的确认对话框触发：危险动作只有这一条路，
            # 因为设置页的按钮弹不出窗口（看不见的警告等于没有）。
            {"actionId": "reset", "label": "清零", "danger": False,
             "description": "清空当前角色的情绪值（涟漪与聊天记录保留）。面板里点会先弹确认"},
            {"actionId": "export_log", "label": "导出日志", "danger": False,
             "description": "导出当前情绪值、逐轮判定与聊天原文。面板里点会先弹隐私确认"},
        ],
    }


__all__ = [
    "DETAIL_OPTIONS", "EDITABLE_KEYS", "MAX_EPISODES", "MEMORY_OPTIONS", "Settings",
    "collection_descriptor", "dashboard_descriptor", "descriptor", "parse",
]
