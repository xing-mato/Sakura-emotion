"""离线自检：不启动 Sakura，也能验证情绪池 v2 的每条规则。

    python selftest.py           # 跑全部用例
    python selftest.py -v        # 顺便把每轮渲染出来的提示词打印出来

用一套假宿主（FakeContext）代替 Sakura：真的插件入口会被完整 setup 一遍，
所以设置描述符、上下文注册、记录表、仪表盘放置、事件订阅都会走真实注册流程。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="surrogateescape")
    except (AttributeError, OSError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from emotion_pool import appraisal_model, prompt, settings as settings_module  # noqa: E402
from emotion_pool import state as state_module  # noqa: E402
from emotion_pool import taxonomy  # noqa: E402
from emotion_pool.memory_bridge import MemoryBridge  # noqa: E402
from emotion_pool.runtime import EmotionRuntime  # noqa: E402
from emotion_pool.store import StateStore  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


# ---------------------------------------------------------------- 假宿主

class FakeLogger:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict]] = []

    def _log(self, severity: str, message: str, fields: Mapping[str, Any] | None = None) -> bool:
        self.lines.append((severity, message, dict(fields or {})))
        return True

    def debug(self, message: str, *, fields=None) -> bool:
        return self._log("debug", message, fields)

    def info(self, message: str, *, fields=None) -> bool:
        return self._log("info", message, fields)

    def warning(self, message: str, *, fields=None) -> bool:
        return self._log("warning", message, fields)

    def error(self, message: str, *, fields=None) -> bool:
        return self._log("error", message, fields)


class FakeConfig:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self) -> dict[str, Any]:
        return dict(self.data)

    def update(self, values: Mapping[str, Any]) -> str:
        self.data.update(dict(values))
        return "applied"

    def replace(self, values: Mapping[str, Any]) -> str:
        self.data = dict(values)
        return "applied"

    def on_change(self, handler) -> Any:
        return lambda: None


class _Registration:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeRegistrationProxy:
    def __init__(self, holder: dict, kind: str) -> None:
        self._holder = holder
        self._kind = kind

    def register(self, *args: Any, **kwargs: Any) -> Any:
        self._holder[self._kind] = _Registration(args=args, kwargs=kwargs)
        return lambda: self._holder.pop(self._kind, None)

    def register_page(self, descriptor: Any) -> Any:
        self._holder.setdefault("pages", []).append(dict(descriptor))
        return lambda: None

    def place(self, section_id: str, *, page_id: str, region: str = "content", order: float = 100):
        self._holder.setdefault("placements", []).append({
            "sectionId": section_id, "pageId": page_id, "region": region, "order": order,
        })
        return lambda: None

    def catalog(self) -> list:
        return self._holder.get("catalog", [])

    def execute(self, registration_id: str, name: str, arguments: Mapping[str, Any], *, timeout_seconds=None):
        self._holder.setdefault("calls", []).append((name, dict(arguments)))
        handler = self._holder.get("handler")
        if handler is not None:
            return handler(name, dict(arguments))
        return {"status": "ready", "memories": []}


class FakeTimeline:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def read_recent(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {"entries": list(self.entries[-int(request.get("limit", 8)):])}


class FakeCharacter:
    def __init__(self, character_id: str = "Sakura", persona: str = "她冷静克制，不喜欢撒娇。") -> None:
        self.character_id = character_id
        self.persona = persona

    def current(self) -> dict[str, str]:
        return {"id": self.character_id, "systemPrompt": self.persona}


class FakeContext:
    def __init__(self, root: Path, *, catalog: list | None = None, tools_handler=None,
                 timeline: FakeTimeline | None = None, character: FakeCharacter | None = None) -> None:
        self.root = root
        self.plugin_id = "sakuratools.emotion_pool"
        self.config = FakeConfig()
        self.logger = FakeLogger()
        self.effects: list = []
        self.events: dict[str, list] = {}
        self.holders: dict[str, dict] = {}
        self.slots: dict[str, Any] = {}
        self.timeline = timeline or FakeTimeline()
        self.character = character or FakeCharacter()
        if catalog is not None:
            self.holders["sakura.host.tools"] = {"catalog": catalog, "handler": tools_handler}

    def get(self, key: str):
        if key == "sakura.host.logging":
            return self.logger
        if key == "sakura.host.context":
            return FakeRegistrationProxy(self.holders.setdefault("context", {}), "provider")
        if key == "sakura.host.settings":
            return FakeRegistrationProxy(self.holders.setdefault("settings", {}), "section")
        if key == "sakura.host.settings.collection-v0":
            return FakeRegistrationProxy(self.holders.setdefault("collection", {}), "table")
        if key == "sakura.host.tools":
            return FakeRegistrationProxy(self.holders.setdefault("sakura.host.tools", {}), "tools")
        if key == "sakura.host.model_slots.v2":
            return _ModelSlots(self.slots)
        if key == "sakura.host.timeline":
            return self.timeline
        if key == "sakura.host.character":
            return self.character
        raise KeyError(key)

    def data_path(self, relative: str) -> Path:
        return self.root / relative

    def effect(self, cleanup):
        self.effects.append(cleanup)
        return lambda: None

    def on(self, name: str, handler):
        self.events.setdefault(name, []).append(handler)
        return lambda: None


class _ModelSlots:
    def __init__(self, slots: dict) -> None:
        self._slots = slots

    def active(self) -> dict:
        return dict(self._slots)


# ---------------------------------------------------------------- 工具

def check(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as error:
        FAILED.append((name, str(error)))
        print(f"FAIL  {name}: {error}")
    except Exception as error:  # noqa: BLE001
        FAILED.append((name, traceback.format_exc(limit=3)))
        print(f"ERROR {name}: {type(error).__name__}: {error}")
        if VERBOSE:
            traceback.print_exc()
    else:
        PASSED.append(name)
        print(f"ok    {name}")


class Clock:
    def __init__(self, start: float | None = None) -> None:
        self.now = float(start if start is not None else time.time())

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def make_runtime(root: Path, *, clock: Clock, settings: dict | None = None,
                 timeline=None, appraiser=None, memory=None, character="Sakura",
                 persona: str = "她冷静克制，不喜欢撒娇，说话公事公办。") -> EmotionRuntime:
    store = StateStore(root / "state.json", logger=FakeLogger())
    values = dict(settings or {})
    return EmotionRuntime(
        settings_provider=lambda: settings_module.parse(values),
        store=store,
        logger=FakeLogger(),
        timeline=timeline,
        appraiser=appraiser,
        memory=memory,
        character_provider=lambda: character,
        persona_provider=lambda: persona,
        clock=clock,
    )


def turn_request(text: str, turn_id: str, *, source: str = "chat", character: str = "Sakura") -> dict:
    return {
        "current_input": text,
        "character_id": character,
        "character_name": character,
        "current_turn_id": turn_id,
        "source": source,
        "mode": "normal",
        "step_index": 0,
        "recent_messages": [],
        "available_tools": [],
        "current_time": "",
    }


class StubAppraiser:
    def __init__(self, verdicts: list[appraisal_model.ModelVerdict] | None = None, *, rebuild=None) -> None:
        self.verdicts = list(verdicts or [])
        self.rebuild_verdict = rebuild
        self.calls = 0
        self.rebuild_calls = 0
        self.baseline_calls = 0
        self.baseline_text = "冷静、克制、可靠"
        self.last_history = ""
        self.last_state_line = ""
        self.last_persona = ""
        self.last_elapsed = ""

    def available(self) -> bool:
        return True

    def appraise(self, *, user_text, reply_text, state_line="", summary="", elapsed="",
                 character_name="", persona=""):
        self.calls += 1
        self.last_state_line = state_line
        self.last_persona = persona
        self.last_elapsed = elapsed
        if not self.verdicts:
            return None
        return self.verdicts.pop(0)

    def baseline(self, *, persona="", character_name=""):
        self.baseline_calls += 1
        return self.baseline_text

    def rebuild(self, *, history, state_line="", summary="", character_name="", persona=""):
        self.rebuild_calls += 1
        self.last_history = history
        self.last_persona = persona
        return self.rebuild_verdict


class StubMemory:
    def __init__(self) -> None:
        self.remembered: list[str] = []
        self.recalls: list[str] = []
        self.recall_result: list[str] = []

    def recall(self, query: str, *, limit: int = 3) -> list[str]:
        self.recalls.append(query)
        return list(self.recall_result)

    def remember(self, content: str, *, importance: float = 0.6, category: str = "emotion") -> bool:
        self.remembered.append(content)
        return True

    def test(self) -> str:
        return "记忆桥可用"


def describe_pools(values: Mapping[str, float]) -> str:
    return "、".join(f"{key} {value:.2f}" for key, value in values.items() if value > 0.01)


# ---------------------------------------------------------------- 用例

def test_taxonomy() -> None:
    listed = {
        "喜悦", "爱慕", "好奇", "害羞", "悲伤", "生气", "害怕", "厌恶", "失望", "委屈",
        # 2.1 补的六根：她本来天天在演，但没有桶
        "无奈", "无语", "自信", "疑惑", "吃醋", "温柔",
        # 2.2 又补的五根
        "担忧", "尴尬", "孤独", "惊讶", "傲娇",
        # 3.4 再补的三根
        "平静", "欣慰", "期待",
        # 3.5 再补一根
        "撒娇",
        # 3.6 再补一根
        "兴奋",
    }
    assert len(taxonomy.EMOTIONS) == 26, "应该是二十六种情绪"
    assert set(taxonomy.EMOTIONS) == listed, "情绪清单必须和素材一致"
    assert all(taxonomy.spec(key).phases for key in taxonomy.EMOTIONS), "每种情绪都要有三档措辞"
    for group in (taxonomy.POSITIVE, taxonomy.NEUTRAL, taxonomy.NEGATIVE, taxonomy.MIXED):
        assert 0 < len(group) < 26, group
    assert set(taxonomy.POSITIVE) | set(taxonomy.NEUTRAL) | set(taxonomy.NEGATIVE) | set(taxonomy.MIXED) == listed
    for key in taxonomy.EMOTIONS:
        assert -1.0 <= taxonomy.spec(key).valence <= 1.0, key


def test_elapsed_measures_real_gap() -> None:
    """「上一轮至今」要量的是距上一轮多久，不是她这次生成花了几秒。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-gap-"))
    try:
        clock = Clock(12_000_000.0)
        appraiser = StubAppraiser([
            appraisal_model.ModelVerdict(deltas={"温柔": 10}),
            appraisal_model.ModelVerdict(deltas={"温柔": 2}),
        ])
        runtime = make_runtime(root, clock=clock, appraiser=appraiser)

        runtime.fragment_for(turn_request("我先去忙了", "g1"))
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "g1"})
        assert appraiser.last_elapsed == "刚刚", appraiser.last_elapsed

        clock.advance(3 * 3600)
        runtime.fragment_for(turn_request("我回来了", "g2"))
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "g2"})
        assert appraiser.last_elapsed == "3 小时前", f"应该是 3 小时前：{appraiser.last_elapsed}"
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_persona_excerpt_whole_card() -> None:
    """整张卡都给；只有超预算时才按段落等比例收窄，不许整段消失。"""

    from emotion_pool.runtime import _persona_excerpt
    card = ("【人格定位】" + "甲" * 300 + "【外貌特征】" + "乙" * 300 +
            "【性格】你不热烈撒娇。" + "丙" * 300)

    whole = _persona_excerpt(card, budget=2000)
    assert whole == " ".join(card.split()), "不超预算就原样给整张卡"
    assert "不热烈撒娇" in whole, "关键句必须在"

    narrow = _persona_excerpt(card, budget=400)
    assert len(narrow) <= 400, len(narrow)
    assert "甲" in narrow and "乙" in narrow and "丙" in narrow, "每段都要留一点，不能整段丢"

    plain = "普通文本" * 100
    assert _persona_excerpt(plain, budget=50) == plain[:50], "没有段落标题就硬切"


def test_persona_block_not_reclipped() -> None:
    """runtime 挑好的人设，不能再被判断器的 _persona_block 截断一次。"""

    long_persona = ("甲" * 600) + "你不热烈撒娇。" + ("乙" * 400)
    block = appraisal_model._persona_block(long_persona)
    body = block[-1]
    assert "不热烈撒娇" in body, "第 611 字之后的关键句不能被切掉"
    assert len(body) == len(long_persona), len(body)


def test_ghost_value_cleared() -> None:
    """低于 5% 的弦不进面板也不进提示词，就不该占着名额。"""

    state = state_module.CharacterState()
    state.apply_mood(["委屈"], {"委屈": 3})
    assert state.values["委屈"] == 0.0, "3% 低于在场线，应当清零"
    state.apply_mood(["委屈"], {"委屈": 6})
    assert abs(state.values["委屈"] - 0.06) < 1e-6, "6% 在场"


def test_episode_records_drop() -> None:
    """模型写了负数要记一条；只是不再点名（退场）就不算情绪事件。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-drop-"))
    try:
        clock = Clock(13_000_000.0)
        runtime = make_runtime(root, clock=clock)
        state = runtime._state("Sakura")
        settings = settings_module.parse({})

        state.apply_mood(["委屈"], {"委屈": 40})
        applied = state.apply_mood(["委屈"], {"委屈": -30})
        assert applied["委屈"] == -30.0, applied
        assert "委屈" not in applied.evicted, applied.evicted
        snapshot = runtime._record_episode(
            state, applied, appraisal_model.ModelVerdict(gist="他走了"),
            settings, clock(), "那我先走了", "drop-1",
        )
        assert snapshot is not None and snapshot["delta"] == -30.0, snapshot

        # 名册里不再点名：只是退场，不记涟漪
        gone = state.apply_mood([], {})
        assert "委屈" in gone.evicted, gone.evicted
        assert runtime._record_episode(
            state, gone, appraisal_model.ModelVerdict(gist=""),
            settings, clock(), "嗯", "drop-2",
        ) is None, "退场不该进涟漪"
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_eviction_is_not_an_emotion() -> None:
    """被名额挤出名单的那几根，不算情绪变化：涟漪不记，面板不报。"""

    state = state_module.CharacterState()
    state.apply_mood(["温柔"], {"温柔": 40})
    applied = state.apply_mood(["害羞", "无奈"], {"害羞": 30, "无奈": 20})
    assert "温柔" in applied.evicted, applied.evicted
    assert applied["温柔"] == -40.0, applied

    described = state_module.describe(state)
    assert [item["key"] for item in described["slots"]] == ["害羞", "无奈"]

    # 面板那一行也不该把驱逐写成「温柔 -40」
    state.last_judge = {
        "at": 1_000.0, "deltas": {}, "applied": dict(applied),
        "evicted": list(applied.evicted), "summary": "转移了", "source": "refresh",
    }
    status = prompt.status_last(state, 2_000.0)
    assert "温柔" not in status["label"], status
    assert "害羞" in status["label"], status


def test_main_streak_recheck() -> None:
    """连续当主的弦，判断器要能看到「已经几轮」，提示词里也要有复核要求。"""

    from emotion_pool.runtime import _state_lines
    state = state_module.CharacterState()
    state.apply_mood(["害羞"], {"害羞": 50})
    for _ in range(4):
        state.note_main("害羞")
    assert state.streak_of("害羞") == 4
    line, _summary = _state_lines(state)
    assert "已连续 4 轮为主" in line, line

    state.note_main("无奈")
    assert state.streak_of("害羞") == 0 and state.streak_of("无奈") == 1
    assert "连续三轮以上当主" in appraisal_model.SYSTEM, "提示词里要有复核规则"


def test_state_line_shows_trend() -> None:
    """判断器要能看出这根弦在涨还是在落。"""

    from emotion_pool.runtime import _state_lines
    state = state_module.CharacterState()
    state.apply_mood(["温柔"], {"温柔": 40})
    state.last_judge = {"before": {"温柔": 0.28}, "deltas": {}, "source": "turn"}
    line, _summary = _state_lines(state)
    assert "温柔 40%（上一轮 28%）" in line, line

    state.last_judge = None
    line, _summary = _state_lines(state)
    assert "上一轮" not in line, "没有上一轮就不该硬编"


def test_baseline_extracted_once() -> None:
    """性格底色只提炼一次，存进状态，面板读它。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-base-"))
    try:
        clock = Clock(14_000_000.0)
        appraiser = StubAppraiser([
            appraisal_model.ModelVerdict(deltas={"温柔": 10}),
            appraisal_model.ModelVerdict(deltas={"温柔": 2}),
        ])
        runtime = make_runtime(root, clock=clock, appraiser=appraiser)
        for turn in ("b1", "b2"):
            runtime.fragment_for(turn_request("你好", turn))
            runtime.on_chat_completed({"characterId": "Sakura", "turnId": turn})
        state = runtime._state("Sakura")
        assert state.baseline == "冷静、克制、可靠", state.baseline
        assert appraiser.baseline_calls == 1, f"只该提炼一次：{appraiser.baseline_calls}"
        payload = runtime.dashboard_payload()
        assert payload["baseline"] == "冷静、克制、可靠", payload
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_peek_does_not_judge() -> None:
    """面板轮询用的是 peek：只读，不碰模型。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-peek-"))
    try:
        import plugin as plugin_module
        context = FakeContext(root)
        instance = plugin_module.EmotionPoolPlugin()
        instance.setup(context)
        stub = StubAppraiser([appraisal_model.ModelVerdict(deltas={"温柔": 10})])
        instance.runtime._appraiser = stub
        before = stub.calls
        result = instance._action_dashboard_peek({})
        assert stub.calls == before, "peek 不该触发判断"
        assert set(result) == {"values"} and "payload" in result["values"], result
        instance.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_judge_prompt_covers_taxonomy() -> None:
    """加了新情绪，判定提示词里必须也提到它——不然模型根本不会输出这个键。"""

    for name, text in (("SYSTEM", appraisal_model.SYSTEM),
                       ("REBUILD_SYSTEM", appraisal_model.REBUILD_SYSTEM)):
        missing = [key for key in taxonomy.EMOTIONS if key not in text]
        assert not missing, f"{name} 里漏了新情绪：{missing}"
    assert "惊讶是短情绪" in appraisal_model.SYSTEM, "惊讶要提醒它短，但降多少由模型判断"
    # 宿主用的是 json_object，模型服务要求提示词里必须出现 json 字样，否则直接 400。
    assert "json" in appraisal_model.BASELINE_SYSTEM.lower(), "性格概括的提示词必须提到 json"
    assert "json" in appraisal_model.SYSTEM.lower()
    # 跨语言：她的台词多半是日语，同一个词在两种语境下是两种情绪
    assert "不要按字面直译" in appraisal_model.SYSTEM, "要有跨语言语境规则"
    assert "不要按字面直译" in appraisal_model.REBUILD_SYSTEM
    # 欣慰 / 温柔 / 撒娇 的边界
    assert "欣慰是看到他好" in appraisal_model.SYSTEM
    # 撒娇 / 傲娇 是按行为区分的两种情绪，而且互斥
    assert "撒娇 = 行为" in appraisal_model.SYSTEM
    assert "傲娇 = 行为" in appraisal_model.SYSTEM
    assert "不能同时超过 40" in appraisal_model.SYSTEM
    assert "不能同时高于 40" in appraisal_model.REBUILD_SYSTEM
    assert "不按时间衰减" in appraisal_model.SYSTEM, "总原则里要写死：不按时间衰减"


def test_residual_not_expressed() -> None:
    """余韵有值、写在注入里，但明确标着「不表现」。"""

    state = state_module.CharacterState()
    state.apply_mood(
        ["温柔", "欣慰"],
        {"温柔": 55, "欣慰": 20, "好奇": 12, "期待": 8, "尴尬": 7},
    )
    text = prompt.build(state, settings_module.parse({}), time.time())
    assert "主·温柔 55%" in text and "次·欣慰 20%" in text and "辅·好奇 12%" in text, text
    residual_line = next((line for line in text.splitlines() if line.startswith("余韵")), "")
    assert "期待 8%" in residual_line and "尴尬 7%" in residual_line, f"余韵要写进去：{text}"
    assert "不表现" in residual_line, f"并且要说清它不该表现：{residual_line}"
    assert len(text) <= prompt.MAX_CHARS["standard"], len(text)

    # 极简档不带余韵，只留一行
    brief = prompt.build(state, settings_module.parse({"detail": "brief"}), time.time())
    assert "余韵" not in brief, brief


def test_mood_roles() -> None:
    """注入的是主·次·辅三个位置，而不是一张二十一根的清单。"""

    state = state_module.CharacterState()
    state.apply_mood(
        ["委屈", "无奈", "担忧"],
        {"委屈": 46, "无奈": 22, "担忧": 12},
    )
    state.summary = "等了一小时，心里发凉却还撑着"
    text = prompt.build(state, settings_module.parse({}), time.time())
    assert "主·委屈 46%" in text and "次·无奈 22%" in text, text
    assert "辅·担忧 12%" in text, text
    assert text.count("·") == 3, f"只该有三个位置：{text}"
    assert len(text) <= prompt.MAX_CHARS["standard"], len(text)

    described = state_module.describe(state)
    assert [item["role"] for item in described["slots"]] == ["主", "次", "辅"], described


def test_mood_capacity() -> None:
    """三根表现 + 三根余韵：一共留六根，第七根挤出去；表现层只取前三。"""

    state = state_module.CharacterState()
    state.apply_mood(
        ["委屈", "失望", "生气", "担忧", "无奈", "好奇", "喜悦"],
        {"委屈": 40, "失望": 30, "生气": 20, "担忧": 10, "无奈": 8, "好奇": 6, "喜悦": 4},
    )
    alive = {key for key, value in state.values.items() if value > 0}
    assert alive == {"委屈", "失望", "生气", "担忧", "无奈", "好奇"}, alive
    assert abs(state.values["喜悦"]) < 1e-6, "第七根必须被挤出去"

    described = state_module.describe(state)
    assert [item["key"] for item in described["slots"]] == ["委屈", "失望", "生气"]
    assert [item["key"] for item in described["residual"]] == ["担忧", "无奈", "好奇"]

    # 明确说平静：全部归零
    applied = state.apply_mood([], {})
    assert all(value == 0 for value in state.values.values()), state.values
    assert applied, "归零也要报出变化量"

    # 没给名册（None）时不清场，但只留最强的三根
    state.apply_mood(["委屈", "失望"], {"委屈": 30, "失望": 20})
    state.apply_mood(None, {"生气": 10})
    alive = {key for key, value in state.values.items() if value > 0}
    assert alive == {"委屈", "失望", "生气"}, alive
    assert state.values["委屈"] > state.values["失望"] > state.values["生气"], state.values


def test_apply_deltas() -> None:
    state = state_module.CharacterState()
    applied = state.apply_deltas({"委屈": 24, "失望": 10})
    assert applied == {"委屈": 24.0, "失望": 10.0}, applied
    assert abs(state.values["委屈"] - 0.24) < 1e-6
    assert state.values["喜悦"] == 0.0, "没提到的情绪一位不能动"

    state.apply_deltas({"委屈": -100})
    assert state.values["委屈"] == 0.0, "负数不能穿过 0"

    state.apply_deltas({"生气": 500})
    assert state.values["生气"] == 1.0, "正数不能超过 100%"

    applied = state.apply_deltas({"生气": 999})
    assert applied.get("生气", 0) == 0.0, "已经满格的再加是 0，不是异常"


def test_calm_threshold() -> None:
    state = state_module.CharacterState()
    assert state.calm(), "全零应该算平静"
    state.apply_deltas({"喜悦": 3})
    assert state.calm(), "3% 低于在场线"
    state.apply_deltas({"喜悦": 10})
    assert not state.calm(), "10% 应该在场"


def test_prompt_levels() -> None:
    state = state_module.CharacterState()
    state.apply_deltas({"失望": 43, "委屈": 12})
    state.summary = "被放了鸽子，堵着"
    for detail in ("brief", "standard", "detailed"):
        settings = settings_module.parse({"detail": detail})
        text = prompt.build(state, settings, time.time())
        assert text.startswith("【情绪】"), f"{detail} 应该以标记开头：{text[:20]}"
        assert "失望 43%" in text, f"{detail} 必须给百分比：{text}"
        assert "被放了鸽子" in text, f"{detail} 必须带上心境"
        assert len(text) <= prompt.MAX_CHARS[detail], f"{detail} 超出长度上限：{len(text)}"
        if VERBOSE:
            print(f"      [{detail} {len(text)} 字]")
            for line in text.splitlines():
                print("        " + line)


def test_prompt_calm() -> None:
    state = state_module.CharacterState()
    state.summary = "平稳"
    text = prompt.build(state, settings_module.parse({}), time.time())
    assert "没有哪根弦" in text, text
    status = prompt.status_now(state)
    assert status["state"] == "neutral", status


def test_status_last_shows_refresh_change() -> None:
    """刷新给的是绝对值，没有 deltas——面板不能因为 deltas 空就说"没有变化"。"""

    state = state_module.CharacterState()
    state.apply_deltas({"温柔": 87})
    state.last_judge = {
        "at": 1_000.0, "values": {"温柔": 87}, "deltas": {},
        "applied": {"温柔": 20.0}, "summary": "心里软下来",
        "gist": "他道了谢", "mode": "set", "source": "refresh",
    }
    status = prompt.status_last(state, 2_000.0)
    assert status["label"] == "温柔 +20", status
    assert "没有变化" not in status["label"], status

    # 按轮判断照旧显示增减值
    state.last_judge = {
        "at": 1_000.0, "deltas": {"傲娇": 15.0, "害羞": 8.0}, "applied": {"傲娇": 15.0, "害羞": 8.0},
        "summary": "嘴硬", "gist": "", "source": "turn",
    }
    status = prompt.status_last(state, 2_000.0)
    assert status["label"] == "傲娇 +15、害羞 +8", status


def test_log_pairs_text() -> None:
    """中文键的字典会被宿主的日志净化器丢掉，所以日志里必须打文本。"""

    from emotion_pool.runtime import _pairs_text
    assert _pairs_text({"委屈": 24.0, "失望": -10.0}) == "委屈+24,失望-10"
    assert "=" not in _pairs_text({"委屈": 24.0}), "值里不能出现等号，否则日志查看器会把它当成新键"
    assert _pairs_text({"委屈": 0.0}) == ""
    assert _pairs_text(None) == ""


def test_store_roundtrip() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-store-"))
    try:
        clock = Clock(5_000_000.0)
        runtime = make_runtime(root, clock=clock)
        state = runtime._state("Sakura")
        state.apply_deltas({"委屈": 30})
        state.summary = "堵着"
        runtime._save()
        runtime.close()

        reloaded = make_runtime(root, clock=clock)
        after = reloaded._state("Sakura")
        assert abs(after.values["委屈"] - 0.30) < 1e-6, after.values
        assert after.summary == "堵着", after.summary
        payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
        assert payload["version"] == 3, payload["version"]
        reloaded.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v1_migration() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-migrate-"))
    try:
        legacy = {
            "version": 1,
            "characters": {
                "Sakura": {
                    "pools": {"委屈": 0.5, "喜悦": 0.1},
                    "mood": -0.3,
                    "episodes": [{"at": 1_700_000_000.0, "emotion": "委屈", "intensity": 0.5,
                                  "source": "fast", "gist": "", "trigger": "旧记录", "turn_id": "t"}],
                    "pending": {},
                }
            },
        }
        (root / "state.json").write_text(json.dumps(legacy), encoding="utf-8")
        state = StateStore(root / "state.json", logger=FakeLogger()).load()["Sakura"]
        assert abs(state.values["委屈"] - 0.5) < 1e-6, state.values
        assert state.episodes and abs(state.episodes[0].intensity - 0.5) < 1e-6, state.episodes
        assert state.summary == "", "v1 没有心境，迁移后应为空"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_turn_dedupe() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-dedupe-"))
    try:
        clock = Clock(6_000_000.0)
        appraiser = StubAppraiser([appraisal_model.ModelVerdict(deltas={"委屈": 20})])
        runtime = make_runtime(root, clock=clock, appraiser=appraiser)
        request = turn_request("你说好陪我的", "same-turn")
        runtime.fragment_for(request)
        runtime.fragment_for(request)
        state = runtime._state("Sakura")
        assert list(state.pending) == ["same-turn"], state.pending
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "same-turn"})
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "same-turn"})
        assert appraiser.calls == 1, f"同一轮只能判断一次，实际 {appraiser.calls}"
        assert abs(state.values["委屈"] - 0.20) < 1e-6, state.values
        assert not state.pending, "判断完要把待判断的轮次清掉"
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_judge_applies_and_records() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-judge-"))
    try:
        clock = Clock(7_000_000.0)
        timeline = FakeTimeline()
        timeline.entries = [
            {"turnId": "j1", "kind": "human", "payload": {"text": "说好今天陪我的"}},
            {"turnId": "j1", "kind": "assistant", "payload": {"segments": [{"text": "……那你先忙吧。"}]}},
        ]
        appraiser = StubAppraiser([
            appraisal_model.ModelVerdict(deltas={"委屈": 24, "失望": 10},
                                         summary="被放了鸽子，堵着", gist="他说好陪她又走"),
        ])
        memory = StubMemory()
        runtime = make_runtime(root, clock=clock, timeline=timeline, appraiser=appraiser, memory=memory,
                               settings={"memory_mode": "write"})
        runtime.fragment_for(turn_request("说好今天陪我的，结果又要出门", "j1"))
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "j1"})
        state = runtime._state("Sakura")
        assert abs(state.values["委屈"] - 0.24) < 1e-6, state.values
        assert abs(state.values["失望"] - 0.10) < 1e-6, state.values
        assert state.summary == "被放了鸽子，堵着", state.summary
        assert len(state.episodes) == 1, state.episodes
        assert state.episodes[0].emotion == "委屈" and state.episodes[0].delta == 24, state.episodes[0]
        assert state.last_judge and state.last_judge["source"] == "turn", state.last_judge
        assert appraiser.last_state_line == "", "第一次判断前她还没有情绪"
        assert "冷静克制" in appraiser.last_persona, f"判断必须带上角色设定：{appraiser.last_persona!r}"
        rows = runtime.episode_rows(5)
        assert rows and rows[0]["values"]["level"] == "+24", rows
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_model_freezes() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-nomodel-"))
    try:
        clock = Clock(8_000_000.0)
        runtime = make_runtime(root, clock=clock, appraiser=None)
        runtime.fragment_for(turn_request("你真烦，滚", "n1"))
        runtime.on_chat_completed({"characterId": "Sakura", "turnId": "n1"})
        state = runtime._state("Sakura")
        assert all(value == 0.0 for value in state.values.values()), state.values
        assert state.last_judge is None
        assert state.turn_count == 1, "轮次还是要计数，只是不动情绪"
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_judge_timeout_keeps_state() -> None:
    """先判后答要有硬上限：模型慢过 JUDGE_TIMEOUT 就不等，这一轮按当前情绪回答。"""

    from emotion_pool import runtime as runtime_module

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-slow-"))
    try:
        clock = Clock(17_000_000.0)

        class SlowAppraiser(StubAppraiser):
            def appraise(self, **kwargs):  # noqa: ANN003
                time.sleep(0.8)
                return appraisal_model.ModelVerdict(deltas={"委屈": 40})

        appraiser = SlowAppraiser()
        runtime = make_runtime(root, clock=clock, appraiser=appraiser)
        saved = runtime_module.JUDGE_TIMEOUT
        runtime_module.JUDGE_TIMEOUT = 0.15
        try:
            started = time.monotonic()
            fragments = runtime.fragment_for(turn_request("你说好今天陪我的", "slow1"))
            consumed = time.monotonic() - started
        finally:
            runtime_module.JUDGE_TIMEOUT = saved
        assert consumed < 0.6, f"超时了就该立刻放行，实际等了 {consumed:.2f}s"
        state = runtime._state("Sakura")
        assert all(value == 0.0 for value in state.values.values()), state.values
        assert state.last_judge is None, state.last_judge
        assert fragments and fragments[0]["id"] == "emotion_state", "超时也要有注入（用当前状态）"
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_refresh_uses_model() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-refresh-"))
    try:
        clock = Clock(9_000_000.0)
        timeline = FakeTimeline()
        timeline.entries = [
            {"turnId": "h1", "kind": "human", "payload": {"text": "我在忙，晚点聊"},
             "createdAt": "2026-11-20T00:10:00+08:00"},
            {"turnId": "h1", "kind": "assistant", "payload": {"segments": [{"text": "……好。"}]},
             "createdAt": "2026-11-20T00:10:03+08:00"},
        ]
        appraiser = StubAppraiser(rebuild=appraisal_model.ModelVerdict(
            values={"失望": 30}, summary="心里有点空", gist="他一直在忙"))
        runtime = make_runtime(root, clock=clock, timeline=timeline, appraiser=appraiser)
        result = runtime.refresh_with_model()
        assert result["ok"] is True, result
        assert "失望" in result["message"], result
        state = runtime._state("Sakura")
        assert abs(state.values["失望"] - 0.30) < 1e-6, state.values
        assert state.last_judge["source"] == "refresh" and state.last_judge["mode"] == "set"
        assert "[00:10] 用户：我在忙" in appraiser.last_history, appraiser.last_history

        # 关键：同一个判断点两遍，数值必须收敛而不是叠加
        for _ in range(3):
            runtime.refresh_with_model()
        assert abs(state.values["失望"] - 0.30) < 1e-6, f"刷新必须是幂等的：{state.values}"

        # 没提到的情绪保持原值
        appraiser.rebuild_verdict = appraisal_model.ModelVerdict(values={"喜悦": 10})
        runtime.refresh_with_model()
        assert abs(state.values["失望"] - 0.30) < 1e-6, "没提到的情绪必须保持原值"
        assert abs(state.values["喜悦"] - 0.10) < 1e-6

        # 模型不按契约给 values、只给 deltas 时才退回增量
        appraiser.rebuild_verdict = appraisal_model.ModelVerdict(deltas={"委屈": 15})
        runtime.refresh_with_model()
        assert abs(state.values["委屈"] - 0.15) < 1e-6, state.values
        assert state.last_judge["mode"] == "add"

        runtime._appraiser = None
        blocked = runtime.refresh_with_model()
        assert blocked["ok"] is False and "没有可用的聊天模型" in blocked["message"], blocked
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_model_parse() -> None:
    good = appraisal_model.parse_verdict(
        '{"deltas": {"委屈": 24, "失望": -10}, "summary": "堵着", "gist": "他说好陪她"}'
    )
    assert good is not None and good.deltas["委屈"] == 24 and good.deltas["失望"] == -10
    assert good.summary == "堵着" and good.gist == "他说好陪她"

    clipped = appraisal_model.parse_verdict('{"deltas": {"生气": 300}}')
    assert clipped is not None and clipped.deltas["生气"] == appraisal_model.MAX_DELTA, "要夹到上限"

    fraction = appraisal_model.parse_verdict('{"deltas": {"委屈": 0.3}}')
    assert fraction is not None and fraction.deltas["委屈"] == 30, "0.3 应当被理解为 30 个点"

    tiny = appraisal_model.parse_verdict('{"deltas": {"委屈": 0.004}, "summary": "轻微"}')
    assert tiny is not None and not tiny.deltas, "小于 1 个点的抖动丢掉"
    assert tiny.summary == "轻微"

    mixed = appraisal_model.parse_verdict('{"deltas": {"喜悦": 8, "委屈": 0.3}}')
    assert mixed is not None and mixed.deltas["喜悦"] == 8 and "委屈" not in mixed.deltas, mixed

    english = appraisal_model.parse_verdict('{"deltas": {"joy": 20, "grievance": 10}, "summary": "x"}')
    assert english is not None and english.deltas["喜悦"] == 20 and english.deltas["委屈"] == 10

    added = appraisal_model.parse_verdict(
        '{"deltas": {"无奈": 18, "无语": 12, "jealousy": 10}, "summary": "拿你没办法"}')
    assert added is not None, added
    assert added.deltas == {"无奈": 18, "无语": 12, "吃醋": 10}, added.deltas

    rescued = appraisal_model.recover_verdict('{"deltas": {"无奈": 20, "吃醋": 15, "summary": "没眼看')
    assert rescued is not None and rescued.deltas.get("无奈") == 20 and rescued.deltas.get("吃醋") == 15, rescued

    truncated = appraisal_model.recover_verdict('{"deltas": {"失望": 25, "委屈": 10, "summary": "他放她鸽子')
    assert truncated is not None and truncated.deltas["失望"] == 25, truncated

    rebuild = appraisal_model.parse_verdict('{"values": {"委屈": 55, "失望": 20}, "summary": "堵着"}')
    assert rebuild is not None and rebuild.values["委屈"] == 55 and rebuild.values["失望"] == 20
    assert not rebuild.deltas, "values 不该被当成增量"

    zero = appraisal_model.parse_verdict('{"values": {"委屈": 0}, "summary": ""}')
    assert zero is not None and zero.values["委屈"] == 0, "0 是有效值（表示回到平静）"

    assert appraisal_model.parse_verdict("我不输出 JSON") is None


def test_settings_parse() -> None:
    parsed = settings_module.parse({"memory_mode": "nope", "detail": "xxx", "episode_keep": 999})
    assert parsed.memory_mode == "off" and parsed.detail == "standard"
    assert parsed.episode_keep == settings_module.MAX_EPISODES
    assert settings_module.parse({}).values() == settings_module.Settings().values()
    field_keys = {field["key"] for field in settings_module.descriptor()["fields"]}
    for removed in ("reactivity", "decay_speed", "inertia_seconds", "switch_percent",
                    "model_refine", "debug_emotion"):
        assert removed not in field_keys, f"{removed} 应该已经删掉"


def _status_ok(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"state", "label", "message"}:
        return False
    return (value.get("state") in {"neutral", "ready", "working", "warning", "error"}
            and isinstance(value.get("label"), str) and 1 <= len(value["label"]) <= 120
            and isinstance(value.get("message"), str) and len(value["message"]) <= 240)


def test_descriptors() -> None:
    descriptor = settings_module.descriptor()
    keys = [field["key"] for field in descriptor["fields"]]
    assert len(keys) == len(set(keys)), "字段 key 不能重复"
    for field in descriptor["fields"]:
        assert field["label"] and isinstance(field.get("description"), str), field
        kind = field["type"]
        default = field.get("default")
        if kind == "boolean":
            assert isinstance(default, bool), field
        elif kind == "integer":
            assert isinstance(default, int) and not isinstance(default, bool), field
        elif kind == "select":
            assert default in {option["value"] for option in field["options"]}, field
        elif kind == "status":
            assert _status_ok(default), field
        elif kind == "readonly":
            assert isinstance(default, str) and isinstance(field.get("maxLength"), int), field
        cond = field.get("enabledWhen")
        if cond:
            assert cond["field"] in keys and isinstance(cond["equals"], str), field
    for action in descriptor["actions"]:
        assert action["danger"] is False and action["label"], action
    ids = [action["actionId"] for action in descriptor["actions"]]
    for gone in ("preview", "open_preview", "open_state", "reset", "export_log"):
        assert gone not in ids, f"{gone} 不该留在设置页"
    assert ids == ["test_memory"], ids
    fields = {field["key"] for field in descriptor["fields"]}
    assert "status_now" not in fields and "status_last" not in fields, "两个 status 字段要去掉"
    assert [action["actionId"] for action in descriptor["actions"]].count("surge") == 0

    source = (HERE / "frontend" / "dashboard.js").read_text(encoding="utf-8")
    dash = settings_module.dashboard_descriptor(source)
    assert dash["presentation"]["component"] == "module"
    assert "export function mount" in dash["presentation"]["source"]
    assert [field["key"] for field in dash["fields"]] == ["payload"]

    collection = settings_module.collection_descriptor()
    for column in collection["columns"]:
        assert column["type"] in {"string", "number", "boolean", "datetime"}, column


def test_plugin_setup() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-plugin-"))
    try:
        import plugin as plugin_module
        context = FakeContext(root)
        instance = plugin_module.EmotionPoolPlugin()
        instance.setup(context)
        assert "context" in context.holders, "上下文提供者要注册"
        assert "settings" in context.holders, "设置页要注册"
        assert "collection" in context.holders, "记录表要注册"
        assert "sakura.host.chat.completed" in context.events, "要订阅对话完成事件"
        assert context.effects, "要登记关闭清理"
        pages = context.holders["settings"].get("pages", [])
        assert pages, "要在左侧栏给自己开一页"
        assert pages[0]["group"] == "behavior" and pages[0]["title"] == "情绪", pages
        placements = context.holders["settings"].get("placements", [])
        assert placements and placements[0]["pageId"] == "sakuratools.emotion_pool:emotion", placements
        assert placements[0]["region"] == "content", placements

        fragments = instance._fragments(turn_request("今天天气不错", "p1"))
        assert len(fragments) == 1 and fragments[0]["content"].startswith("【情绪】"), fragments
        assert fragments[0]["required"] is False and fragments[0]["sensitivity"] == "private"
        assert 0 <= fragments[0]["priority"] <= 100 and 1 <= fragments[0]["budgetHint"] <= 4096

        values = instance._load_settings()
        assert "status_now" not in values and "status_last" not in values, values
        assert isinstance(values.get("saved"), str), values
        assert instance._save_settings({"detail": "detailed"})["applicationState"] == "applied"
        assert context.config.get()["detail"] == "detailed"
        assert not hasattr(instance, "_action_preview"), "预览动作应该已经删掉"
        assert not hasattr(instance, "_action_surge"), "不该有任何手动改情绪的入口"
        # 确认在前端对话框里，插件这层直接执行
        result = instance._action_reset({})
        assert "归零" in result["message"], result
        rows = instance._query_episodes({"limit": 5})
        assert set(rows) == {"items", "nextCursor", "total"}
        instance.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_export_log_needs_confirmation() -> None:
    """导出先警告一次隐私，再点一次才写文件；文件里要有判定与聊天原文。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-export-"))
    try:
        import plugin as plugin_module
        context = FakeContext(root)
        instance = plugin_module.EmotionPoolPlugin()
        instance.setup(context)
        runtime = instance.runtime
        runtime._clock = Clock(16_000_000.0)
        runtime._appraiser = StubAppraiser([
            appraisal_model.ModelVerdict(deltas={"温柔": 12}, summary="有点暖"),
        ])
        context.timeline.entries = [
            {"turnId": "x1", "kind": "human", "payload": {"text": "我回来了"}},
            {"turnId": "x1", "kind": "assistant", "payload": {"segments": [{"text": "おかえり。"}]}},
        ]
        instance._fragments(turn_request("我回来了", "x1"))
        instance._on_chat_completed({"characterId": "Sakura", "turnId": "x1"})

        second = instance._action_export_log({})
        assert "已导出" in second["message"], second
        files = list(Path(root).glob("emotion-pool-log-*.txt"))
        assert files, "第二次应该写出文件"
        text = files[0].read_text(encoding="utf-8")
        assert "逐轮判定记录" in text and "我回来了" in text, text[:200]
        assert "有点暖" in text, "要包含当时的心境"
        instance.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_plugin_end_to_end() -> None:
    """走一遍真实插件入口：一轮对话 → 判断 → 面板 → 记录表。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-e2e-"))
    try:
        import plugin as plugin_module
        context = FakeContext(root)
        instance = plugin_module.EmotionPoolPlugin()
        instance.setup(context)
        runtime = instance.runtime
        assert runtime is not None
        clock = Clock(11_000_000.0)
        runtime._clock = clock
        runtime._appraiser = StubAppraiser([
            appraisal_model.ModelVerdict(deltas={"委屈": 32, "失望": 18},
                                         summary="被放了鸽子，堵着", gist="他又要走"),
        ])
        context.timeline.entries = [
            {"turnId": "e2e", "kind": "human", "payload": {"text": "说好今天陪我的"}},
            {"turnId": "e2e", "kind": "assistant",
             "payload": {"segments": [{"text": "……那你先忙吧。"}]}},
        ]
        instance._fragments(turn_request("说好今天陪我的，结果又要出门", "e2e"))
        state = runtime._state("Sakura")
        # 先判后答：拿到注入的时候情绪已经定了，不再是「先回完再判」
        assert abs(state.values["委屈"] - 0.32) < 1e-6, state.values
        assert state.summary == "被放了鸽子，堵着", state.summary
        instance._on_chat_completed({"characterId": "Sakura", "turnId": "e2e"})
        assert not state.pending, "她开口之后要清掉待判断标记"
        payload = instance._load_dashboard()["payload"]
        assert set(payload) == {"slots", "others", "baseline", "updated", "turns"}, payload
        assert len(payload["others"]) == len(taxonomy.EMOTIONS) - len(payload["slots"]), payload
        head = payload["slots"][0]
        assert head["role"] == "主" and head["key"] == "委屈" and head["value"] == 32, head
        assert payload["slots"][1]["key"] == "失望", payload
        assert "上一次判断" in payload["updated"], payload
        rows = instance._query_episodes({"limit": 10})
        assert rows["items"], "记录表应该有内容"
        first = rows["items"][0]["values"]
        assert set(first) == {"at", "emotion", "level", "source", "gist"}, first
        assert first["level"] == "+32", first
        instance.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dashboard_module() -> None:
    """仪表盘：模块源码、payload 形状、刷新走模型。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-dash-"))
    try:
        import plugin as plugin_module
        context = FakeContext(root)
        instance = plugin_module.EmotionPoolPlugin()
        instance.setup(context)
        runtime = instance.runtime
        assert runtime is not None
        descriptor = context.holders["settings"]["section"].args[0]
        assert descriptor["presentation"]["component"] == "module"
        assert "epd-track" in descriptor["presentation"]["source"], "胶囊轨道要真的画出来"
        assert "epd-dialog" in descriptor["presentation"]["source"], "确认对话框要在仪表盘里"
        dash_ids = {a["actionId"] for a in descriptor["actions"]}
        assert {"refresh", "peek", "reset", "export_log"} <= dash_ids, dash_ids
        assert "confirmDialog" in descriptor["presentation"]["source"]
        assert "adoptedStyleSheets" in descriptor["presentation"]["source"], "CSP 下必须用构造式样式表"

        payload = instance._load_dashboard()["payload"]
        # 只有三个位置：心境、headline、强弱标签一律不给——让她在对话里被感受，不在面板上被描述
        assert set(payload) == {"slots", "others", "baseline", "updated", "turns"}, payload
        assert payload["slots"] == [], "平静时三个位置都是空的"
        assert len(payload["others"]) == 26, "平静时全部二十六种都收在抽屉里"
        assert all(item["value"] == 0 for item in payload["others"]), payload["others"]
        runtime._state("Sakura").apply_mood(["失望", "委屈"], {"失望": 43, "委屈": 12})
        payload = instance._load_dashboard()["payload"]
        assert [item["role"] for item in payload["slots"]] == ["主", "次"], payload
        for item in payload["slots"]:
            assert set(item) == {"role", "key", "value"}, f"每个位置只有角色、名字、数值：{item}"
            assert item["key"] in taxonomy.EMOTIONS
            assert isinstance(item["value"], int) and 0 <= item["value"] <= 100, item



        context.timeline.entries = [
            {"turnId": "h1", "kind": "human", "payload": {"text": "我在忙"}, "createdAt": "2026-11-20T00:10:00+08:00"},
            {"turnId": "h1", "kind": "assistant", "payload": {"segments": [{"text": "……好。"}]},
             "createdAt": "2026-11-20T00:10:03+08:00"},
        ]
        stub = StubAppraiser(rebuild=appraisal_model.ModelVerdict(values={"委屈": 20}, summary="有点闷"))
        runtime._appraiser = stub
        refreshed = instance._action_dashboard_refresh({})
        assert "委屈" in refreshed["message"], refreshed
        assert stub.rebuild_calls == 1, "刷新必须真的去问模型"
        slots = {item["key"]: item["value"] for item in refreshed["values"]["payload"]["slots"]}
        assert slots.get("委屈") == 20, slots
        instance.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_memory_bridge() -> None:
    root = Path(tempfile.mkdtemp(prefix="emotion-pool-memory-"))
    try:
        catalog = [
            {"name": "memory_search", "registrationId": "reg_search", "source": "plugin"},
            {"name": "memory_remember", "registrationId": "reg_remember", "source": "plugin"},
        ]

        def handler(name: str, arguments: Mapping[str, Any]):
            if name == "memory_search":
                return {"status": "ready", "memories": [{"content": "他养了一只叫团子的猫"}]}
            return {"status": "ready"}

        context = FakeContext(root, catalog=catalog, tools_handler=handler)
        bridge = MemoryBridge(context, logger=FakeLogger())
        assert bridge.available(), bridge.status_text()
        lines = bridge.recall("你还记得团子吗")
        assert lines and "团子" in lines[0], lines
        assert bridge.remember("（情绪记忆）他说要走了", importance=0.8) is True
        names = [name for name, _ in context.holders["sakura.host.tools"]["calls"]]
        assert names == ["memory_search", "memory_remember"], names

        empty = FakeContext(root, catalog=[], tools_handler=handler)
        assert MemoryBridge(empty, logger=FakeLogger()).available() is False
        assert MemoryBridge(empty, logger=FakeLogger()).recall("你还记得团子吗") == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_long_session_stability() -> None:
    """连聊 40 轮：不漂、不卡、不重复应用。"""

    root = Path(tempfile.mkdtemp(prefix="emotion-pool-long-"))
    try:
        clock = Clock(10_000_000.0)
        verdicts = [
            appraisal_model.ModelVerdict(deltas={"喜悦": 12}, summary="心情不错"),
            appraisal_model.ModelVerdict(deltas={}, summary=""),
            appraisal_model.ModelVerdict(deltas={"失望": 25, "委屈": 10}, gist="他又忙"),
            appraisal_model.ModelVerdict(deltas={"生气": 15}),
        ]
        appraiser = StubAppraiser([verdicts[index % len(verdicts)] for index in range(40)])
        runtime = make_runtime(root, clock=clock, appraiser=appraiser)
        texts = [
            "你今天好可爱", "我在加班，晚点聊", "对不起，刚才是我不对", "为什么你会这么想",
            "随便吧", "哈哈你说得对", "我要睡了", "我回来了", "你骗我，闭嘴", "我想你",
        ]
        for index in range(40):
            started = time.perf_counter()
            runtime.fragment_for(turn_request(texts[index % len(texts)], f"L{index}"))
            runtime.on_chat_completed({"characterId": "Sakura", "turnId": f"L{index}"})
            elapsed = time.perf_counter() - started
            assert elapsed < 0.5, f"第 {index} 轮太慢：{elapsed * 1000:.0f}ms"
            clock.advance(120)
        state = runtime._state("Sakura")
        assert all(0.0 <= value <= 1.0 for value in state.values.values()), state.values
        assert 0 < len(state.episodes) <= settings_module.MAX_EPISODES, len(state.episodes)
        assert not state.pending, "判断完不该堆积待判断的轮次"
        assert state.turn_count == 40, state.turn_count
        runtime.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    print("情绪池自检（v3 · 主·次·辅 · 轮驱动）")
    print("=" * 60)
    cases = [
        ("情绪清单：二十六种", test_taxonomy),
        ("判定提示词覆盖每一种", test_judge_prompt_covers_taxonomy),
        ("提示词：主·次·辅", test_mood_roles),
        ("判断：上一轮至今量对了", test_elapsed_measures_real_gap),
        ("判断：能看到涨落趋势", test_state_line_shows_trend),
        ("面板：性格底色只提炼一次", test_baseline_extracted_once),
        ("面板：peek 只读不判断", test_peek_does_not_judge),
        ("人设：整张卡给判断器", test_persona_excerpt_whole_card),
        ("人设：判断器不再二次截断", test_persona_block_not_reclipped),
        ("池子：幽灵值清零", test_ghost_value_cleared),
        ("涟漪：退场不算情绪事件", test_episode_records_drop),
        ("池子：驱逐不算情绪变化", test_eviction_is_not_an_emotion),
        ("判断：主情绪久了要复核", test_main_streak_recheck),
        ("池子：表现三根 + 余韵三根", test_mood_capacity),
        ("注入：余韵标着不表现", test_residual_not_expressed),
        ("状态行：刷新也要说清改了什么", test_status_last_shows_refresh_change),
        ("日志：中文键打成文本", test_log_pairs_text),
        ("增减：稀疏与夹紧", test_apply_deltas),
        ("在场线：5%", test_calm_threshold),
        ("提示词：三档", test_prompt_levels),
        ("提示词：平静", test_prompt_calm),
        ("落盘：存取一致", test_store_roundtrip),
        ("迁移：v1 结构", test_v1_migration),
        ("轮次：同一轮只判断一次", test_turn_dedupe),
        ("判断：应用并记录", test_judge_applies_and_records),
        ("判断：没有模型就冻住", test_no_model_freezes),
        ("判断：先判超时不卡对话", test_judge_timeout_keeps_state),
        ("刷新：补轮次、不清零", test_refresh_uses_model),
        ("解析：百分比与截断", test_model_parse),
        ("设置：坏值归一、旧参数已删", test_settings_parse),
        ("描述符：宿主要求的形状", test_descriptors),
        ("插件：完整 setup", test_plugin_setup),
        ("插件：导出日志要二次确认", test_export_log_needs_confirmation),
        ("插件：端到端一轮", test_plugin_end_to_end),
        ("仪表盘：胶囊模块与刷新", test_dashboard_module),
        ("记忆桥：工具目录", test_memory_bridge),
        ("长会话：40 轮稳定", test_long_session_stability),
    ]
    for name, fn in cases:
        check(name, fn)
    print("=" * 60)
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败：")
        for name, error in FAILED:
            print(f"  - {name}: {error.splitlines()[0] if error else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())