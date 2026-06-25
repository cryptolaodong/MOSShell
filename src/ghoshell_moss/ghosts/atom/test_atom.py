"""Atom Ghost 原型测试.

只测真实数据路径和算法正确性，不 mock pydantic AI 内部行为.
"""

import os
from pathlib import Path

import pytest
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.mindflow import Moment, Reaction
from ghoshell_moss.core.blueprint.ghost import GhostWorkspace
from ghoshell_moss.message import Message
from ghoshell_moss.contracts.system_prompter import SystemPrompter, BaseSystemPrompter
from ghoshell_moss.contracts.workspace import Workspace, LocalWorkspace
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger


# ── helpers ─────────────────────────────────────────


def _atom_meta(**kwargs):
    from ._meta import AtomMeta
    defaults = dict(name="test_atom", soul_content="you are a helpful assistant.")
    return AtomMeta(**{**defaults, **kwargs})


def _container(*bindings):
    c = Container()
    for contract, instance in bindings:
        c.set(contract, instance)
    return c


# ── soul 加载 ───────────────────────────────────────


class TestSoul:
    def test_soul_content_direct(self):
        meta = _atom_meta(soul_content="direct")
        assert meta.soul_content == "direct"

    def test_soul_content_none_is_empty(self):
        meta = _atom_meta(soul_content=None)
        assert meta.soul_content == ""

    def test_soul_path_resolves_to_name(self, tmp_path: Path):
        souls_dir = tmp_path
        (souls_dir / "soul.md").write_text("from file")
        ws = GhostWorkspace(home=tmp_path, source=None)
        meta = _atom_meta(soul_path=None, soul_content=None)
        meta._load_soul(ws)
        assert meta.soul_content == "from file"

    def test_soul_path_explicit_str(self, tmp_path: Path):
        souls_dir = tmp_path
        (souls_dir / "custom.md").write_text("custom")
        meta = _atom_meta(soul_path="custom.md", soul_content=None)
        ws = GhostWorkspace(home=tmp_path, source=None)
        meta._load_soul(ws)
        assert meta.soul_content == "custom"

    def test_soul_path_absolute(self, tmp_path: Path):
        file = tmp_path / "soul.md"
        file.write_text("absolute")
        meta = _atom_meta(soul_path=file, soul_content=None)
        meta._load_soul(GhostWorkspace(Path("/irrelevant"), source=None))
        assert meta.soul_content == ""

    def test_soul_content_skips_file_load(self, tmp_path: Path):
        souls_dir = tmp_path
        (souls_dir / "soul.md").write_text("should not load")
        ws = GhostWorkspace(home=tmp_path, source=None)
        meta = _atom_meta(soul_content="preset")
        meta._load_soul(ws)
        assert meta.soul_content == "preset"


# ── build_instruction ───────────────────────────────


class TestBuildInstruction:
    def test_no_system_prompter(self):
        meta = _atom_meta(soul_content="my soul")
        c = Container()
        assert meta.build_instruction_from_ioc(c) == "my soul"

    def test_with_system_prompter(self):
        meta = _atom_meta(soul_content="my soul")
        prompter = BaseSystemPrompter(own_instruction="moss instruction")
        c = _container((SystemPrompter, prompter))
        instruction = meta.build_instruction_from_ioc(c)
        assert "moss instruction" in instruction
        assert "my soul" in instruction

    def test_order_moss_before_soul(self):
        meta = _atom_meta(soul_content="soul")
        prompter = BaseSystemPrompter(own_instruction="moss")
        c = _container((SystemPrompter, prompter))
        lines = meta.build_instruction_from_ioc(c).split("\n")
        assert lines[0] == "moss"
        assert lines[1] == "soul"


# ── build_agent (真依赖) ─────────────────────────────


class TestBuildAgent:
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_MODEL"),
        reason="ANTHROPIC_MODEL env var not set",
    )
    def test_with_env_model(self):
        meta = _atom_meta(model=None, soul_content="test")
        agent = meta.build_agent(Container())
        assert agent._model is not None

    def test_loads_soul_from_workspace(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        (souls_dir / "test_atom.md").write_text("soul from ws")
        ws = GhostWorkspace(home=tmp_path, source=None)
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content=None,
            soul_path='souls/test_atom.md',
        )
        c = _container((GhostWorkspace, ws))
        meta.build_agent(c)
        assert meta.soul_content == "soul from ws"

    def test_on_agent_build_called(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        calls = []
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
            on_agent_build=lambda a: calls.append(a),
        )
        c = _container((Workspace, LocalWorkspace(tmp_path)))
        agent = meta.build_agent(c)
        assert len(calls) == 1
        assert calls[0] is agent


# ── factory ─────────────────────────────────────────


class TestFactory:
    def test_returns_atom_with_workspace(self, tmp_path: Path):
        from ._runtime import Atom
        from pydantic_ai.models.anthropic import AnthropicModel
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        atom = meta.factory(c)
        assert isinstance(atom, Atom)
        assert atom.meta is meta


# ── 消息协议 ────────────────────────────────────────


class TestAtomMessages:
    def _atom(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        return meta.factory(c)

    def test_to_model_request(self, tmp_path: Path):
        from pydantic_ai.messages import ModelRequest
        atom = self._atom(tmp_path)
        msg = Message.new().with_content("hello")
        moment = Moment(percepts=[msg])
        request = atom.to_model_request(moment)
        assert isinstance(request, ModelRequest)

    def test_history_initially_empty(self, tmp_path: Path):
        atom = self._atom(tmp_path)
        assert atom.model_history() == []

    def test_save_adds_to_history(self, tmp_path: Path, monkeypatch):
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
        monkeypatch.setenv("MOSS_ATOM_HISTORY_ENABLED", "1")
        atom = self._atom(tmp_path)
        moment = Moment(percepts=[Message.new().with_content("hi")])
        response = ModelResponse(parts=[TextPart(content="hello")])
        atom.save_model_request(moment, response)
        history = atom.model_history()
        assert len(history) == 2
        assert isinstance(history[0], ModelRequest)
        assert isinstance(history[1], ModelResponse)

    def test_save_skips_empty_response(self, tmp_path: Path, monkeypatch):
        from pydantic_ai.messages import ModelResponse
        monkeypatch.setenv("MOSS_ATOM_HISTORY_ENABLED", "1")
        atom = self._atom(tmp_path)
        moment = Moment(percepts=[Message.new().with_content("hi")])
        response = ModelResponse(parts=[])
        atom.save_model_request(moment, response)
        assert atom.model_history() == []


# ── system_prompt ───────────────────────────────────


class TestSystemPrompt:
    def test_returns_build_instruction(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        prompter = BaseSystemPrompter(own_instruction="moss hi")
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="i am atom",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container(
            (Workspace, ws),
            (SystemPrompter, prompter),
            (LoggerItf, get_moss_logger()),
        )
        atom = meta.factory(c)
        prompt = atom.system_prompt()
        assert "moss hi" in prompt
        assert "i am atom" in prompt


# ── 生命周期 ────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_enter_exit(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        atom = meta.factory(c)
        async with atom as ctx:
            assert ctx is atom


# ── Adapter ─────────────────────────────────────────


class TestAdapter:
    def test_messages_to_parts_text(self):
        from ._adapter import messages_to_parts
        from pydantic_ai import TextContent
        msg = Message.new().with_content("hello")
        parts = messages_to_parts([msg])
        assert len(parts) == 1
        assert isinstance(parts[0], TextContent)

    def test_messages_to_parts_multiple(self):
        from ._adapter import messages_to_parts
        msgs = [Message.new().with_content("a"), Message.new().with_content("b")]
        parts = messages_to_parts(msgs)
        assert len(parts) == 2

    def test_moment_to_request(self):
        from ._adapter import moment_to_request
        from pydantic_ai.messages import ModelRequest
        moment = Moment(percepts=[Message.new().with_content("test")])
        request = moment_to_request(moment)
        assert isinstance(request, ModelRequest)
        assert len(request.parts) > 0

    def test_moment_to_request_includes_percepts(self):
        from ._adapter import moment_to_request
        msg = Message.new().with_content("percept content")
        request = moment_to_request(Moment(percepts=[msg]))
        assert any("percept content" in str(p) for p in request.parts)

    def test_moment_to_request_includes_previous_outcomes(self):
        from ._adapter import moment_to_request
        prev = Reaction(moment_id="p1",
                        outcomes=[Message.new().with_content("outcome")])
        request = moment_to_request(Moment(previous=prev))
        assert any("outcome" in str(p) for p in request.parts)

    def test_moment_to_request_empty(self):
        from ._adapter import moment_to_request
        from pydantic_ai.messages import ModelRequest
        request = moment_to_request(Moment())
        assert isinstance(request, ModelRequest)
        assert len(request.parts) == 0

    def test_simple_fast_reply_for_greeting(self):
        from ._runtime import _request_text, _simple_fast_reply_for_text
        from pydantic_ai import TextContent

        assert _simple_fast_reply_for_text("你好。") == "在呢。"
        assert _simple_fast_reply_for_text("小白你好") == "在呢。"
        assert _simple_fast_reply_for_text("你好，帮我看看延迟") is None
        assert _request_text([TextContent(content="context"), TextContent(content="你好")]) == "你好"

    def test_simple_fast_reply_for_ack_request(self):
        from ._runtime import _simple_fast_reply_for_text

        assert (
            _simple_fast_reply_for_text(
                "小白请听我说完这是一句比较长的话我想测试你会不会等我完整说完以后再回答我你只需要回答收到"
            )
            == "收到。"
        )
        assert _simple_fast_reply_for_text("请详细解释一下为什么回答收到比较好") is None

    def test_simple_fast_reply_for_capability_request(self):
        from ._runtime import _simple_fast_reply_for_text

        assert _simple_fast_reply_for_text("小白你现在能做什么请用一句话说") == (
            "我能听你说话、回答问题，并同步表情和头部动作。"
        )
        assert _simple_fast_reply_for_text("小白用一句话回答你现在能帮我做什么") == (
            "我能听你说话、回答问题，并同步表情和头部动作。"
        )
        assert _simple_fast_reply_for_text("小白用一句话回答你现在能帮") == (
            "我能听你说话、回答问题，并同步表情和头部动作。"
        )
        assert _simple_fast_reply_for_text("小白用一句话回答你现在的话我付什么") == (
            "我能听你说话、回答问题，并同步表情和头部动作。"
        )
        assert _simple_fast_reply_for_text("小白用一句话回答你现在的方法做什么") == (
            "我能听你说话、回答问题，并同步表情和头部动作。"
        )
        assert _simple_fast_reply_for_text("你能做什么，详细展开讲讲") is None

    def test_simple_fast_reply_for_latency_probe(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind(
            "小白我想测试一下长句子的理解和延迟，请你一定等我这句话全部说完以后，再用一句话简单回答我。"
        )

        assert kind == "latency_probe"
        assert reply == "我会等你说完，再简短回答。"
        clipped_kind, clipped_reply = _simple_fast_reply_with_kind(
            "请你一定等我这句话全部说完以后，再用一句话简单回答我"
        )
        assert clipped_kind == "latency_probe"
        assert clipped_reply == "我会等你说完，再简短回答。"
        clipped_mis_asr_kind, clipped_mis_asr_reply = _simple_fast_reply_with_kind(
            "请你一定躲，这句话全部说完以后。"
        )
        assert clipped_mis_asr_kind == "latency_probe"
        assert clipped_mis_asr_reply == "我会等你说完，再简短回答。"
        reachy_base_kind, reachy_base_reply = _simple_fast_reply_with_kind(
            "想白我想測試一下長句子的理解和言直请你等我把这句话全部說完以后再一迷句话回答我"
        )
        assert reachy_base_kind == "latency_probe"
        assert reachy_base_reply == "我会等你说完，再简短回答。"
        reachy_actual_kind, reachy_actual_reply = _simple_fast_reply_with_kind(
            "小白我想測試一下長句子的理解和延遲请你等我把这句话全部收完以后再一迷句话回答我"
        )
        assert reachy_actual_kind == "latency_probe"
        assert reachy_actual_reply == "我会等你说完，再简短回答。"

        reachy_live_latency_kind, reachy_live_latency_reply = _simple_fast_reply_with_kind(
            "小白你好我想测试一向堂俊子的理解可以言之请你跟我说我以后再简单回家"
        )
        assert reachy_live_latency_kind == "latency_probe"
        assert reachy_live_latency_reply == "我会等你说完，再简短回答。"

        reachy_live_latency_2_kind, reachy_live_latency_2_reply = _simple_fast_reply_with_kind(
            "小白你好我想测试一下长去子的理解和言之请你等我说完以后再趕回答"
        )
        assert reachy_live_latency_2_kind == "latency_probe"
        assert reachy_live_latency_2_reply == "我会等你说完，再简短回答。"

    def test_simple_fast_reply_for_turn_ack(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind(
            "小白我现在要说一个长句子来测试你会不会抢答，请等我把这句话全部说完以后再用一句话回答我听明白了。"
        )
        assert kind == "turn_ack"
        assert reply == "我听明白了。"

        clipped_kind, clipped_reply = _simple_fast_reply_with_kind(
            "小白请不要在中间停顿的时。明白了。"
        )
        assert clipped_kind == "turn_ack"
        assert clipped_reply == "我听明白了。"

        clipped_answer_kind, clipped_answer_reply = _simple_fast_reply_with_kind(
            "小白我现在要说一个长句子来测试你会吐。再用一句话回答我，听明白了。"
        )
        assert clipped_answer_kind == "turn_ack"
        assert clipped_answer_reply == "我听明白了。"

        reachy_reply_kind, reachy_reply = _simple_fast_reply_with_kind(
            "小白你好这是一个长距测试请不要强大等我接受再回覆"
        )
        assert reachy_reply_kind == "turn_ack"
        assert reachy_reply == "我听明白了。"

        interruption_probe_kind, interruption_probe_reply = _simple_fast_reply_with_kind(
            "小白你好我想测试让女会会在中途打论请最后再说一句话"
        )
        assert interruption_probe_kind == "turn_ack"
        assert interruption_probe_reply == "我听明白了。"

        reachy_live_pause_kind, reachy_live_pause_reply = _simple_fast_reply_with_kind(
            "小白请不要在中间停顿的时候搶答最后只需要说我听你敗了"
        )
        assert reachy_live_pause_kind == "turn_ack"
        assert reachy_live_pause_reply == "我听明白了。"

        reachy_live_pause_2_kind, reachy_live_pause_2_reply = _simple_fast_reply_with_kind(
            "小白不要在中间停盾的时候抢答了以后只需要说我以你了"
        )
        assert reachy_live_pause_2_kind == "turn_ack"
        assert reachy_live_pause_2_reply == "我听明白了。"

        reachy_live_wait_kind, reachy_live_wait_reply = _simple_fast_reply_with_kind(
            "小白你好这是一个場具此事请不要想打答我结束之后再回复"
        )
        assert reachy_live_wait_kind == "turn_ack"
        assert reachy_live_wait_reply == "我听明白了。"

        reachy_live_wait_2_kind, reachy_live_wait_2_reply = _simple_fast_reply_with_kind(
            "小白你好这是一个場具测试请不要搶打答我结束之后再回复"
        )
        assert reachy_live_wait_2_kind == "turn_ack"
        assert reachy_live_wait_2_reply == "我听明白了。"

    def test_simple_fast_reply_for_noise_management_instruction(self):
        from ._runtime import _simple_fast_reply_with_kind

        typing_kind, typing_reply = _simple_fast_reply_with_kind(
            "小白我旁边有打字声音你只需要回答我听到了"
        )
        assert typing_kind == "noise_instruction"
        assert typing_reply == "我听到了。"

        tv_kind, tv_reply = _simple_fast_reply_with_kind(
            "小白如果电视里有人说话你应该等我叫你之后再回答"
        )
        assert tv_kind == "noise_instruction"
        assert tv_reply == "好的，没叫我我就不接话。"

        background_kind, background_reply = _simple_fast_reply_with_kind(
            "电视里有人说话你应该等我叫你之后再回答"
        )
        assert background_kind is None
        assert background_reply is None

    def test_simple_fast_reply_for_reachy_live_brief_color_mishear(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind(
            "小白你好请你移居滑回答你今天最喜欢什么颜色为什么"
        )
        assert kind == "brief_color"
        assert reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

        live_kind, live_reply = _simple_fast_reply_with_kind(
            "小白你好请用一句话回答你小白你好请用一句话回答你今天最喜欢什么的色为什么"
        )
        assert live_kind == "brief_color"
        assert live_reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

    def test_brief_voice_request_routes_to_brief_llm(self):
        from ._runtime import _brief_voice_request_kind

        assert _brief_voice_request_kind("小白你觉得北京这个城市怎么样请用一句话回答") == "brief_llm"
        assert _brief_voice_request_kind("小白请简短回答你喜欢什么颜色") == "brief_llm"

    def test_simple_fast_reply_handles_open_question_asr_distortions(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind("小白起点的回答你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白你这位金这个城市怎么样緊用一句话回答")
        assert kind == "brief_opinion"
        assert reply.startswith("我觉得北京这个城市")

    def test_brief_llm_defaults_on_for_reachy_voice(self, monkeypatch):
        from ._runtime import _fast_brief_default_enabled

        monkeypatch.delenv("MOSS_VOICE_INPUT_BACKEND", raising=False)
        monkeypatch.delenv("REACHY_ROBOT_HOST", raising=False)
        assert not _fast_brief_default_enabled()

        monkeypatch.setenv("MOSS_VOICE_INPUT_BACKEND", "reachy")
        assert _fast_brief_default_enabled()

    def test_brief_voice_request_keeps_body_actions_on_full_llm(self):
        from ._runtime import _brief_voice_request_kind

        assert _brief_voice_request_kind("小白动动脑袋并用一句话回答") is None
        assert _brief_voice_request_kind("小白点头说收到") is None

    def test_simple_fast_reply_for_direct_body_actions(self):
        from ._runtime import _simple_fast_reply_for_text, _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind("小白点头说收到")
        assert kind == "head_move"
        assert reply == (
            '<apps.bodies_reachymini:head_move pitch="-8" duration="0.5"/>我点一下头。'
        )

        kind, reply = _simple_fast_reply_with_kind("小白动动脑袋")
        assert kind == "head_move"
        assert reply == (
            '<apps.bodies_reachymini:head_move yaw="10" duration="0.6"/>我现在动动脑袋。'
        )

        assert _simple_fast_reply_for_text("小白请点头并用一句话回答你喜欢什么颜色") is None

    def test_simple_fast_reply_for_brief_opinion(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind("小白你觉得北京这个城市怎么样请用一句话回答")

        assert kind == "brief_opinion"
        assert reply == "我觉得北京这个城市有活力。"

    def test_simple_fast_reply_for_common_open_brief_questions(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind(
            "小白请用一句话解单回答地今天最喜欢什么颜色为什么"
        )
        assert kind == "brief_color"
        assert reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

        kind, reply = _simple_fast_reply_with_kind(
            "小白姐用一句话解答回答你一天自己的神点色为什么"
        )
        assert kind == "brief_color"
        assert reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

        kind, reply = _simple_fast_reply_with_kind(
            "小白琴用一句话简单回答你今天自行二十年次为什么"
        )
        assert kind == "brief_color"
        assert reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

        kind, reply = _simple_fast_reply_with_kind(
            "小白请丁秘去换简单回答你今天最喜欢手机的字会什么"
        )
        assert kind == "brief_color"
        assert reply == "我喜欢蓝色，因为它像天空一样安静又可靠。"

        kind, reply = _simple_fast_reply_with_kind("小白请简短回答你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白请请回答你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白请简直回答你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白情简直回家你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白晴监督我让你最喜欢做什么")
        assert kind == "brief_preference"
        assert reply == "我最喜欢听你说话，然后把回答和动作配合好。"

        kind, reply = _simple_fast_reply_with_kind("小白请简单回答机器人为什么需要耳朵")
        assert kind == "brief_robot_ears"
        assert reply == "机器人需要耳朵，是为了听见你、理解你，再及时回应你。"

        kind, reply = _simple_fast_reply_with_kind("小白请简单回答机器人为時需要耳朵")
        assert kind == "brief_robot_ears"
        assert reply == "机器人需要耳朵，是为了听见你、理解你，再及时回应你。"

        kind, reply = _simple_fast_reply_with_kind("小白一起写单回家机器人为什么去哪儿做")
        assert kind == "brief_robot_ears"
        assert reply == "机器人需要耳朵，是为了听见你、理解你，再及时回应你。"

    def test_simple_fast_reply_for_noisy_brief_opinion(self):
        from ._runtime import _simple_fast_reply_with_kind

        kind, reply = _simple_fast_reply_with_kind(
            "小白李觉得北京这个城市怎么样请一尼俊望回答"
        )

        assert kind == "brief_opinion"
        assert reply == "我觉得北京这个城市有活力。"

        kind, reply = _simple_fast_reply_with_kind(
            "小白领觉得北京这个城市怎么样请用一句话回答"
        )

        assert kind == "brief_opinion"
        assert reply == "我觉得北京这个城市有活力。"

        kind, reply = _simple_fast_reply_with_kind(
            "小白你觉得北京这个城市怎么样请你进换回答"
        )

        assert kind == "brief_opinion"
        assert reply == "我觉得北京这个城市有活力。"

    def test_common_open_brief_fast_reply_keeps_actions_on_full_llm(self):
        from ._runtime import _simple_fast_reply_for_text

        assert _simple_fast_reply_for_text("小白请点头并用一句话回答你喜欢什么颜色") is None
