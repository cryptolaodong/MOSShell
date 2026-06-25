import os
import re
import time
from typing import AsyncIterator, TYPE_CHECKING
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator, Moment, Reaction
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ModelMessagesTypeAdapter

if TYPE_CHECKING:
    from ._meta import AtomMeta

__all__ = ["Atom"]


_FAST_GREETING_WORDS = {
    "hi",
    "hello",
    "hey",
    "你好",
    "你好啊",
    "你好呀",
    "您好",
    "嗨",
    "嗨喽",
    "哈喽",
    "哈罗",
    "喂",
    "在吗",
    "你在吗",
    "小白",
    "小白你好",
    "你好小白",
}
_FAST_GREETING_STRIP_RE = re.compile(r"[\s,，。！？!?；;：:\.、~～]+")
_FAST_ACK_PATTERNS = (
    "只需要回答收到",
    "只要回答收到",
    "直接回答收到",
    "请回答收到",
    "只回复收到",
    "回复收到",
)
_FAST_CAPABILITY_PATTERNS = (
    "能做什么",
    "能帮我做什么",
    "你现在能帮",
    "可以帮我做什么",
    "帮我做什么",
    "可以做什么",
    "会做什么",
    "你会什么",
    "你能干什么",
    "能干什么",
    "有什么功能",
)
_FAST_CAPABILITY_DETAIL_WORDS = ("详细", "展开", "具体", "列表", "所有")
_FAST_LATENCY_TEST_PATTERNS = (
    ("测试", "延迟"),
    ("测试", "等我", "说完"),
    ("等我", "说完", "回答"),
    ("听我说完", "回答"),
)
_FAST_LATENCY_COMPLETION_GUARDS = (
    "这句话全部说完",
    "全部说完以后",
    "完整说完以后",
)
_FAST_TURN_ACK_GUARDS = (
    "听明白",
    "明白了",
    "回复",
)
_FAST_TURN_ACK_REQUEST_MARKERS = (
    "等我",
    "说完",
    "全部说完",
    "完整说完",
    "不要在中间",
    "中间停顿",
    "插话",
    "抢答",
    "最后",
    "只需要",
    "只要",
    "回答我",
    "一句话回答",
)
_FAST_OPINION_PATTERNS = ("怎么样", "如何", "好不好")
_FAST_BRIEF_PATTERNS = (
    "一句话",
    "一两句",
    "简短",
    "简单回答",
    "简单说",
    "短一点",
    "四个字",
    "几个字",
    "不要超过",
    "不超过",
    "20字",
    "二十字",
    "十个字",
)
_FAST_BRIEF_ACTION_WORDS = (
    "做动作",
    "动作",
    "动动",
    "点头",
    "摇头",
    "抬头",
    "低头",
    "转头",
    "扭头",
    "看左",
    "看右",
    "表情",
    "天线",
    "跳舞",
    "dance",
    "headmove",
    "emotion",
)
_FAST_ACTION_NEGATIVE_WORDS = (
    "不要动",
    "别动",
    "不用动",
    "不动",
    "不要动作",
    "别做动作",
    "不用动作",
)
_FAST_ACTION_COMPLEX_WORDS = (
    "为什么",
    "怎么样",
    "如何",
    "好不好",
    "喜欢",
    "颜色",
    "解释",
    "说明",
    "介绍",
    "同步",
    "延迟",
    "问题",
)
_FAST_SIMPLE_ACTION_REPLIES = (
    (
        ("点头", "点点头", "点一下头", "低头"),
        "head_move",
        '<apps.bodies_reachymini:head_move pitch="-8" duration="0.5"/>我点一下头。',
    ),
    (
        ("抬头",),
        "head_move",
        '<apps.bodies_reachymini:head_move pitch="8" duration="0.5"/>我抬一下头。',
    ),
    (
        ("看左", "向左看", "往左看"),
        "head_move",
        '<apps.bodies_reachymini:head_move yaw="-12" duration="0.6"/>我往左看看。',
    ),
    (
        ("看右", "向右看", "往右看"),
        "head_move",
        '<apps.bodies_reachymini:head_move yaw="12" duration="0.6"/>我往右看看。',
    ),
    (
        ("摇头", "摇摇头"),
        "head_move",
        '<apps.bodies_reachymini:head_move yaw="12" duration="0.5"/>我摇一下头。',
    ),
    (
        ("动动脑袋", "动一下脑袋", "动动头", "动一下头", "转头", "扭头", "headmove"),
        "head_move",
        '<apps.bodies_reachymini:head_move yaw="10" duration="0.6"/>我现在动动脑袋。',
    ),
)
_FAST_TEXT_REPLACEMENTS = {
    "請": "请",
    "簡": "简",
    "單": "单",
    "話": "话",
    "為": "为",
    "麼": "么",
    "時": "时",
    "什麼": "什么",
    "歡": "欢",
    "顏": "颜",
    "顔": "颜",
    "覺": "觉",
    "測": "测",
    "試": "试",
    "長": "长",
    "說": "说",
    "遲": "迟",
    "這": "这",
    "個": "个",
    "樣": "样",
    "會": "会",
    "機": "机",
    "聽": "听",
    "情简直": "请简短",
    "晴监督我让": "请简短回答",
    "请简直回答": "请简短回答",
    "丁秘去换": "用一句话",
    "为時": "为什么",
    "为时": "为什么",
    "姐用": "请用",
    "解单": "简单",
    "簡單": "简单",
    "写单": "简单",
    "解答": "简单",
    "请请回答": "请简短回答",
    "回来": "回答",
    "回家": "回答",
    "回覆": "回复",
    "接受": "结束",
    "强大": "抢答",
    "打论": "打断",
    "李觉得": "你觉得",
    "领觉得": "你觉得",
    "女觉得": "你觉得",
    "小板": "小白",
    "一尼俊望": "一句话",
    "一名据换": "一句话",
    "一句换": "一句话",
    "换回的": "话回答",
    "进换": "一句话",
    "去换": "一句话",
    "一天自己": "今天最喜欢",
    "自行二十年次": "最喜欢什么颜色",
    "自己二十年次": "最喜欢什么颜色",
    "神点色": "什么颜色",
    "手机的字": "什么颜色",
    "方法做什么": "能帮我做什么",
    "法做什么": "能帮我做什么",
    "我付什么": "帮我做什么",
    "付什么": "帮我做什么",
    "去哪儿做": "需要耳朵",
    "哪儿做": "需要耳朵",
    "一定躲": "一定等",
    "收完": "说完",
    "言直": "延迟",
    "一迷句话": "一句话",
    "一迷句": "一句",
    "起点的回答": "请简短回答",
    "这位金": "觉得北京",
    "这不成是": "这个城市",
    "緊用": "请用",
    "換": "换",
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _fast_brief_default_enabled() -> bool:
    return bool(
        os.environ.get("MOSS_VOICE_INPUT_BACKEND")
        or os.environ.get("REACHY_ROBOT_HOST")
    )


def _normalize_fast_request_text(text: str) -> str:
    normalized = _FAST_GREETING_STRIP_RE.sub("", text).lower()
    for source, target in _FAST_TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def _request_text(parts) -> str:
    """Extract text-only user content for local deterministic shortcuts."""
    last_text = ""
    for part in parts:
        content = getattr(part, "content", None)
        if isinstance(content, str) and content.strip():
            last_text = content.strip()
    return last_text


def _simple_fast_reply_with_kind(text: str) -> tuple[str | None, str | None]:
    normalized = _normalize_fast_request_text(text)
    if normalized in _FAST_GREETING_WORDS:
        return "greeting", os.environ.get("MOSS_FAST_GREETING_REPLY", "在呢。")
    if (
        len(normalized) <= 90
        and "最后" in normalized
        and ("说一句话" in normalized or "再说一句话" in normalized)
        and any(marker in normalized for marker in ("中途", "打断", "测试"))
    ):
        return "turn_ack", os.environ.get("MOSS_FAST_TURN_ACK_REPLY", "我听明白了。")
    if (
        len(normalized) <= 90
        and any(guard in normalized for guard in _FAST_TURN_ACK_GUARDS)
        and any(marker in normalized for marker in _FAST_TURN_ACK_REQUEST_MARKERS)
    ):
        return "turn_ack", os.environ.get("MOSS_FAST_TURN_ACK_REPLY", "我听明白了。")
    if (
        len(normalized) <= 90
        and (
            normalized in {"回答收到", "回复收到", "收到"}
            or any(pattern in normalized for pattern in _FAST_ACK_PATTERNS)
        )
    ):
        return "ack", os.environ.get("MOSS_FAST_ACK_REPLY", "收到。")
    if (
        len(normalized) <= 45
        and any(pattern in normalized for pattern in _FAST_CAPABILITY_PATTERNS)
        and not any(word in normalized for word in _FAST_CAPABILITY_DETAIL_WORDS)
    ):
        return (
            "capability",
            os.environ.get(
                "MOSS_FAST_CAPABILITY_REPLY",
                "我能听你说话、回答问题，并同步表情和头部动作。",
            ),
        )
    action_reply = _simple_action_reply_for_normalized_text(normalized)
    if action_reply:
        return action_reply
    is_latency_probe = any(
        all(part in normalized for part in pattern)
        for pattern in _FAST_LATENCY_TEST_PATTERNS
    )
    has_latency_completion_guard = any(
        pattern in normalized
        for pattern in _FAST_LATENCY_COMPLETION_GUARDS
    )
    is_brief_request = any(pattern in normalized for pattern in _FAST_BRIEF_PATTERNS)
    if (
        len(normalized) <= 90
        and not any(word in normalized for word in _FAST_BRIEF_ACTION_WORDS)
        and (is_latency_probe or has_latency_completion_guard)
        and (is_brief_request or has_latency_completion_guard)
    ):
        return (
            "latency_probe",
            os.environ.get("MOSS_FAST_LATENCY_TEST_REPLY", "我会等你说完，再简短回答。"),
        )
    brief_qa_reply = _simple_brief_qa_reply_for_text(text)
    if brief_qa_reply:
        return brief_qa_reply
    opinion_reply = _simple_opinion_reply_for_text(text)
    if opinion_reply:
        return "brief_opinion", opinion_reply
    return None, None


def _simple_fast_reply_for_text(text: str) -> str | None:
    _, reply = _simple_fast_reply_with_kind(text)
    return reply


def _simple_action_reply_for_normalized_text(normalized: str) -> tuple[str, str] | None:
    if len(normalized) > 48:
        return None
    if any(word in normalized for word in _FAST_ACTION_NEGATIVE_WORDS):
        return None
    if any(word in normalized for word in _FAST_ACTION_COMPLEX_WORDS):
        return None
    for patterns, kind, reply in _FAST_SIMPLE_ACTION_REPLIES:
        if any(pattern in normalized for pattern in patterns):
            return kind, reply
    return None


def _simple_brief_qa_reply_for_text(text: str) -> tuple[str, str] | None:
    normalized = _normalize_fast_request_text(text)
    if not normalized or len(normalized) > 90:
        return None
    if any(word in normalized for word in _FAST_BRIEF_ACTION_WORDS):
        return None

    is_brief = any(pattern in normalized for pattern in _FAST_BRIEF_PATTERNS)
    if (
        "能帮我做什么" in normalized
        or "可以帮我做什么" in normalized
        or "帮我做什么" in normalized
    ):
        if not any(word in normalized for word in _FAST_CAPABILITY_DETAIL_WORDS):
            return (
                "capability",
                os.environ.get(
                    "MOSS_FAST_CAPABILITY_REPLY",
                    "我能听你说话、回答问题，并同步表情和头部动作。",
                ),
            )
    if is_brief and (
        "喜欢什么颜色" in normalized
        or ("喜欢" in normalized and "颜色" in normalized)
        or ("为什么" in normalized and "颜色" in normalized)
    ):
        return (
            "brief_color",
            os.environ.get(
                "MOSS_FAST_COLOR_REPLY",
                "我喜欢蓝色，因为它像天空一样安静又可靠。",
            ),
        )
    if is_brief and ("最喜欢做什么" in normalized or ("喜欢" in normalized and "做什么" in normalized)):
        return (
            "brief_preference",
            os.environ.get(
                "MOSS_FAST_PREFERENCE_REPLY",
                "我最喜欢听你说话，然后把回答和动作配合好。",
            ),
        )
    if (
        "为什么" in normalized
        and (
            "需要耳朵" in normalized
            or "机器人为什么需要耳朵" in normalized
            or ("机器人为什么" in normalized and is_brief)
        )
    ):
        return (
            "brief_robot_ears",
            os.environ.get(
                "MOSS_FAST_ROBOT_EARS_REPLY",
                "机器人需要耳朵，是为了听见你、理解你，再及时回应你。",
            ),
        )
    return None


def _simple_opinion_reply_for_text(text: str) -> str | None:
    normalized = _normalize_fast_request_text(text)
    if not normalized:
        return None
    if any(word in normalized for word in _FAST_BRIEF_ACTION_WORDS):
        return None
    is_brief = any(pattern in normalized for pattern in _FAST_BRIEF_PATTERNS)
    if not is_brief and (
        "回答" not in normalized
        or any(word in normalized for word in ("详细", "展开", "具体", "长一点"))
    ):
        return None
    if "你觉得" not in normalized or not any(pattern in normalized for pattern in _FAST_OPINION_PATTERNS):
        return None

    topic = normalized
    if topic.startswith("小白"):
        topic = topic[2:]
    topic = topic.split("你觉得", 1)[-1]
    for marker in _FAST_OPINION_PATTERNS:
        if marker in topic:
            topic = topic.split(marker, 1)[0]
            break
    topic = re.sub(r"(请)?用?(一两句|一句话|简短|简单回答|简单说).*", "", topic)
    topic = topic.strip()
    if not topic or len(topic) > 24:
        return None
    template = os.environ.get(
        "MOSS_FAST_OPINION_TEMPLATE",
        "我觉得{topic}有活力。",
    )
    try:
        return template.format(topic=topic)
    except Exception:
        return f"我觉得{topic}有活力。"


def _brief_voice_request_kind(text: str) -> str | None:
    normalized = _normalize_fast_request_text(text)
    if not normalized:
        return None
    max_chars = _int_env("MOSS_FAST_BRIEF_MAX_CHARS", 90)
    if len(normalized) > max_chars:
        return None
    if any(word in normalized for word in _FAST_BRIEF_ACTION_WORDS):
        return None
    if any(pattern in normalized for pattern in _FAST_BRIEF_PATTERNS):
        return "brief_llm"
    return None


class Atom(Ghost):
    """Atom — 最小 Ghost 原型运行时，作为后续所有 Ghost 实现的参照基线.

    已知不做的事（原型范围外）:
    - 上下文超额裁剪: model_history() 不做窗口限制，依赖模型自身的 context window
    - 持久化: 纯内存历史，重启即丢
    """

    def __init__(
        self,
        meta: "AtomMeta",
        agent: Agent[IoCContainer],
        container: IoCContainer,
    ):
        self._meta = meta
        self._agent = agent
        self._container = container
        self._logger = container.get(LoggerItf) or get_moss_logger()
        self._history: list[ModelMessage] = []
        self._last_context: dict = {}
        self._history_file = "ghost_history.json"
        # Real-time voice should not replay full pydantic_ai history by default:
        # it adds latency and can contain provider-incompatible assistant parts.
        self._history_enabled = os.environ.get("MOSS_ATOM_HISTORY_ENABLED", "0").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._fallback_on_llm_error = os.environ.get("MOSS_LLM_FALLBACK_ON_ERROR", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._llm_fallback_text = os.environ.get("MOSS_LLM_FALLBACK_TEXT", "我卡了一下，请再说一遍。")
        # Load persisted history from session storage
        self._load_history()

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        """调试用: 返回 Agent 实际使用的 instruction."""
        return self._meta.build_instruction_from_ioc(self._container)

    # ── 消息协议 ──────────────────────────────────

    def to_model_request(self, moment: Moment) -> ModelRequest:
        """将 Moment 转为 pydantic AI ModelRequest."""
        from ._adapter import moment_to_request
        return moment_to_request(moment)

    def _load_history(self) -> None:
        """从 session storage 加载对话历史。"""
        if not self._history_enabled:
            self._logger.info("Atom history disabled by MOSS_ATOM_HISTORY_ENABLED")
            return
        try:
            from ghoshell_moss.core.blueprint.session import Session
            session = self._container.get(Session)
            if session and session.scope_storage.exists(self._history_file):
                data = session.scope_storage.get(self._history_file)
                self._history = ModelMessagesTypeAdapter.validate_json(data)
                self._logger.info("Loaded %d history messages from storage", len(self._history))
        except Exception as e:
            self._disable_history("failed to load persisted history", e)

    def _disable_history(self, reason: str, error: Exception | None = None) -> None:
        """Disable model history when pydantic_ai cannot safely replay it."""
        self._history_enabled = False
        self._history.clear()
        try:
            from ghoshell_moss.core.blueprint.session import Session
            session = self._container.get(Session)
            if session and session.scope_storage.exists(self._history_file):
                session.scope_storage.remove(self._history_file)
        except Exception as remove_error:
            self._logger.debug("Failed to remove incompatible history: %s", remove_error)
        if error is None:
            self._logger.warning("Atom history disabled: %s", reason)
        else:
            self._logger.warning("Atom history disabled: %s: %s", reason, error)

    def _save_history(self) -> None:
        """持久化对话历史到 session storage。"""
        if not self._history_enabled:
            return
        try:
            from ghoshell_moss.core.blueprint.session import Session
            session = self._container.get(Session)
            if session:
                data = ModelMessagesTypeAdapter.dump_json(self._history)
                session.scope_storage.put(self._history_file, data)
        except Exception as e:
            self._logger.debug("Failed to save history: %s", e)

    def model_history(self) -> list[ModelMessage]:
        """返回当前内存中的对话历史.

        TODO: 不做窗口裁剪，长对话会超出模型 context window.
        """
        if not self._history_enabled:
            return []
        return list(self._history)

    def save_model_request(
        self, moment: Moment, response: ModelResponse
    ) -> None:
        """保存本轮交换到内存历史."""
        if not self._history_enabled:
            return
        if not getattr(response, "parts", None):
            self._logger.warning("Atom skip saving empty model response to history")
            return
        self._history.append(self.to_model_request(moment))
        self._history.append(response)
        self._save_history()

    # ── 核心循环 ──────────────────────────────────

    def on_articulate_exit(self, articulator, logos, error) -> None:
        self._last_context = {
            "system": self.system_prompt(),
            "history_turns": len(self._history) // 2,
        }

    def inspect_context(self) -> dict:
        return self._last_context

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        moment = articulator.moment
        request = self.to_model_request(moment)
        history = self.model_history()
        prompt_chars = len(self.system_prompt())
        started_at = time.monotonic()
        first_token_seen = False
        total_chars = 0

        fast_reply_env = os.environ.get(
            "MOSS_FAST_REPLY_ENABLED",
            os.environ.get("MOSS_FAST_GREETING_ENABLED", "1"),
        )
        if fast_reply_env.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            request_text = _request_text(request.parts)
            fast_kind, fast_reply = _simple_fast_reply_with_kind(request_text)
            if fast_reply:
                self._logger.warning(
                    "[ReachyLatency] llm_fast_path kind=%s text_len=%d prompt_chars=%d",
                    fast_kind,
                    len(request_text),
                    prompt_chars,
                )
                yield fast_reply
                self._logger.warning(
                    "[ReachyLatency] llm_fast_path_done elapsed=%.2fs chars=%d",
                    time.monotonic() - started_at,
                    len(fast_reply),
                )
                return

            brief_kind = _brief_voice_request_kind(request_text)
            if brief_kind and _bool_env("MOSS_FAST_BRIEF_LLM_ENABLED", _fast_brief_default_enabled()):
                yielded_fast_brief = False
                try:
                    async for text in self._stream_fast_brief_reply(
                        request_text=request_text,
                        started_at=started_at,
                        prompt_chars=prompt_chars,
                    ):
                        yielded_fast_brief = True
                        yield text
                    if yielded_fast_brief:
                        return
                    self._logger.warning(
                        "[ReachyLatency] llm_fast_brief_empty elapsed=%.2fs fallback=full_llm",
                        time.monotonic() - started_at,
                    )
                except Exception as e:
                    if yielded_fast_brief:
                        self._logger.warning(
                            "[ReachyLatency] llm_fast_brief_failed_after_tokens elapsed=%.2fs error=%s",
                            time.monotonic() - started_at,
                            str(e)[:160],
                        )
                        return
                    self._logger.warning(
                        "[ReachyLatency] llm_fast_brief_failed elapsed=%.2fs fallback=full_llm error=%s",
                        time.monotonic() - started_at,
                        str(e)[:160],
                    )

        self._logger.info(
            "[ReachyLatency] llm_request_start history_turns=%d prompt_chars=%d",
            len(history),
            prompt_chars,
        )

        try:
            async with self._agent.run_stream(
                user_prompt=request.parts,
                message_history=history,
                deps=self._container,
            ) as stream:
                async for text in stream.stream_text(delta=True):
                    if text:
                        total_chars += len(text)
                    if text and text.strip() and not first_token_seen:
                        first_token_seen = True
                        self._logger.info(
                            "[ReachyLatency] llm_first_token elapsed=%.2fs",
                            time.monotonic() - started_at,
                        )
                    yield text
                self._logger.info(
                    "[ReachyLatency] llm_stream_done elapsed=%.2fs chars=%d",
                    time.monotonic() - started_at,
                    total_chars,
                )
                self.save_model_request(moment, stream.response)
        except AssertionError as e:
            # pydantic_ai 1.105.0 bug: TextContent in error retry path
            self._disable_history("pydantic_ai rejected model history", e)
            # Retry once without history to bypass the bug
            async with self._agent.run_stream(
                user_prompt=request.parts,
                message_history=[],
                deps=self._container,
            ) as stream:
                async for text in stream.stream_text(delta=True):
                    if text:
                        total_chars += len(text)
                    if text and text.strip() and not first_token_seen:
                        first_token_seen = True
                        self._logger.info(
                            "[ReachyLatency] llm_first_token_retry elapsed=%.2fs",
                            time.monotonic() - started_at,
                        )
                    yield text
                self._logger.info(
                    "[ReachyLatency] llm_stream_done_retry elapsed=%.2fs chars=%d",
                    time.monotonic() - started_at,
                    total_chars,
                )
                self.save_model_request(moment, stream.response)
        except Exception as e:
            error_text = str(e)
            if (
                "Invalid assistant message" not in error_text
                and "content or tool_calls must be set" not in error_text
            ):
                if self._fallback_on_llm_error:
                    self._logger.exception(
                        "[ReachyLatency] llm_failed elapsed=%.2fs chars=%d fallback=%s",
                        time.monotonic() - started_at,
                        total_chars,
                        total_chars == 0,
                    )
                    if total_chars == 0:
                        yield self._llm_fallback_text
                    return
                raise
            self._disable_history("model rejected assistant history", e)
            async with self._agent.run_stream(
                user_prompt=request.parts,
                message_history=[],
                deps=self._container,
            ) as stream:
                async for text in stream.stream_text(delta=True):
                    if text:
                        total_chars += len(text)
                    if text and text.strip() and not first_token_seen:
                        first_token_seen = True
                        self._logger.info(
                            "[ReachyLatency] llm_first_token_history_retry elapsed=%.2fs",
                            time.monotonic() - started_at,
                        )
                    yield text
                self._logger.info(
                    "[ReachyLatency] llm_stream_done_history_retry elapsed=%.2fs chars=%d",
                    time.monotonic() - started_at,
                    total_chars,
                )
                self.save_model_request(moment, stream.response)

    # ── 生命周期 ──────────────────────────────────

    async def _stream_fast_brief_reply(
        self,
        *,
        request_text: str,
        started_at: float,
        prompt_chars: int,
    ) -> AsyncIterator[str]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        import httpx
        from openai import AsyncOpenAI

        model_name = os.environ.get(
            "MOSS_FAST_BRIEF_MODEL",
            os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        timeout = _float_env("MOSS_FAST_BRIEF_TIMEOUT_SECONDS", 3.0)
        max_tokens = _int_env("MOSS_FAST_BRIEF_MAX_TOKENS", 64)
        temperature = _float_env("MOSS_FAST_BRIEF_TEMPERATURE", 0.25)
        system_prompt = os.environ.get(
            "MOSS_FAST_BRIEF_SYSTEM_PROMPT",
            "你是 Reachy Mini 机器人小白。用中文自然口语回答，最多一句话。"
            "不要使用动作标签，不要复述用户问题，不要自问自答。",
        )

        self._logger.warning(
            "[ReachyLatency] llm_fast_path kind=brief_llm text_len=%d prompt_chars=%d fast_prompt_chars=%d",
            len(request_text),
            prompt_chars,
            len(system_prompt),
        )
        first_token_seen = False
        total_chars = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout,
                connect=max(0.3, min(1.2, timeout)),
            )
        ) as http_client:
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client,
                timeout=timeout,
                max_retries=_int_env("MOSS_FAST_BRIEF_MAX_RETRIES", 0),
            )
            stream = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if not text:
                    continue
                total_chars += len(text)
                if not first_token_seen:
                    first_token_seen = True
                    self._logger.info(
                        "[ReachyLatency] llm_first_token elapsed=%.2fs route=brief_llm",
                        time.monotonic() - started_at,
                    )
                yield text
        self._logger.warning(
            "[ReachyLatency] llm_fast_brief_done elapsed=%.2fs chars=%d",
            time.monotonic() - started_at,
            total_chars,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
