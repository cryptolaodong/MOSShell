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
    "可以做什么",
    "会做什么",
    "你会什么",
    "你能干什么",
    "能干什么",
    "有什么功能",
)
_FAST_CAPABILITY_DETAIL_WORDS = ("详细", "展开", "具体", "列表", "所有")


def _request_text(parts) -> str:
    """Extract text-only user content for local deterministic shortcuts."""
    last_text = ""
    for part in parts:
        content = getattr(part, "content", None)
        if isinstance(content, str) and content.strip():
            last_text = content.strip()
    return last_text


def _simple_fast_reply_with_kind(text: str) -> tuple[str | None, str | None]:
    normalized = _FAST_GREETING_STRIP_RE.sub("", text).lower()
    if normalized in _FAST_GREETING_WORDS:
        return "greeting", os.environ.get("MOSS_FAST_GREETING_REPLY", "在呢。")
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
    return None, None


def _simple_fast_reply_for_text(text: str) -> str | None:
    _, reply = _simple_fast_reply_with_kind(text)
    return reply


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

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
