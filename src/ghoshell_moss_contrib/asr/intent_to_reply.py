from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ghoshell_moss_contrib.asr.voice_turn_gate import (
    VoiceTurnDecision,
    looks_like_active_followup_request,
    normalize_voice_text,
)


BACKGROUND_CONTEXT_TOKENS = (
    "打字",
    "键盘",
    "敲键盘",
    "电视",
    "视频",
    "音箱",
    "音乐",
    "旁边",
    "别人",
    "其他人",
    "有人",
    "聊天",
    "说话",
    "背景",
    "杂音",
    "噪声",
    "噪音",
)

DIRECT_REQUEST_TOKENS = (
    "小白",
    "请",
    "帮",
    "回答",
    "回复",
    "告诉",
    "解释",
    "介绍",
    "应该",
    "需要",
    "能不能",
    "可以",
    "不要",
    "别",
    "等我",
    "说完",
    "一句话",
    "听明白",
    "你",
)


@dataclass(frozen=True)
class IntentToReplyDecision:
    action: str
    reason: str
    text_len: int
    addressed: bool
    active_before: bool
    gate_reason: str
    background_context: bool
    direct_request: bool

    @property
    def should_reply(self) -> bool:
        return self.action == "reply"


def looks_like_background_context(text: str) -> bool:
    normalized = normalize_voice_text(text)
    return any(token in normalized for token in BACKGROUND_CONTEXT_TOKENS)


def looks_like_direct_request(
    text: str,
    *,
    extra_keywords: Sequence[str] = (),
    min_chars: int = 2,
    max_chars: int = 90,
) -> bool:
    normalized = normalize_voice_text(text)
    if not normalized:
        return False
    tokens = tuple(DIRECT_REQUEST_TOKENS) + tuple(token for token in extra_keywords if token)
    if any(token and token in normalized for token in tokens):
        return True
    return looks_like_active_followup_request(
        text,
        keywords=extra_keywords or (),
        min_chars=min_chars,
        max_chars=max_chars,
    )


def decide_intent_to_reply(
    text: str,
    gate_decision: VoiceTurnDecision,
    *,
    extra_followup_keywords: Sequence[str] = (),
    followup_min_chars: int = 2,
    followup_max_chars: int = 90,
) -> IntentToReplyDecision:
    stripped = (text or "").strip()
    background_context = looks_like_background_context(stripped)
    direct_request = looks_like_direct_request(
        stripped,
        extra_keywords=extra_followup_keywords,
        min_chars=followup_min_chars,
        max_chars=followup_max_chars,
    )

    if not stripped:
        return IntentToReplyDecision(
            "drop",
            "empty",
            0,
            gate_decision.addressed,
            gate_decision.active_before,
            gate_decision.reason,
            background_context,
            direct_request,
        )

    if not gate_decision.accept:
        reason = gate_decision.reason
        if background_context:
            reason = f"{reason}_background_context"
        return IntentToReplyDecision(
            "drop",
            reason,
            len(stripped),
            gate_decision.addressed,
            gate_decision.active_before,
            gate_decision.reason,
            background_context,
            direct_request,
        )

    if gate_decision.addressed:
        return IntentToReplyDecision(
            "reply",
            "addressed_direct" if direct_request else "addressed",
            len(stripped),
            True,
            gate_decision.active_before,
            gate_decision.reason,
            background_context,
            direct_request,
        )

    if gate_decision.reason in {"wake_recovery_followup", "clipped_address_rescue"}:
        return IntentToReplyDecision(
            "reply",
            gate_decision.reason,
            len(stripped),
            False,
            gate_decision.active_before,
            gate_decision.reason,
            background_context,
            direct_request,
        )

    if gate_decision.reason == "active_followup":
        if background_context and not direct_request:
            return IntentToReplyDecision(
                "drop",
                "active_followup_background_context",
                len(stripped),
                False,
                gate_decision.active_before,
                gate_decision.reason,
                background_context,
                direct_request,
            )
        if not direct_request:
            return IntentToReplyDecision(
                "drop",
                "active_followup_no_request",
                len(stripped),
                False,
                gate_decision.active_before,
                gate_decision.reason,
                background_context,
                direct_request,
            )
        return IntentToReplyDecision(
            "reply",
            "active_followup_direct",
            len(stripped),
            False,
            gate_decision.active_before,
            gate_decision.reason,
            background_context,
            direct_request,
        )

    return IntentToReplyDecision(
        "reply",
        gate_decision.reason,
        len(stripped),
        gate_decision.addressed,
        gate_decision.active_before,
        gate_decision.reason,
        background_context,
        direct_request,
    )
