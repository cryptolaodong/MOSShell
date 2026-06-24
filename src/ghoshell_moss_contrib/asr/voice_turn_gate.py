from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence


def normalize_voice_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    replacements = {
        "請": "请",
        "簡": "简",
        "單": "单",
        "話": "话",
        "為": "为",
        "麼": "么",
        "什麼": "什么",
        "歡": "欢",
        "顏": "颜",
        "顔": "颜",
        "覺": "觉",
        "這": "这",
        "個": "个",
        "樣": "样",
        "會": "会",
        "機": "机",
        "聽": "听",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.strip(" \t\r\n，,。！？!?；;：:")
    for mark in ("，", ",", "。", "！", "!", "？", "?", "；", ";", "：", ":", "、"):
        normalized = normalized.replace(mark, "")
    return normalized.replace(" ", "")


def looks_like_clipped_address_request(
    text: str,
    *,
    prefixes: Sequence[str],
    keywords: Sequence[str],
    min_chars: int = 12,
    max_chars: int = 90,
) -> bool:
    normalized = normalize_voice_text(text)
    if not normalized or len(normalized) < min_chars or len(normalized) > max_chars:
        return False
    cleaned_prefixes = tuple(normalize_voice_text(prefix) for prefix in prefixes if prefix)
    cleaned_keywords = tuple(normalize_voice_text(keyword) for keyword in keywords if keyword)
    if not cleaned_prefixes or not cleaned_keywords:
        return False
    if not any(normalized.startswith(prefix) for prefix in cleaned_prefixes):
        return False
    return any(keyword in normalized for keyword in cleaned_keywords)


@dataclass(frozen=True)
class VoiceTurnDecision:
    accept: bool
    reason: str
    addressed: bool
    active_before: bool
    active_left_seconds: float
    text_len: int


@dataclass(frozen=True)
class VoicePartialDecision:
    commit: bool
    reason: str
    addressed: bool
    active_before: bool
    active_left_seconds: float
    age_seconds: float
    text_len: int


class VoiceTurnGate:
    """
    Small, testable gate for voice turns.

    Policy:
    - When idle, a final ASR turn must contain a wake/address word.
    - After an accepted turn, follow-up turns are accepted for a short window.
    - Idle unaddressed partial commits are disabled when the configured char
      threshold is intentionally high.
    """

    def __init__(
        self,
        *,
        require_address_when_idle: bool = True,
        address_words: Sequence[str] = ("小白",),
        active_seconds: float = 12.0,
        idle_partial_chars: int = 999,
        idle_partial_seconds: float = 1.2,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.require_address_when_idle = bool(require_address_when_idle)
        self.address_words = tuple(word.strip() for word in address_words if word and word.strip())
        self.active_seconds = max(0.0, float(active_seconds))
        self.idle_partial_chars = max(1, int(idle_partial_chars))
        self.idle_partial_seconds = max(0.0, float(idle_partial_seconds))
        self._clock = clock or time.monotonic
        self._active_until = 0.0
        self._idle_partial_first_ts = 0.0
        self._idle_partial_last_commit_ts = 0.0

    def active_left(self, now: float | None = None) -> float:
        timestamp = self._clock() if now is None else now
        return max(0.0, self._active_until - timestamp)

    def reset_idle_partial(self) -> None:
        self._idle_partial_first_ts = 0.0

    def open_followup_window(self, now: float | None = None) -> None:
        timestamp = self._clock() if now is None else now
        if self.active_seconds > 0:
            self._active_until = timestamp + self.active_seconds
        self.reset_idle_partial()

    def is_addressed(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(word.lower() in normalized for word in self.address_words)

    def decide_final(self, text: str, now: float | None = None) -> VoiceTurnDecision:
        timestamp = self._clock() if now is None else now
        stripped = (text or "").strip()
        addressed = self.is_addressed(stripped)
        left = self.active_left(timestamp)
        active_before = left > 0

        if not stripped:
            return VoiceTurnDecision(False, "empty", addressed, active_before, round(left, 3), 0)

        if self.require_address_when_idle and not active_before and not addressed:
            self.reset_idle_partial()
            return VoiceTurnDecision(
                False,
                "idle_not_addressed",
                addressed,
                active_before,
                round(left, 3),
                len(stripped),
            )

        if self.active_seconds > 0:
            self._active_until = timestamp + self.active_seconds
        self.reset_idle_partial()
        reason = "addressed" if addressed else "active_followup"
        return VoiceTurnDecision(True, reason, addressed, active_before, round(left, 3), len(stripped))

    def decide_partial(self, text: str, now: float | None = None) -> VoicePartialDecision:
        timestamp = self._clock() if now is None else now
        stripped = (text or "").strip()
        addressed = self.is_addressed(stripped)
        left = self.active_left(timestamp)
        active_before = left > 0

        if not stripped:
            self.reset_idle_partial()
            return VoicePartialDecision(False, "empty", addressed, active_before, round(left, 3), 0.0, 0)

        if not self.require_address_when_idle or active_before or addressed:
            self.reset_idle_partial()
            return VoicePartialDecision(
                False,
                "addressed_or_active",
                addressed,
                active_before,
                round(left, 3),
                0.0,
                len(stripped),
            )

        if self._idle_partial_first_ts <= 0:
            self._idle_partial_first_ts = timestamp
        age = timestamp - self._idle_partial_first_ts
        should_commit = (
            len(stripped) >= self.idle_partial_chars
            and age >= self.idle_partial_seconds
            and timestamp - self._idle_partial_last_commit_ts >= 1.0
        )
        if should_commit:
            self._idle_partial_last_commit_ts = timestamp
        return VoicePartialDecision(
            should_commit,
            "idle_unaddressed_commit" if should_commit else "idle_unaddressed_wait",
            addressed,
            active_before,
            round(left, 3),
            round(age, 3),
            len(stripped),
        )
