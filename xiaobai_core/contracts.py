"""Stable contracts shared by Xiaobai 1.0 and 2.0.

Feature modules should depend on these small data shapes instead of depending
directly on a specific runtime such as the turn-based 1.0 loop or the MOSS 2.0
realtime chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class RuntimeKind(StrEnum):
    """Runtime families that can host the same Xiaobai feature module."""

    TURN_BASED_1 = "xiaobai_1_turn_based"
    MOSS_REALTIME_2 = "xiaobai_2_moss_realtime"


@dataclass(slots=True)
class TurnInput:
    """One user turn after the input adapter has committed it."""

    text: str
    runtime: RuntimeKind
    source: str = "text"
    started_at: float | None = None
    ended_at: float | None = None
    confidence: float | None = None
    audio_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MotionStep:
    """A single motion that can be scheduled against speech."""

    name: str
    start_offset: float = 0.0
    duration: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MotionPlan:
    """Robot body work for one reply."""

    steps: list[MotionStep] = field(default_factory=list)
    sync_to_speech: bool = True


@dataclass(slots=True)
class AudioOutput:
    """Speech or sound output for one reply."""

    text: str
    voice: str | None = None
    emotion: str | None = None
    audio_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplyPlan:
    """The runtime-neutral output that a Xiaobai feature produces."""

    audio: AudioOutput
    motion: MotionPlan = field(default_factory=MotionPlan)
    should_reply: bool = True
    safety_level: str = "safe"
    metadata: dict[str, Any] = field(default_factory=dict)


class TurnInputAdapter(Protocol):
    """Runtime-specific input adapter."""

    async def next_turn(self) -> TurnInput:
        """Return the next committed user turn."""


class BrainAdapter(Protocol):
    """Runtime-neutral reasoning or feature module."""

    async def plan_reply(self, turn: TurnInput) -> ReplyPlan:
        """Produce one reply plan for one committed user turn."""


class AudioOutputAdapter(Protocol):
    """Runtime-specific audio player."""

    async def play_audio(self, audio: AudioOutput) -> None:
        """Speak or play the planned audio."""


class MotionOutputAdapter(Protocol):
    """Runtime-specific motion executor."""

    async def play_motion(self, motion: MotionPlan) -> None:
        """Run robot motion, preferably synchronized with speech."""
