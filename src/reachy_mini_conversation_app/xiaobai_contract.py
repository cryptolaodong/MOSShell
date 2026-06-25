"""Shared Xiaobai contract shapes for official 1.0 adapters."""

from enum import Enum
from dataclasses import field, dataclass


class RuntimeKind(str, Enum):
    """Runtime family hosting a Xiaobai capability."""

    OFFICIAL_1 = "xiaobai_1_official"
    MOSS_2 = "xiaobai_2_moss"


@dataclass(slots=True)
class TurnInput:
    """One committed user turn."""

    text: str
    runtime: RuntimeKind
    source: str = "text"
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MotionStep:
    """One safe robot motion request."""

    name: str
    params: dict[str, object] = field(default_factory=dict)
    start_offset_s: float = 0.0
    duration_s: float = 1.0


@dataclass(slots=True)
class MotionPlan:
    """Robot motion attached to a reply."""

    steps: list[MotionStep] = field(default_factory=list)
    sync_to_speech: bool = True


@dataclass(slots=True)
class AudioOutput:
    """Speech output attached to a reply."""

    text: str
    voice: str | None = None
    emotion: str | None = None


@dataclass(slots=True)
class ReplyPlan:
    """Runtime-neutral Xiaobai reply plan."""

    audio: AudioOutput
    motion: MotionPlan = field(default_factory=MotionPlan)
    should_reply: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


def make_official_turn(text: str, *, source: str = "official_profile") -> TurnInput:
    """Create a Xiaobai turn from the official Conversation App."""
    return TurnInput(text=text, runtime=RuntimeKind.OFFICIAL_1, source=source)


def make_reply_plan(text: str, *, motion_steps: list[MotionStep] | None = None) -> ReplyPlan:
    """Create a Xiaobai reply plan for adapters and tests."""
    return ReplyPlan(
        audio=AudioOutput(text=text),
        motion=MotionPlan(steps=motion_steps or []),
    )
