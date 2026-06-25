"""Shared Xiaobai contracts for 1.0 and 2.0 runtimes."""

from .contracts import (
    AudioOutput,
    AudioOutputAdapter,
    BrainAdapter,
    MotionOutputAdapter,
    MotionPlan,
    MotionStep,
    ReplyPlan,
    RuntimeKind,
    TurnInput,
    TurnInputAdapter,
)

__all__ = [
    "AudioOutput",
    "AudioOutputAdapter",
    "BrainAdapter",
    "MotionOutputAdapter",
    "MotionPlan",
    "MotionStep",
    "ReplyPlan",
    "RuntimeKind",
    "TurnInput",
    "TurnInputAdapter",
]
