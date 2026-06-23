"""Helpers for aligning Reachy Mini body actions with speech playback."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx

from .speaking_gate import is_speaking, is_speech_pending, remaining_seconds

DEFAULT_TIMEOUT_SECONDS = 25.0
PENDING_GRACE_SECONDS = 1.0
POLL_INTERVAL_SECONDS = 0.05
ACTION_SYNC_OFFSET_SECONDS = 0.2


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def should_sync_interpreter_command(command_name: str | None = None) -> bool:
    """Return whether the current command came from an interpreter response."""
    task = ChannelCtx.task()
    if task is None:
        return False
    if command_name is not None and task.meta.name != command_name:
        return False
    return bool(task.context.get("interpreter_id"))


async def wait_for_speech_start(
    command_name: str,
    logger: LoggerItf | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Wait until upload-based TTS playback has actually started.

    Body/tool commands may be emitted just before the audio player sends
    ``play_uploaded_audio``. Waiting on the shared speaking gate prevents the
    robot from moving visibly before it starts talking.
    """
    if os.environ.get("MOSS_REACHY_SYNC_BODY_TO_SPEECH", "1").lower() in {"0", "false", "no"}:
        return False

    log = logger or logging.getLogger("ReachyMiniSpeechSync")
    started_at = time.monotonic()
    deadline = started_at + max(0.0, timeout)
    pending_deadline = started_at + PENDING_GRACE_SECONDS
    saw_pending = False

    while time.monotonic() <= deadline:
        if is_speaking():
            waited = time.monotonic() - started_at
            offset = _env_float("MOSS_REACHY_ACTION_SYNC_OFFSET", ACTION_SYNC_OFFSET_SECONDS)
            log.info(
                "[ReachyMiniSpeechSync] command=%s speech_started waited=%.2fs remaining=%.2fs offset=%.2fs",
                command_name,
                waited,
                remaining_seconds(),
                offset,
            )
            if offset > 0:
                await asyncio.sleep(offset)
            return True
        if is_speech_pending():
            saw_pending = True
        elif not saw_pending and time.monotonic() >= pending_deadline:
            waited = time.monotonic() - started_at
            log.info(
                "[ReachyMiniSpeechSync] command=%s no_speech_pending waited=%.2fs",
                command_name,
                waited,
            )
            return False
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    log.warning(
        "[ReachyMiniSpeechSync] command=%s speech_start_timeout timeout=%.1fs",
        command_name,
        timeout,
    )
    return False
