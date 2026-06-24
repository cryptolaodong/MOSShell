import asyncio
import logging
import os
import time

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx
from reachy_mini import ReachyMini

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import (
    should_sync_interpreter_command,
    wait_for_speech_start,
)
from ghoshell_moss_contrib.moss_in_reachy_mini.state.abcd import BaseReachyState


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class BoringState(BaseReachyState):
    NAME = "boring"
    DESCRIPTION = "无聊状态：电机保持使能，等待用户交互，长时间无交互将自动进入休眠。"

    def __init__(self, mini: ReachyMini, logger: LoggerItf = None):
        self._mini = mini
        self.logger = logger or logging.getLogger("BoringState")

    async def on_startup(self):
        self.logger.info("BoringState.on_startup Enter")
        if should_sync_interpreter_command("switch_state"):
            await wait_for_speech_start("switch_state", self.logger)
        self._mini.enable_motors()

    async def on_close(self):
        pass

    async def on_idle(self):
        self.logger.info("BoringState.on_idle Enter")
        try:
            auto_sleep_seconds = _env_float("MOSS_REACHY_BORING_AUTO_SLEEP_SECONDS", 0.0)
            if auto_sleep_seconds <= 0:
                self.logger.info("BoringState auto sleep disabled; keeping motors enabled")
                await asyncio.Future()
                return
            start = time.time()
            while time.time() - start < auto_sleep_seconds:
                await asyncio.sleep(0.1)
            self.logger.info("BoringState idle %.1fs, switching to asleep", auto_sleep_seconds)
            runtime = ChannelCtx.runtime()
            if runtime is not None:
                await runtime.execute_command("switch_state", kwargs={"name": "asleep"})
            else:
                self.logger.warning("BoringState: ChannelCtx.runtime() returned None, cannot switch to asleep")
        except asyncio.CancelledError:
            self.logger.info("BoringState.on_idle cancelled")
        self.logger.info("BoringState.on_idle Exit")
