import asyncio
import logging
import time

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx
from reachy_mini import ReachyMini

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import (
    should_sync_interpreter_command,
    wait_for_speech_start,
)
from ghoshell_moss_contrib.moss_in_reachy_mini.state.abcd import BaseReachyState


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
            start = time.time()
            while time.time() - start < 30:
                await asyncio.sleep(0.1)
            self.logger.info("BoringState idle 30s, switching to asleep")
            runtime = ChannelCtx.runtime()
            if runtime is not None:
                await runtime.execute_command("switch_state", kwargs={"name": "asleep"})
            else:
                self.logger.warning("BoringState: ChannelCtx.runtime() returned None, cannot switch to asleep")
        except asyncio.CancelledError:
            self.logger.info("BoringState.on_idle cancelled")
        self.logger.info("BoringState.on_idle Exit")
