import asyncio
import logging
import time

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.concepts.command import PyCommand
from reachy_mini import ReachyMini

from ghoshell_moss_contrib.moss_in_reachy_mini.components.antennas import Antennas
from ghoshell_moss_contrib.moss_in_reachy_mini.components.body import Body
from ghoshell_moss_contrib.moss_in_reachy_mini.components.head import Head
from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import (
    should_sync_interpreter_command,
    wait_for_speech_start,
)
from ghoshell_moss_contrib.moss_in_reachy_mini.state.abcd import BaseReachyState


class WakenState(BaseReachyState):
    NAME = "waken"
    DESCRIPTION = "唤醒状态：电机使能，头部追踪活跃，所有交互命令可用。"

    def __init__(
        self,
        mini: ReachyMini,
        body: Body,
        head: Head,
        antennas: Antennas,
        logger: LoggerItf = None,
    ):
        self._mini = mini
        self._body = body
        self._head = head
        self._antennas = antennas
        self.logger = logger or logging.getLogger("WakenState")

        self._own_commands = {
            "dance": PyCommand(
                self._body.dance,
                name="dance",
                doc=self._body.dance_docstring,
                blocking=True,
            ),
            "emotion": PyCommand(
                self._body.emotion,
                name="emotion",
                doc=self._body.emotion_docstring,
                blocking=True,
            ),
            "head_move": PyCommand(
                self._head.move,
                name="head_move",
                blocking=True,
            ),
            "head_reset": PyCommand(
                self._head.reset,
                name="head_reset",
                blocking=True,
            ),
            "antennas_move": PyCommand(
                self._antennas.move,
                name="antennas_move",
                blocking=True,
            ),
            "antennas_reset": PyCommand(
                self._antennas.reset,
                name="antennas_reset",
                blocking=True,
            ),
        }

    async def on_startup(self):
        if should_sync_interpreter_command("switch_state"):
            await wait_for_speech_start("switch_state", self.logger)
        self._mini.enable_motors()
        self._mini.wake_up()
        self._head.switch_idle_mode("breathing")

    async def on_close(self):
        pass

    async def on_idle(self):
        self.logger.info("WakenState.on_idle Enter")
        idle_start = time.time()
        head_task = None
        try:
            head_task = asyncio.create_task(self._head.on_idle())
            # 每 1s 检查是否 300s 无交互
            while not head_task.done():
                await asyncio.wait([head_task], timeout=1.0)
                if time.time() - idle_start > 300:
                    self.logger.info("WakenState idle 300s, switching to boring")
                    head_task.cancel()
                    try:
                        await head_task
                    except asyncio.CancelledError:
                        pass
                    runtime = ChannelCtx.runtime()
                    if runtime is not None:
                        await runtime.execute_command("switch_state", kwargs={"name": "boring"})
                    return
        except asyncio.CancelledError:
            raise
        finally:
            if head_task is not None and not head_task.done():
                head_task.cancel()
        self.logger.info("WakenState.on_idle Exit")

    def own_commands(self) -> dict[str, PyCommand]:
        return self._own_commands
