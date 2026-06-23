import asyncio
import enum
import logging

from ghoshell_common.contracts import LoggerItf
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import wait_for_speech_start
from ghoshell_moss_contrib.moss_in_reachy_mini.moves.head_move import HeadMove, BreathingMove


class IdleMode(enum.Enum):
    hold = "hold"
    breathing = "breathing"


class Head:
    def __init__(
            self,
            mini: ReachyMini,
            logger: LoggerItf=None,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger("Head")

        self._idle_mode: str = IdleMode.hold.value
        self._last_idle_mode: str = IdleMode.hold.value  # 记录上一个空闲模式，方便恢复

        self._current_body_yaw = 0.0

    async def move(
            self,
            x: float = 0,
            y: float = 0,
            z: float = 0,
            roll: float = 0,
            pitch: float = 0,
            yaw: float = 0,
            body_yaw: float = 0,
            duration: float = 0.5,
    ):
        """Move to a pose in 6D space (position and orientation).

        For synchronized speech and motion, place this CTML command before or
        inside the sentence it should accompany, not after the whole sentence.
        Example:
        <apps.bodies_reachymini:head_move yaw="10" duration="0.6"/>我现在动动脑袋。

        Args:
            x (float): X coordinate of the position. range: [-1.5cm, +2.5cm]
            y (float): Y coordinate of the position. range: [-4cm, +4cm]
            z (float): Z coordinate of the position. range: [-4cm, +2.5cm]
            roll (float): Roll angle. range(degree): [-40, +40]
            pitch (float): Pitch angle. range(degree): [-40, +40]
            yaw (float): Yaw angle. range(degree)，正数是左转头，负数是右转头: [-60, +60]
            body_yaw (float): Body yaw angle. range(degree): [-155, +155]
            duration (float): Duration in seconds.
        """
        await wait_for_speech_start("head_move", self.logger)
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(x, y, z, roll, pitch, yaw),
            self._current_body_yaw,
            body_yaw,
            duration=duration
        ))

        self._current_body_yaw = body_yaw

    def switch_idle_mode(self, mode: str):
        self._last_idle_mode = self._idle_mode
        self._idle_mode = mode

    async def reset(self, idle_mode: str, duration: float = 0.5):
        """
        Reset the head, watching forward

        :param idle_mode: 重置后保持的空闲模式，可选："hold"（空闲时静止）、"breathing"（空闲时呼吸，推荐）
        :param duration: 重置时间，单位秒
        """
        self.switch_idle_mode(idle_mode)
        await wait_for_speech_start("head_reset", self.logger)
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
            self._current_body_yaw,
            0.0,
            duration=duration,
        ))
        self._current_body_yaw = 0.0

    async def _breathing(self):
        while True:
            try:
                _, current_antennas = self.mini.get_current_joint_positions()
                current_head_pose = self.mini.get_current_head_pose()
                breathing_move = BreathingMove(
                    interpolation_start_pose=current_head_pose,
                    interpolation_start_antennas=current_antennas,
                    interpolation_duration=1.0,
                )
                await self.mini.async_play_move(breathing_move)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("Head breathing move failed; retrying")
                await asyncio.sleep(1.0)

    async def on_idle(self):
        self.logger.info("Head on-idle entering")
        try:
            if self._idle_mode == IdleMode.hold.value:
                await asyncio.Future()  # 挂起一个空事件，等待被cancel
            if self._idle_mode == IdleMode.breathing.value:
                await self._breathing()
        except asyncio.CancelledError:
            self.logger.info("Head on_idle task cancelled successfully")
