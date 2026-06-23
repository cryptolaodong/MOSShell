import logging

import numpy as np
from ghoshell_common.contracts import LoggerItf
from numpy import typing as npt
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import time_trajectory

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import wait_for_speech_start


class AntennasMove(Move):

    def __init__(
            self,
            current_left: float,
            current_right: float,
            target_left: float = 0,
            target_right: float = 0,
            duration: float = 2.0,
    ):
        self.current_left = current_left
        self.current_right = current_right
        self.target_left = target_left
        self.target_right = target_right
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        interp_time = time_trajectory(t / self.duration)

        r_rad = np.deg2rad(self.current_right + (self.target_right - self.current_right) * interp_time)
        l_rad = np.deg2rad(self.current_left + (self.target_left - self.current_left) * interp_time)

        return None, np.array([r_rad, l_rad]), None

class Antennas:
    def __init__(self, mini: ReachyMini, logger: LoggerItf = None):
        self.mini = mini
        self.logger = logger or logging.getLogger("Antennas")


    async def move(self, left: float=0, right: float=0, duration: float=2.0):
        """
        Move the antenna to the given position and duration.

        For synchronized speech and motion, place this CTML command before or
        inside the sentence it should accompany, not after the whole sentence.

        Args:
            left (float): X coordinate of the position. range(degree): [-180, +180]
            right (float): Y coordinate of the position. range(degree): [-180, +180]
            duration (float): Duration of the movement.
        """
        r_degree, l_degree = self._get_current_position()
        await wait_for_speech_start("antennas_move", self.logger)
        await self.mini.async_play_move(AntennasMove(
            current_left=l_degree,
            current_right=r_degree,
            target_left=left,
            target_right=right,
            duration=duration,
        ))

    async def reset(self, duration: float=2.0):
        """
        Reset the antenna to zero in duration.
        """
        r_degree, l_degree = self._get_current_position()
        await wait_for_speech_start("antennas_reset", self.logger)
        await self.mini.async_play_move(AntennasMove(
            current_left=l_degree,
            current_right=r_degree,
            duration=duration,
        ))

    def _get_current_position(self):
        r_rad, l_rad = self.mini.get_present_antenna_joint_positions()
        r_degree, l_degree = round(np.rad2deg(r_rad), 1), round(np.rad2deg(l_rad), 1)
        return r_degree, l_degree
