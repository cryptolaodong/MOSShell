from typing import Tuple
import os
import time

import numpy as np
from numpy import typing as npt
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import time_trajectory, linear_pose_interpolation

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speaking_gate import (
    is_speaking,
    is_speech_pending,
    is_thinking,
)


_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class HeadMove(Move):
    def __init__(
            self,
            start_pose: npt.NDArray[np.float64],
            target_pose: npt.NDArray[np.float64],
            start_body_yaw: float=None,
            target_body_yaw: float=None,
            duration: float = 0.5,
    ):
        self.start_pose = start_pose
        self.target_pose = target_pose
        self._duration = duration

        self.start_body_yaw = start_body_yaw
        self.target_body_yaw = (
            target_body_yaw if target_body_yaw is not None else start_body_yaw
        )

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        interp_time = time_trajectory(t / self.duration)
        interp_head_pose = linear_pose_interpolation(
            self.start_pose, self.target_pose, interp_time
        )
        interp_body_yaw_joint = None
        if self.start_body_yaw is not None:
            interp_body_yaw_joint = (
                self.start_body_yaw
                + (self.target_body_yaw - self.start_body_yaw) * interp_time
            )
        return interp_head_pose, None, interp_body_yaw_joint


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: npt.NDArray[np.float64],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral positions for breathing base
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])

        # Breathing parameters
        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5  # Hz (faster antenna sway)

        # While waiting for LLM/TTS to start, make the idle motion visibly alive.
        self.thinking_enabled = (
            os.environ.get("MOSS_REACHY_THINKING_MOTION_ENABLED", "1").strip().lower()
            not in _FALSE_VALUES
        )
        self.thinking_yaw_amplitude = _env_float("MOSS_REACHY_THINKING_HEAD_YAW", 7.0)
        self.thinking_pitch_amplitude = _env_float("MOSS_REACHY_THINKING_HEAD_PITCH", 3.0)
        self.thinking_antenna_amplitude = np.deg2rad(
            _env_float("MOSS_REACHY_THINKING_ANTENNA", 8.0)
        )
        self.thinking_gate_check_interval = _env_float(
            "MOSS_REACHY_THINKING_GATE_CHECK_INTERVAL",
            0.2,
            minimum=0.05,
        )
        self._thinking_active = False
        self._thinking_started_t = 0.0
        self._last_thinking_check = 0.0

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(self, t: float) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            thinking_blend = self._thinking_blend(t)
            thinking_time = max(0.0, t - self._thinking_started_t)
            yaw = (
                thinking_blend
                * self.thinking_yaw_amplitude
                * np.sin(2 * np.pi * 0.22 * thinking_time)
            )
            pitch = (
                thinking_blend
                * self.thinking_pitch_amplitude
                * np.sin((2 * np.pi * 0.16 * thinking_time) + 0.7)
            )
            head_pose = create_head_pose(
                x=0,
                y=0,
                z=z_offset,
                roll=0,
                pitch=pitch,
                yaw=yaw,
                degrees=True,
                mm=False,
            )

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antenna_sway += (
                thinking_blend
                * self.thinking_antenna_amplitude
                * np.sin((2 * np.pi * 0.75 * thinking_time) + 0.5)
            )
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return head_pose, antennas, 0.0

    def _thinking_blend(self, t: float) -> float:
        if not self.thinking_enabled:
            return 0.0

        now = time.monotonic()
        if now - self._last_thinking_check >= self.thinking_gate_check_interval:
            self._last_thinking_check = now
            active = is_thinking() and not is_speech_pending() and not is_speaking(tail=0.1)
            if active and not self._thinking_active:
                self._thinking_started_t = t
            self._thinking_active = active

        if not self._thinking_active:
            return 0.0
        return min(1.0, max(0.0, (t - self._thinking_started_t) / 0.5))
