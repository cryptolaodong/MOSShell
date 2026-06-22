"""
基于能量的本地 VAD（Voice Activity Detection）实现。

状态机：
  idle → speech_detected → silence_detected → (触发 commit)

只有在检测到语音之后的持续静音才会触发，避免纯静音录音被误提交。
"""

import time
from typing import Optional

import numpy as np


class EnergyVAD:
    """基于 RMS 能量的静音检测器。

    用法：每个音频 chunk 调用一次 `check(audio_data)`，
    当检测到"说过话之后持续静音"时返回 True。
    """

    def __init__(
        self,
        *,
        silence_threshold: float = 400.0,
        speech_threshold: float = 600.0,
        silence_hold_time: float = 1.2,
    ):
        """
        Args:
            silence_threshold: RMS 低于此值视为静音
            speech_threshold: RMS 高于此值视为有语音活动
            silence_hold_time: 语音结束后静音持续多久触发（秒）
        """
        self._silence_threshold = silence_threshold
        self._speech_threshold = speech_threshold
        self._silence_hold_time = silence_hold_time

        self._speech_detected = False
        self._silence_start: Optional[float] = None

    def reset(self) -> None:
        """重置状态（开始新的录音段时调用）"""
        self._speech_detected = False
        self._silence_start = None

    def check(self, audio_data: np.ndarray) -> bool:
        """检查一个音频 chunk，返回 True 表示应结束录音。

        Args:
            audio_data: int16 PCM 音频数据

        Returns:
            True 如果检测到"语音后的持续静音"
        """
        rms = float(np.sqrt(np.mean(audio_data.astype(float) ** 2)))

        if rms >= self._speech_threshold:
            # 检测到语音
            self._speech_detected = True
            self._silence_start = None
            return False

        if not self._speech_detected:
            # 还没说过话，不触发
            return False

        if rms < self._silence_threshold:
            # 静音中
            now = time.monotonic()
            if self._silence_start is None:
                self._silence_start = now
                return False
            if now - self._silence_start >= self._silence_hold_time:
                return True
        else:
            # 介于阈值之间（轻微噪音），重置静音计时
            self._silence_start = None

        return False

    def __call__(self, audio_data: np.ndarray, vad_time: Optional[int] = None) -> bool:
        """兼容 VAD callable 签名：`Callable[[np.ndarray, Optional[int]], bool]`"""
        return self.check(audio_data)
        silence_threshold: float = 400.0,
