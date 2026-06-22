import logging
import os
from typing import Literal, Optional, Dict

import numpy as np
# pyaudio imported lazily
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss.contracts import ConfigType
from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss_contrib.asr.concepts import ListenerStateName
# PyAudioInput imported lazily
from ghoshell_moss_contrib.asr.volcengine_bm_protocol import VolcanoBigModelASRConfig


class PyAudioInputConfig(BaseModel):
    """
    使用 pyaudio input 的配置项.
    """
    device_index: str = Field(default="$PYAUDIO_INPUT_DEVICE_INDEX", description="音频输入设备的 pyaudio 设备号. ")
    rate: str = Field("$PYAUDIO_INPUT_DEVICE_RATE", description="pyaudio 音频输入使用的采样率")
    channels: int = Field(1, description="音频输入 channels")
    read_interval: float = Field(0.128, description="读取音频片段的帧时长, 单位是秒")
    chunk_size: int = Field(2048, description="默认的块大小")

    def resolve_env(self) -> Self:
        if self.device_index.startswith('$'):
            self.device_index = os.environ.get(self.device_index[1:], '')
        if self.rate.startswith('$'):
            self.rate = os.environ.get(self.rate[1:], '44100')

        # Optional env override for channel count.
        # Some capture devices expose only stereo input; others only mono.
        channels_env = os.environ.get("PYAUDIO_INPUT_DEVICE_CHANNELS")
        if channels_env:
            try:
                self.channels = int(channels_env)
            except Exception:
                pass
        return self

    def get_device_index(self) -> Optional[int]:
        if self.device_index:
            return int(self.device_index)
        return None

    def new_audio_input(
            self,
            *,
            pa = None,
            logger: Optional[LoggerItf] = None,
            dtype: np.dtype = np.int16,
    ):
        """
        快速创建一个.
        """
        import pyaudio
        from ghoshell_moss_contrib.asr.pyaudio_input_impl import PyAudioInput
        pa = pa or pyaudio.PyAudio()
        conf = self.resolve_env()
        logger = logger or logging.getLogger("PyAudioInput")

        device_index = self.get_device_index()
        rate = int(conf.rate)
        channels = int(conf.channels)

        # Validate device capabilities to avoid opaque PortAudio errors.
        if device_index is not None:
            try:
                info = pa.get_device_info_by_index(device_index)
                max_in = int(info.get("maxInputChannels") or 0)
                if max_in <= 0:
                    logger.warning(
                        "PyAudioInput device_index=%s has no input channels; falling back to default input device",
                        device_index,
                    )
                    device_index = None
                elif channels > max_in:
                    logger.warning(
                        "PyAudioInput channels=%s exceeds device_index=%s maxInputChannels=%s; clamping",
                        channels,
                        device_index,
                        max_in,
                    )
                    channels = max_in
            except Exception as e:
                logger.warning(
                    "PyAudioInput failed to read device info for device_index=%s: %s",
                    device_index,
                    e,
                )

        channels = max(1, int(channels))
        return PyAudioInput(
            pa,
            rate=rate,
            device_index=device_index,
            channels=channels,
            logger=logger,
            dtype=dtype,
            read_interval=conf.read_interval,
            chunk_size=conf.chunk_size,
        )


class ListenerConfig(ConfigType):
    """
    音频输入 (v2) 的配置项.
    """

    @classmethod
    def conf_name(cls) -> str:
        return "audio/listener.yml"

    use_audio_input: str = Field(
        default="USE_AUDIO_INPUT_CONFIG",
        description="通过环境变量选择 audio_input 配置"
    )
    audio_input: Dict[str, PyAudioInputConfig] = Field(
        default_factory=lambda: dict(default=PyAudioInputConfig()),
        description="所有可用的 audio_input 配置项"
    )

    state_loop_interval: float = Field(
        default=0.1,
        description="检查 state 变更的频率",
    )

    use_asr: str = Field(
        default="volcengine_bm_asr",
        json_schema_extra=dict(
            enum={"volcengine_bm_asr", "volcengine_stream_asr"},
        ),
    )
    use_vad: Literal[""] = Field(default="", description="使用的 vad 类型. 目前没有")
    use_waken: Literal[""] = Field(default="", description="使用的唤醒词模型, 目前没有")

    asr_on_vad_state: str = Field(
        default="",
        description="当 vad 发生时, 是否要指定变更的状态."
    )
    asr_max_idle_time: float = Field(
        default=20,
        description="多长空闲时间重置一次 asr 状态."
    )
    asr_on_idle_state: str = Field(
        default="",
        description="如果 asr 闲置了太长时间, 是否要切换成别的状态. "
    )

    default_state_name: Literal["listening", "deaf", "asleep"] = Field(
        default=ListenerStateName.pdt_waiting.value,
        description="开启后默认的状态",
    )

    volcengine_bm_asr: VolcanoBigModelASRConfig = Field(
        default_factory=VolcanoBigModelASRConfig,
    )

    def resolve_env(self) -> Self:
        config = self.model_copy(deep=True)
        # 解析 use_audio_input 环境变量
        if config.use_audio_input.startswith('$'):
            config.use_audio_input = os.environ.get(
                config.use_audio_input[1:], "default")
        # 递归解析 audio_input 下每个 profile 的 env
        for k, v in config.audio_input.items():
            if isinstance(v, dict):
                config.audio_input[k] = PyAudioInputConfig(**v).resolve_env()
            else:
                config.audio_input[k] = v.resolve_env()
        config.volcengine_bm_asr = config.volcengine_bm_asr.resolve_env()
        return config

    def get_audio_input_config(self) -> PyAudioInputConfig:
        # 返回当前选中的 profile 的配置
        return self.audio_input.get(self.use_audio_input, self.audio_input["default"])
