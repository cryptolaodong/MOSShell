from typing import Iterable, Literal, Type, Optional

from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.configs import ConfigType, ConfigStore
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer
from pydantic import Field

__all__ = ["AudioPlayerProvider", "AudioPlayerConfig"]


class AudioPlayerConfig(ConfigType):
    backend: Literal["miniaudio", "pyaudio", "reachy_mini"] = Field(
        default="miniaudio",
        description="Audio player backend. 'miniaudio' is zero-dependency default; 'pyaudio' requires `pip install ghoshell_moss[audio]`.",
    )
    samplerate: int = Field(
        default=44100,
        description="Sample rate of audio player stream",
    )
    safety_delay: float = Field(
        default=0.1,
        description="Delay for time calculation after player finishes a stream",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "audio_player"


class AudioPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return False

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        store = con.force_fetch(ConfigStore)
        conf = store.get_or_create(AudioPlayerConfig())
        logger = con.force_fetch(LoggerItf)

        if conf.backend == "miniaudio":
            return MiniAudioStreamPlayer(
                sample_rate=conf.samplerate,
                channels=1,
                logger=logger,
                safety_delay=conf.safety_delay,
            )

        if conf.backend == "reachy_mini":
            from ghoshell_moss_contrib.moss_in_reachy_mini.audio.player import ReachyMiniStreamPlayer
            from reachy_mini import ReachyMini
            import os, time
            robot_host = os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local")
            # Step 1: connect without media to acquire it
            try:
                tmp = ReachyMini(host=robot_host, media_backend="no_media")
                if tmp.media_released:
                    tmp.acquire_media()
                    logger.info("[AudioPlayer] Media acquired, waiting for WebRTC...")
                tmp.__exit__(None, None, None)
                time.sleep(5)  # Wait for WebRTC producer to register
            except Exception as e:
                logger.warning("[AudioPlayer] acquire_media failed: %s", e)
            # Step 2: connect with media
            try:
                mini = ReachyMini(host=robot_host)
                logger.info("[AudioPlayer] Connected with media: sr=%s", mini.media.get_output_audio_samplerate())
                return ReachyMiniStreamPlayer(
                    mini,
                    logger=logger,
                    safety_delay=conf.safety_delay,
                )
            except Exception as e:
                logger.warning("[AudioPlayer] reachy_mini media failed: %s, falling back to miniaudio", e)
                return MiniAudioStreamPlayer(
                    sample_rate=conf.samplerate,
                    channels=1,
                    logger=logger,
                    safety_delay=conf.safety_delay,
                )

        if conf.backend == "pyaudio":
            try:
                from ghoshell_moss.host.speech.player.pyaudio_player import PyAudioStreamPlayer
            except ImportError:
                raise ImportError(
                    "PyAudio backend selected but not installed. "
                    "Run: pip install ghoshell_moss[audio]"
                )
            return PyAudioStreamPlayer(
                device_index=0,
                sample_rate=conf.samplerate,
                channels=1,
                logger=logger,
                safety_delay=conf.safety_delay,
            )

        raise ValueError(f"Unknown audio backend: {conf.backend}")
