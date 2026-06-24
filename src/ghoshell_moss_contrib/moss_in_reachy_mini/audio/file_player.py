"""WAV file playback helpers built on the side-channel AudioMixer."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .mixer import AudioClip, AudioMixer, AudioPlaybackStatus


def load_wav_clip(path: str | Path, *, name: str | None = None) -> AudioClip:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_rate = int(wav.getframerate())
        sample_width = int(wav.getsampwidth())
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.int16, copy=True)
    elif sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        pcm = ((raw - 128) << 8).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    return AudioClip(
        name=name or wav_path.name,
        pcm=pcm,
        sample_rate=sample_rate,
        channels=channels,
    )


class AudioFilePlayer:
    def __init__(self, mixer: AudioMixer) -> None:
        self._mixer = mixer

    def play_file(self, path: str | Path, *, clear_queue: bool = False, name: str | None = None) -> AudioClip:
        clip = load_wav_clip(path, name=name)
        self._mixer.play_clip(clip, clear_queue=clear_queue)
        return clip

    def pause(self) -> None:
        self._mixer.pause()

    def resume(self) -> None:
        self._mixer.resume()

    def stop(self) -> None:
        self._mixer.stop()

    def clear(self) -> None:
        self._mixer.clear()

    def set_volume(self, volume: float) -> None:
        self._mixer.set_volume(volume)

    def status(self) -> AudioPlaybackStatus:
        return self._mixer.status()
