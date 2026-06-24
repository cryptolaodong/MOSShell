import wave

import numpy as np

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.file_player import AudioFilePlayer, load_wav_clip
from ghoshell_moss_contrib.moss_in_reachy_mini.audio.mixer import AudioMixer


class FakeSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def push_pcm(self, pcm: np.ndarray, *, sample_rate: int, channels: int) -> None:
        self.chunks.append(pcm.copy())

    def stop(self) -> None:
        pass

    def samples(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 1), dtype=np.int16)
        return np.concatenate(self.chunks, axis=0)


def test_load_wav_clip_reads_pcm16(tmp_path) -> None:
    path = tmp_path / "tone.wav"
    pcm = np.asarray([0, 100, -100, 200], dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(pcm.tobytes())

    clip = load_wav_clip(path)

    assert clip.name == "tone.wav"
    assert clip.sample_rate == 8000
    assert clip.channels == 1
    assert clip.pcm.tolist() == [0, 100, -100, 200]


def test_audio_file_player_enqueues_loaded_wav(tmp_path) -> None:
    path = tmp_path / "prompt.wav"
    pcm = np.asarray([10, 20, 30], dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(pcm.tobytes())

    sink = FakeSink()
    mixer = AudioMixer(sink, realtime=False)
    player = AudioFilePlayer(mixer)

    clip = player.play_file(path, name="prompt", clear_queue=True)

    assert clip.name == "prompt"
    assert mixer.wait_idle(timeout=1.0)
    assert sink.samples().reshape(-1).tolist() == [10, 20, 30]
    mixer.close()
