import threading
import time

import numpy as np

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.mixer import AudioClip, AudioMixer


class FakeSink:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.chunks: list[np.ndarray] = []
        self.names: list[tuple[int, int]] = []
        self.stop_calls = 0
        self.first_chunk = threading.Event()
        self.lock = threading.Lock()

    def push_pcm(self, pcm: np.ndarray, *, sample_rate: int, channels: int) -> None:
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.chunks.append(pcm.copy())
            self.names.append((sample_rate, channels))
            self.first_chunk.set()

    def stop(self) -> None:
        self.stop_calls += 1

    def samples(self) -> np.ndarray:
        with self.lock:
            if not self.chunks:
                return np.zeros((0, 1), dtype=np.int16)
            return np.concatenate(self.chunks, axis=0)


def _clip(name: str, values: list[int], *, sample_rate: int = 1000) -> AudioClip:
    return AudioClip(name=name, pcm=np.asarray(values, dtype=np.int16), sample_rate=sample_rate, channels=1)


def test_mixer_plays_queue_in_order() -> None:
    sink = FakeSink()
    mixer = AudioMixer(sink, chunk_ms=5, realtime=False)

    mixer.play_clip(_clip("a", [1, 2, 3]))
    mixer.play_clip(_clip("b", [4, 5]))

    assert mixer.wait_idle(timeout=1.0)
    assert sink.samples().reshape(-1).tolist() == [1, 2, 3, 4, 5]
    status = mixer.status()
    assert status.state == "idle"
    assert status.queued == 0
    mixer.close()


def test_mixer_applies_master_volume() -> None:
    sink = FakeSink()
    mixer = AudioMixer(sink, chunk_ms=10, realtime=False)
    mixer.set_volume(0.5)

    mixer.play_clip(_clip("quiet", [1000, -1000, 300]))

    assert mixer.wait_idle(timeout=1.0)
    assert sink.samples().reshape(-1).tolist() == [500, -500, 150]
    assert mixer.status().volume == 0.5
    mixer.close()


def test_mixer_pause_and_resume_holds_position() -> None:
    sink = FakeSink(delay=0.002)
    mixer = AudioMixer(sink, chunk_ms=10, realtime=True)
    mixer.play_clip(_clip("long", list(range(100)), sample_rate=1000))

    assert sink.first_chunk.wait(timeout=1.0)
    mixer.pause()
    paused_status = mixer.status()
    assert paused_status.state == "paused"
    time.sleep(0.03)
    paused_count = len(sink.chunks)
    time.sleep(0.03)
    assert len(sink.chunks) == paused_count

    mixer.resume()

    assert mixer.wait_idle(timeout=1.0)
    assert sink.samples().reshape(-1).tolist() == list(range(100))
    mixer.close()


def test_mixer_stop_clears_pending_queue() -> None:
    sink = FakeSink(delay=0.005)
    mixer = AudioMixer(sink, chunk_ms=10, realtime=True)
    mixer.play_clip(_clip("first", list(range(200)), sample_rate=1000))
    mixer.play_clip(_clip("second", [9, 9, 9], sample_rate=1000))

    assert sink.first_chunk.wait(timeout=1.0)
    mixer.stop()

    assert mixer.wait_idle(timeout=1.0)
    status = mixer.status()
    assert status.state == "stopped"
    assert status.queued == 0
    assert sink.stop_calls >= 1
    assert sink.samples().shape[0] < 203
    mixer.close()
