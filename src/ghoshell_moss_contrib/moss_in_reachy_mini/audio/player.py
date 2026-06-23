import time

import numpy as np
from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.contracts.speech import AudioFormat
from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer
from reachy_mini import ReachyMini

from .speaking_gate import clear_speaking, mark_speech_pending, mark_speaking_for


def _connect_without_releasing_media(**kwargs) -> ReachyMini:
    """Create a no-media SDK client without releasing daemon-owned mic/camera."""
    original_release_media = ReachyMini.release_media
    ReachyMini.release_media = lambda self: None
    try:
        return ReachyMini(**kwargs)
    finally:
        ReachyMini.release_media = original_release_media


class ReachyMiniStreamPlayer(BaseAudioStreamPlayer):
    def __init__(
        self,
        mini: ReachyMini,
        *,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.5,
    ):
        self.mini = mini
        super().__init__(
            sample_rate=self.mini.media.get_output_audio_samplerate(),
            channels=self.mini.media.get_output_channels(),
            logger=logger,
            safety_delay=safety_delay,
        )

    def _sync_output_format(self) -> None:
        audio = getattr(self.mini.media, "audio", None)
        stream = None if audio is None else getattr(audio, "_output_stream", None)
        if stream is not None:
            try:
                sr = getattr(stream, "samplerate", None)
                if sr is not None:
                    self.sample_rate = int(sr)
            except Exception:
                pass

        try:
            ch = int(self.mini.media.get_output_channels())
            if ch > 0:
                self.channels = ch
        except Exception:
            pass

    def add(
        self,
        chunk: np.ndarray,
        *,
        audio_type: AudioFormat,
        rate: int,
        channels: int = 1,
    ) -> float:
        if self._closed:
            return time.time()

        if audio_type == AudioFormat.PCM_F32LE:
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            audio_data = chunk.astype(np.int16)

        audio_data = self.resample(audio_data, origin_rate=rate, target_rate=self.sample_rate)

        audio_f32 = audio_data.astype(np.float32) / 32768.0
        # Volume at 1x (natural level)
        audio_f32 = np.clip(audio_f32 * 1.0, -1.0, 1.0)
        audio_data = np.column_stack((audio_f32, audio_f32))

        duration = audio_data.shape[0] / self.sample_rate

        self._audio_queue.put_nowait(audio_data)
        self._play_done_event.clear()

        current_time = time.time()
        if current_time > self._estimated_end_time:
            self._estimated_end_time = current_time + duration
        else:
            self._estimated_end_time += duration
        return self._estimated_end_time

    def _audio_stream_start(self):
        pass

    def _audio_stream_stop(self):
        return

    def _audio_stream_write(self, data: np.ndarray):
        self.mini.media.push_audio_sample(data)


def _attach_speaking_gate_to_fallback_player(player: BaseAudioStreamPlayer, logger: LoggerItf) -> None:
    original_clear = player.clear

    def _on_play(audio_data: np.ndarray) -> None:
        try:
            now = time.time()
            estimated_left = max(0.0, getattr(player, "_estimated_end_time", now) - now)
            samples = int(getattr(audio_data, "shape", [len(audio_data)])[0])
            duration = samples / max(1, int(getattr(player, "sample_rate", 24000) or 24000))
            mark_speaking_for(max(duration, estimated_left), tail=1.5)
        except Exception as e:
            logger.debug("[ReachyMiniAudioPlayer] fallback speaking gate mark failed: %s", e)

    async def _finish_stream() -> None:
        mark_speaking_for(0.0, tail=1.5)

    async def _clear() -> None:
        clear_speaking()
        await original_clear()

    player.on_play(_on_play)
    player.mark_pending = mark_speech_pending  # type: ignore[attr-defined]
    player.finish_stream = _finish_stream  # type: ignore[attr-defined]
    player.clear = _clear  # type: ignore[method-assign]


from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.contracts.speech import StreamAudioPlayer


class ReachyMiniStreamPlayerProvider(Provider[StreamAudioPlayer]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        import os, time
        logger = con.force_fetch(LoggerItf)
        robot_host = os.environ.get('REACHY_ROBOT_HOST', 'reachy-mini.local')

        # Primary: upload-based player (WebSocket, no GStreamer needed, proven reliable)
        try:
            mini = _connect_without_releasing_media(
                host=robot_host,
                media_backend='no_media',
                connection_mode='network',
            )
            from ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player import ReachyMiniUploadAudioPlayer
            logger.info('[ReachyMiniAudioPlayer] Using upload-based audio player (WebSocket)')
            return ReachyMiniUploadAudioPlayer(mini, sample_rate=24000, channels=1, logger=logger)
        except Exception as e:
            logger.warning('[ReachyMiniAudioPlayer] Upload player failed: %s', e)

        # Fallback: try WebRTC streaming (requires functional GStreamer + WebRTC signaling)
        try:
            tmp = _connect_without_releasing_media(host=robot_host, media_backend='no_media')
            if tmp.media_released:
                tmp.acquire_media()
                logger.info('[ReachyMiniAudioPlayer] Media acquired, waiting for WebRTC...')
            tmp.__exit__(None, None, None)
            time.sleep(5)
            mini = ReachyMini(host=robot_host)
            sr = mini.media.get_output_audio_samplerate()
            if sr > 0:
                logger.info('[ReachyMiniAudioPlayer] Fallback to WebRTC media: sr=%s', sr)
                return ReachyMiniStreamPlayer(mini, logger=logger)
            mini.__exit__(None, None, None)
        except Exception as e:
            logger.warning('[ReachyMiniAudioPlayer] WebRTC fallback failed: %s', e)

        # Last resort: local miniaudio
        logger.error('[ReachyMiniAudioPlayer] All robot audio methods failed. Using local miniaudio.')
        from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer
        player = MiniAudioStreamPlayer(sample_rate=24000, channels=1, logger=logger, safety_delay=0.5)
        _attach_speaking_gate_to_fallback_player(player, logger)
        return player
