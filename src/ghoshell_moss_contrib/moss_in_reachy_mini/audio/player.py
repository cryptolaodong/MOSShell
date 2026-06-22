import time

import numpy as np
from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.contracts.speech import AudioFormat
from ghoshell_moss.core.speech.base_player import BaseAudioStreamPlayer
from reachy_mini import ReachyMini


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
            mini = ReachyMini(host=robot_host, media_backend='no_media', connection_mode='network')
            from ghoshell_moss_contrib.moss_in_reachy_mini.audio.upload_player import ReachyMiniUploadAudioPlayer
            logger.info('[ReachyMiniAudioPlayer] Using upload-based audio player (WebSocket)')
            return ReachyMiniUploadAudioPlayer(mini, sample_rate=24000, channels=1, logger=logger)
        except Exception as e:
            logger.warning('[ReachyMiniAudioPlayer] Upload player failed: %s', e)

        # Fallback: try WebRTC streaming (requires functional GStreamer + WebRTC signaling)
        try:
            tmp = ReachyMini(host=robot_host, media_backend='no_media')
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
        return MiniAudioStreamPlayer(sample_rate=24000, channels=1, logger=logger, safety_delay=0.5)
