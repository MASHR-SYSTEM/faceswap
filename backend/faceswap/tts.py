from __future__ import annotations

from .lifecycle import WorkerLifecycle, serialized
from .playback import PlaybackWriter

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np

from .lipsync import LipSyncController
from .schemas import TTSSpeakRequest, TTSStatus, TTSVoiceInfo, VoiceConfig
from .speech import SpeechToSpeechProcessor
from .voice import (
    VIRTUAL_SINK_NAME,
    PacatWriter,
    PulseSinkMonitor,
    SounddeviceMonitor,
    _import_sounddevice,
    _pulse_sink_for_output_device,
    ensure_virtual_mic_route,
    get_virtual_mic_route_status,
    set_live_voice_ducking,
)


KOKORO_SAMPLE_RATE = 24_000
TTS_VOICES = [
    TTSVoiceInfo(name="af_heart", label="Heart", language="en-us"),
    TTSVoiceInfo(name="af_bella", label="Bella", language="en-us"),
    TTSVoiceInfo(name="af_nicole", label="Nicole", language="en-us"),
    TTSVoiceInfo(name="af_sarah", label="Sarah", language="en-us"),
    TTSVoiceInfo(name="am_adam", label="Adam", language="en-us"),
    TTSVoiceInfo(name="am_michael", label="Michael", language="en-us"),
    TTSVoiceInfo(name="bf_emma", label="Emma", language="en-gb"),
    TTSVoiceInfo(name="bm_george", label="George", language="en-gb"),
]


class TextToSpeechEngine(Protocol):
    name: str

    def generate(self, text: str, *, voice: str, speed: float) -> Iterator[tuple[int, np.ndarray]]:
        ...


@dataclass
class _TTSRuntimeState:
    phase: str = "idle"
    generation: int = 0
    running: bool = False
    active: bool = False
    ready: bool = False
    fallback: bool = False
    engine: str = "kokoro"
    voice: str = "af_heart"
    synthesis_target: str = "monk"
    synthesis_quality: str = "low_latency"
    synthesis_ready: bool = False
    synthesis_latency_ms: float = 0.0
    synthesis_queue_ms: float = 0.0
    synthesis_rt_factor: float = 0.0
    synthesis_fallback: bool = False
    synthesis_detail: str | None = None
    text_preview: str | None = None
    sample_rate: int = 48000
    generated_seconds: float = 0.0
    playback_seconds: float = 0.0
    audio_level: float = 0.0
    mouth_open: float = 0.0
    lip_sync_enabled: bool = True
    lip_delay_ms: int = 110
    mute_live_mic: bool = True
    virtual_mic: bool = True
    virtual_mic_ready: bool = False
    monitor_enabled: bool = False
    monitor_ready: bool = False
    monitor_output_device: int | None = None
    monitor_output_name: str | None = None
    last_error: str | None = None
    detail: str | None = None


class TTSSession(WorkerLifecycle):
    def __init__(self, engine: TextToSpeechEngine | None = None) -> None:
        self._init_lifecycle()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _TTSRuntimeState()
        self._engine = engine or KokoroTextToSpeechEngine()
        self._fallback_engine = FallbackSpeechEngine()
        self._lip_sync = LipSyncController()
        self._playback_started_at: float | None = None
        self._playback_clock = None

    @serialized
    def speak(self, request: TTSSpeakRequest) -> TTSStatus:
        text = request.text.strip()
        if not text:
            raise RuntimeError("Text speech needs non-empty text")
        if request.interrupt:
            self.stop()
        elif self._is_thread_alive():
            raise RuntimeError("Text speech is already playing")

        duck_live_voice = _should_duck_live_voice(request)
        with self._lock:
            self._require_stopped()
            self._playback_started_at = None
            self._playback_clock = None
            self._state = _TTSRuntimeState(
                running=True,
                active=True,
                ready=False,
                fallback=False,
                engine=f"{self._engine.name}+speech_to_speech" if request.synthesis_enabled else self._engine.name,
                voice=request.voice,
                synthesis_target=request.synthesis_target,
                synthesis_quality=request.synthesis_quality,
                text_preview=_preview_text(text),
                sample_rate=request.sample_rate,
                lip_sync_enabled=request.lip_sync_enabled,
                lip_delay_ms=request.lip_delay_ms,
                mute_live_mic=duck_live_voice,
                virtual_mic=request.virtual_mic,
                monitor_enabled=request.monitor_enabled,
                monitor_output_device=request.monitor_output_device,
                detail="Loading text-to-speech",
            )
            self._launch(request, "faceswap-tts")
            return self.status()

    @serialized
    def stop(self) -> TTSStatus:
        if self._stop_worker():
            with self._lock:
                self._state.active = False
            self._lip_sync.end()
        return self.status()

    def status(self) -> TTSStatus:
        with self._lock:
            data = self._state.__dict__.copy()
            started_at = self._playback_started_at
        if started_at is not None and data["active"]:
            data["playback_seconds"] = min(
                data["generated_seconds"],
                self._playback_clock() if self._playback_clock else max(0.0, time.monotonic() - started_at),
            )
        if not data["running"] and data["virtual_mic"]:
            route = get_virtual_mic_route_status()
            data["virtual_mic_ready"] = route.source_present
        data["mouth_open"] = self._lip_sync.value()
        return TTSStatus(**data)

    def voices(self) -> list[TTSVoiceInfo]:
        return TTS_VOICES

    def lip_value(self) -> float:
        return self._lip_sync.value()

    def _is_thread_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self, request: TTSSpeakRequest) -> None:
        writer: PlaybackWriter | None = None
        monitor: SounddeviceMonitor | PulseSinkMonitor | None = None
        synthesizer: SpeechToSpeechProcessor | None = None
        previous_ducking: float | None = None
        generated_samples = 0
        try:
            duck_live_voice = _should_duck_live_voice(request)
            route = ensure_virtual_mic_route() if request.virtual_mic else get_virtual_mic_route_status()
            if request.virtual_mic:
                writer = PlaybackWriter(
                    sample_rate=request.sample_rate,
                    channels=1,
                    sink_name=VIRTUAL_SINK_NAME,
                    queue_blocks=24,
                )
                writer.start()
            if request.monitor_enabled:
                monitor = _open_monitor(
                    sample_rate=request.sample_rate,
                    block_size=1024,
                    output_device=request.monitor_output_device,
                    volume=request.monitor_volume,
                )
            self._playback_clock = (writer.position_seconds if writer else
                                    monitor.position_seconds if monitor else None)
            if duck_live_voice:
                previous_ducking = set_live_voice_ducking(0.0)
            if request.synthesis_enabled:
                synthesizer = SpeechToSpeechProcessor(request.sample_rate, _voice_config_for_tts(request))

            self._lip_sync.begin(
                sample_rate=request.sample_rate,
                delay_ms=request.lip_delay_ms,
                amount=request.lip_amount,
                smoothing=request.lip_smoothing,
                enabled=request.lip_sync_enabled,
            )
            self._set_state(
                ready=True,
                virtual_mic_ready=route.source_present if request.virtual_mic else False,
                monitor_ready=monitor is not None,
                monitor_output_name=monitor.output_name if monitor is not None else None,
                detail=None,
            )
            if synthesizer is not None:
                self._update_synthesis_state(synthesizer)

            self._mark_ready()
            for source_rate, audio in self._audio_chunks(request):
                if self._stop_event.is_set():
                    break
                source_audio = _declick_audio_chunk(_mono_float32(audio), source_rate)
                chunk = _resample_audio(source_audio, source_rate, request.sample_rate)
                if chunk.size == 0:
                    continue
                chunk = np.clip(chunk, -0.98, 0.98).astype(np.float32, copy=False)
                if synthesizer is not None:
                    chunk = self._synthesize_chunk(
                        synthesizer,
                        chunk,
                        request.sample_rate,
                        request.synthesis_quality,
                    )
                    if chunk.size == 0:
                        continue
                for block in _audio_output_blocks(chunk, request.sample_rate):
                    if self._stop_event.is_set():
                        break
                    if self._playback_started_at is None:
                        with self._lock:
                            self._playback_started_at = time.monotonic()
                    self._lip_sync.append_audio(block)
                    if generated_samples == 0:
                        self._lip_sync.start_playback(writer.position_seconds if writer else
                            monitor.position_seconds if monitor else None)
                    generated_samples += int(block.size)
                    level = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if block.size else 0.0
                    if writer is not None:
                        self._write_virtual_mic_block(writer, block)
                    if monitor is not None:
                        self._write_monitor_block(monitor, block)
                    if writer is None and monitor is None:
                        self._sleep_visual_block(block, request.sample_rate)
                    if writer is not None and writer.failed:
                        raise RuntimeError(writer.last_error or "Text speech virtual mic stopped")
                    if monitor is not None and monitor.failed:
                        raise RuntimeError(monitor.last_error or "Text speech monitor stopped")
                    self._set_state(
                        generated_seconds=generated_samples / float(request.sample_rate),
                        audio_level=level,
                        mouth_open=self._lip_sync.value(),
                        last_error=None,
                    )
                    if synthesizer is not None:
                        self._update_synthesis_state(synthesizer)

            self._lip_sync.finish_audio()
            playback = writer or monitor
            if playback is not None:
                while not self._stop_event.wait(.01):
                    if playback.failed:
                        raise RuntimeError(playback.last_error or "Playback failed")
                    if playback.position_seconds() >= generated_samples / request.sample_rate:
                        break
            else:
                self._wait_for_playback_tail(generated_samples, request.sample_rate)
        except Exception as exc:
            self._set_state(last_error=str(exc), active=False, detail=str(exc))
        finally:
            if previous_ducking is not None:
                set_live_voice_ducking(previous_ducking)
            if monitor is not None:
                monitor.close()
            if writer is not None:
                writer.close()
            if synthesizer is not None:
                synthesizer.close()
            self._lip_sync.end()
            close_engine = getattr(self._engine, 'close', None)
            if close_engine is not None:
                close_engine()
            with self._lock:
                self._state.running = False
                self._state.active = False
                self._state.monitor_ready = False

    def _audio_chunks(self, request: TTSSpeakRequest) -> Iterator[tuple[int, np.ndarray]]:
        try:
            yield from self._engine.generate(request.text, voice=request.voice, speed=request.speed)
        except Exception as exc:
            detail = f"Kokoro unavailable: {exc}"
            engine = f"{self._fallback_engine.name}+speech_to_speech" if request.synthesis_enabled else self._fallback_engine.name
            self._set_state(fallback=True, engine=engine, detail=detail)
            yield from self._fallback_engine.generate(request.text, voice=request.voice, speed=request.speed)

    def _wait_for_playback_tail(self, generated_samples: int, sample_rate: int) -> None:
        started = self._playback_started_at
        if started is None:
            return
        generated_seconds = generated_samples / float(sample_rate)
        end_at = started + generated_seconds + 0.18
        while not self._stop_event.is_set() and time.monotonic() < end_at:
            self._set_state(
                playback_seconds=min(generated_seconds, max(0.0, time.monotonic() - started)),
                mouth_open=self._lip_sync.value(),
            )
            time.sleep(0.025)

    def _write_virtual_mic_block(self, writer: PacatWriter, block: np.ndarray) -> None:
        payload = block.astype(np.float32, copy=False).tobytes()
        while not self._stop_event.is_set():
            if writer.write(payload, timeout=0.25):
                return
            if writer.failed:
                raise RuntimeError(writer.last_error or "Text speech virtual mic stopped")

    def _write_monitor_block(self, monitor: SounddeviceMonitor | PulseSinkMonitor, block: np.ndarray) -> None:
        while not self._stop_event.is_set():
            if monitor.write(block):
                return
            if monitor.failed:
                raise RuntimeError(monitor.last_error or "Text speech monitor stopped")
            time.sleep(0.02)

    def _sleep_visual_block(self, block: np.ndarray, sample_rate: int) -> None:
        duration = min(0.15, block.size / float(sample_rate))
        end_at = time.monotonic() + duration
        while not self._stop_event.is_set() and time.monotonic() < end_at:
            time.sleep(min(0.02, end_at - time.monotonic()))

    def _synthesize_chunk(
        self,
        synthesizer: SpeechToSpeechProcessor,
        chunk: np.ndarray,
        sample_rate: int,
        quality: str,
    ) -> np.ndarray:
        output_blocks: list[np.ndarray] = []
        block_size = _synthesis_block_size(quality, sample_rate)
        for block in _synthesis_input_blocks(chunk, sample_rate, quality):
            if self._stop_event.is_set():
                break
            output, _level = synthesizer.process(block)
            output_blocks.append(output)
            self._update_synthesis_state(synthesizer)
        status = synthesizer.status()
        drain_seconds = min(2.0, max(0.8, (status.latency_ms + status.queue_ms) / 1000.0 + 0.25))
        drain_blocks = max(1, int(math.ceil(drain_seconds * sample_rate / float(block_size))))
        silence = np.zeros(block_size, dtype=np.float32)
        for _index in range(drain_blocks):
            if self._stop_event.is_set():
                break
            output, _level = synthesizer.process(silence)
            output_blocks.append(output)
            self._update_synthesis_state(synthesizer)
        if not output_blocks:
            return np.zeros(0, dtype=np.float32)
        output = np.concatenate(output_blocks).astype(np.float32, copy=False)
        return _trim_silence_edges(output)

    def _update_synthesis_state(self, synthesizer: SpeechToSpeechProcessor) -> None:
        status = synthesizer.status()
        self._set_state(
            synthesis_ready=bool(status.ready),
            synthesis_latency_ms=float(status.latency_ms),
            synthesis_queue_ms=float(status.queue_ms),
            synthesis_rt_factor=float(status.rt_factor),
            synthesis_fallback=bool(status.fallback),
            synthesis_detail=status.detail,
        )

    def _set_state(self, **updates: object) -> None:
        with self._lock:
            for key, value in updates.items():
                setattr(self._state, key, value)

    def _increment_drop_error(self, message: str) -> None:
        with self._lock:
            self._state.last_error = message


class KokoroTextToSpeechEngine:
    name = "kokoro"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pipelines: dict[str, Any] = {}

    def generate(self, text: str, *, voice: str, speed: float) -> Iterator[tuple[int, np.ndarray]]:
        pipeline = self._pipeline(_kokoro_lang_code(voice))
        for text_chunk in _split_text_for_tts(text):
            generator = pipeline(text_chunk, voice=voice, speed=speed, split_pattern=r"\n+")
            for _graphemes, _phonemes, audio in generator:
                yield KOKORO_SAMPLE_RATE, _mono_float32(audio)

    def close(self):
        with self._lock:
            self._pipelines.clear()
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pipeline(self, lang_code: str) -> Any:
        with self._lock:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is not None:
                return pipeline
            try:
                from kokoro import KPipeline
            except Exception as exc:
                raise RuntimeError("install Kokoro with scripts/install_tts_runtime.sh") from exc
            try:
                pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
            except TypeError:
                pipeline = KPipeline(lang_code=lang_code)
            self._pipelines[lang_code] = pipeline
            return pipeline


class FallbackSpeechEngine:
    name = "fallback"

    def generate(self, text: str, *, voice: str, speed: float) -> Iterator[tuple[int, np.ndarray]]:
        del voice
        sample_rate = KOKORO_SAMPLE_RATE
        for chunk in _text_to_fallback_chunks(text, sample_rate, speed):
            yield sample_rate, chunk


def kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
    except Exception:
        return False
    return True


def _should_duck_live_voice(request: TTSSpeakRequest) -> bool:
    return bool(request.mute_live_mic or request.virtual_mic or request.monitor_enabled)


def _voice_config_for_tts(request: TTSSpeakRequest) -> VoiceConfig:
    return VoiceConfig(
        enabled=True,
        preset="monk",
        synthesis_mode="speech_to_speech",
        synthesis_target=request.synthesis_target,
        synthesis_quality=request.synthesis_quality,
        input_device=None,
        virtual_mic=False,
        monitor_enabled=False,
        monitor_output_device=None,
        monitor_volume=request.monitor_volume,
        monitor_mode="processed",
        sample_rate=request.sample_rate,
        block_size=512 if request.synthesis_quality == "low_latency" else 1024,
        noise_gate=0.0,
        gain=1.0,
    )


def _open_monitor(
    *,
    sample_rate: int,
    block_size: int,
    output_device: int | None,
    volume: float,
) -> SounddeviceMonitor | PulseSinkMonitor:
    pulse_sink = _pulse_sink_for_output_device(output_device)
    if pulse_sink is not None:
        monitor = PulseSinkMonitor(
            sink_name=pulse_sink.name,
            output_name=pulse_sink.description,
            sample_rate=sample_rate,
            volume=volume,
        )
        monitor.start()
        return monitor

    sounddevice = _import_sounddevice()
    monitor = SounddeviceMonitor(
        sample_rate=sample_rate,
        block_size=block_size,
        output_device=output_device,
        volume=volume,
        queue_blocks=24,
    )
    monitor.start(sounddevice)
    return monitor


def _mono_float32(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        if samples.shape[0] <= 2 and samples.shape[1] > samples.shape[0]:
            samples = np.mean(samples, axis=0, dtype=np.float32)
        else:
            samples = np.mean(samples, axis=1, dtype=np.float32)
    return samples.reshape(-1).astype(np.float32, copy=False)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy import signal

        divisor = math.gcd(source_rate, target_rate)
        return signal.resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32, copy=False)
    except Exception:
        duration = audio.size / float(source_rate)
        target_size = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, duration, audio.size, endpoint=False)
        target_positions = np.linspace(0.0, duration, target_size, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _declick_audio_chunk(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if samples.size == 0:
        return samples
    samples -= float(np.mean(samples, dtype=np.float64))
    fade_samples = min(samples.size // 3, max(1, int(round(sample_rate * 0.006))))
    if fade_samples > 1:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        samples[:fade_samples] *= fade
        samples[-fade_samples:] *= fade[::-1]
    return samples.astype(np.float32, copy=False)


def _kokoro_lang_code(voice: str) -> str:
    if voice.startswith("b"):
        return "b"
    return "a"


def _preview_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:117] + "..." if len(collapsed) > 120 else collapsed


def _split_text_for_tts(text: str, max_chars: int = 650) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text.strip()]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        for sentence in sentences or [paragraph]:
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current = f"{current} {sentence}"
            else:
                chunks.extend(_hard_wrap_text(current, max_chars))
                current = sentence
        if current and len(current) >= max_chars * 0.7:
            chunks.extend(_hard_wrap_text(current, max_chars))
            current = ""
    if current:
        chunks.extend(_hard_wrap_text(current, max_chars))
    return chunks


def _hard_wrap_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _audio_output_blocks(audio: np.ndarray, sample_rate: int) -> Iterator[np.ndarray]:
    block_size = max(1, int(round(sample_rate * 0.06)))
    for offset in range(0, audio.size, block_size):
        yield audio[offset : offset + block_size]


def _synthesis_input_blocks(audio: np.ndarray, sample_rate: int, quality: str) -> Iterator[np.ndarray]:
    block_size = _synthesis_block_size(quality, sample_rate)
    for offset in range(0, audio.size, block_size):
        block = audio[offset : offset + block_size]
        if block.size:
            yield block


def _synthesis_block_size(quality: str, sample_rate: int) -> int:
    seconds = {"low_latency": 0.18, "balanced": 0.24, "quality": 0.32}.get(quality, 0.18)
    return max(1, int(round(sample_rate * seconds)))


def _trim_silence_edges(audio: np.ndarray, *, threshold: float = 0.0015, keep_samples: int = 480) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    active = np.flatnonzero(np.abs(samples) > threshold)
    if active.size == 0:
        return samples
    start = max(0, int(active[0]) - keep_samples)
    stop = min(samples.size, int(active[-1]) + keep_samples)
    return samples[start:stop].astype(np.float32, copy=False)


def _text_to_fallback_chunks(text: str, sample_rate: int, speed: float) -> Iterator[np.ndarray]:
    words = re.findall(r"[\w']+|[.,!?;:]", text)
    if not words:
        words = ["text"]
    chunk: list[np.ndarray] = []
    chunk_samples = 0
    max_chunk_samples = int(sample_rate * 0.75)
    for index, token in enumerate(words):
        audio = _fallback_token_audio(token, index, sample_rate, speed)
        chunk.append(audio)
        chunk_samples += audio.size
        if chunk_samples >= max_chunk_samples:
            yield np.concatenate(chunk).astype(np.float32, copy=False)
            chunk = []
            chunk_samples = 0
    if chunk:
        yield np.concatenate(chunk).astype(np.float32, copy=False)


def _fallback_token_audio(token: str, index: int, sample_rate: int, speed: float) -> np.ndarray:
    is_punctuation = bool(re.fullmatch(r"[.,!?;:]", token))
    duration = 0.12 if is_punctuation else float(np.clip((0.13 + len(token) * 0.018) / speed, 0.12, 0.42))
    sample_count = max(1, int(round(sample_rate * duration)))
    if is_punctuation:
        return np.zeros(sample_count, dtype=np.float32)

    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    base = 118.0 + (index % 5) * 19.0
    carrier = 0.12 * np.sin(2.0 * np.pi * base * t)
    formant = 0.035 * np.sin(2.0 * np.pi * (base * 2.35) * t)
    envelope_base = np.clip(np.sin(np.linspace(0.0, np.pi, sample_count, dtype=np.float32)), 0.0, 1.0)
    envelope = envelope_base**0.65
    gap = np.zeros(max(1, int(round(sample_rate * 0.035 / speed))), dtype=np.float32)
    return np.concatenate([(carrier + formant) * envelope, gap]).astype(np.float32, copy=False)
