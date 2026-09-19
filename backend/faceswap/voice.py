from __future__ import annotations

import platform

from .lifecycle import WorkerLifecycle, serialized
from .playback import PlaybackWriter

import ctypes.util
import os
import queue
import select
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import AudioDeviceInfo, VoiceConfig, VoiceRouteStatus, VoiceStatus
from .speech import SpeechToSpeechProcessor


VIRTUAL_SINK_NAME = "faceswap_voice_sink"
VIRTUAL_SOURCE_NAME = "faceswap_voice_mic"
VIRTUAL_SINK_DESCRIPTION = "FaceSwap_Voice_Sink"
VIRTUAL_SOURCE_DESCRIPTION = "FaceSwap_Voice_Mic"
PULSE_SOURCE_INDEX_BASE = -100_000
PULSE_SINK_INDEX_BASE = -200_000
_LIVE_VOICE_DUCKING_LOCK = threading.RLock()
_LIVE_VOICE_DUCKING_GAIN = 1.0


def set_live_voice_ducking(gain: float) -> float:
    global _LIVE_VOICE_DUCKING_GAIN
    with _LIVE_VOICE_DUCKING_LOCK:
        previous = _LIVE_VOICE_DUCKING_GAIN
        _LIVE_VOICE_DUCKING_GAIN = float(np.clip(gain, 0.0, 1.0))
        return previous


def live_voice_ducking_gain() -> float:
    with _LIVE_VOICE_DUCKING_LOCK:
        return _LIVE_VOICE_DUCKING_GAIN


@dataclass(frozen=True)
class _PulseSource:
    index: int
    name: str
    description: str
    channels: int
    sample_rate: float


@dataclass(frozen=True)
class _PulseSink:
    index: int
    name: str
    description: str
    channels: int
    sample_rate: float


@dataclass(frozen=True)
class _PulseSourceOutput:
    index: int
    source: int | None
    target_object: str | None
    application_name: str | None
    media_name: str | None


@dataclass
class _VoiceRuntimeState:
    phase: str = "idle"
    generation: int = 0
    running: bool = False
    enabled: bool = False
    preset: str = "monk"
    synthesis_mode: str = "dsp"
    synthesis_target: str = "monk"
    synthesis_quality: str = "low_latency"
    synthesis_ready: bool = False
    synthesis_latency_ms: float = 0.0
    synthesis_queue_ms: float = 0.0
    synthesis_rt_factor: float = 0.0
    synthesis_fallback: bool = False
    synthesis_detail: str | None = None
    sample_rate: int = 48000
    block_size: int = 512
    input_device: int | None = None
    input_backend: str | None = None
    input_name: str | None = None
    virtual_mic: bool = True
    virtual_mic_ready: bool = False
    virtual_sink_name: str = VIRTUAL_SINK_NAME
    virtual_source_name: str = VIRTUAL_SOURCE_NAME
    monitor_enabled: bool = False
    monitor_ready: bool = False
    monitor_output_device: int | None = None
    monitor_output_name: str | None = None
    monitor_volume: float = 0.7
    monitor_mode: str = "processed"
    frames_processed: int = 0
    audio_level: float = 0.0
    audio_meter_level: float = 0.0
    audio_peak_level: float = 0.0
    callback_ms: float = 0.0
    dropped_blocks: int = 0
    monitor_dropped_blocks: int = 0
    last_error: str | None = None


class VoiceSession(WorkerLifecycle):
    def __init__(self) -> None:
        self._init_lifecycle()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _VoiceRuntimeState()
        self._active_config = None

    @serialized
    def start(self, config: VoiceConfig) -> VoiceStatus:
        if not config.enabled:
            return self.stop()

        self.stop()
        with self._lock:
            self._require_stopped()
            self._active_config = config.model_copy(deep=True)
            self._state = _VoiceRuntimeState(
                running=True,
                enabled=config.enabled,
                preset=config.preset,
                synthesis_mode=config.synthesis_mode,
                synthesis_target=config.synthesis_target,
                synthesis_quality=config.synthesis_quality,
                sample_rate=config.sample_rate,
                block_size=config.block_size,
                input_device=config.input_device,
                virtual_mic=config.virtual_mic,
                monitor_enabled=config.monitor_enabled,
                monitor_output_device=config.monitor_output_device,
                monitor_volume=config.monitor_volume,
                monitor_mode=config.monitor_mode,
            )
            self._launch(config, "faceswap-voice")
            return self.status()

    @serialized
    def stop(self) -> VoiceStatus:
        self._stop_worker()
        return self.status()

    def status(self) -> VoiceStatus:
        with self._lock:
            data = self._state.__dict__.copy()
            data["active_config"] = self._active_config
        if not data["running"] and data["virtual_mic"]:
            route = get_virtual_mic_route_status()
            data["virtual_mic_ready"] = route.source_present
            data["virtual_sink_name"] = route.sink_name
            data["virtual_source_name"] = route.source_name
        return VoiceStatus(**data)

    def _run(self, config: VoiceConfig) -> None:
        writer: PacatWriter | None = None
        monitor: SounddeviceMonitor | PulseSinkMonitor | None = None
        pulse_reader: PulseSourceReader | None = None
        route_repair_stop = threading.Event()
        route_repair_thread: threading.Thread | None = None
        stream: Any | None = None
        processor: Any | None = None
        try:
            windows = platform.system() == "Windows"
            if windows and config.synthesis_mode != "dsp":
                raise RuntimeError("Windows currently supports DSP voice only")
            pulse_source = _pulse_source_for_input_device(config.input_device)
            pulse_sink = _pulse_sink_for_output_device(config.monitor_output_device)
            needs_sounddevice = pulse_source is None or (config.monitor_enabled and pulse_sink is None)
            sounddevice = _import_sounddevice() if needs_sounddevice else None
            route = ensure_virtual_mic_route() if config.virtual_mic else get_virtual_mic_route_status()
            if config.virtual_mic and not windows:
                repair_virtual_mic_capture_routes()
                route_repair_thread = threading.Thread(
                    target=_repair_virtual_mic_routes_worker,
                    args=(route_repair_stop,),
                    name="faceswap-voice-route-repair",
                    daemon=True,
                )
                route_repair_thread.start()
            if windows and config.virtual_mic:
                writer = WindowsCableWriter(config)
                writer.start()
            elif not windows and (config.virtual_mic or not config.monitor_enabled):
                writer = PacatWriter(sample_rate=config.sample_rate, channels=1,
                                     sink_name=VIRTUAL_SINK_NAME if config.virtual_mic else None, queue_blocks=12)
                writer.start()
            if config.monitor_enabled:
                if pulse_sink is not None:
                    monitor = PulseSinkMonitor(
                        sink_name=pulse_sink.name,
                        output_name=pulse_sink.description,
                        sample_rate=config.sample_rate,
                        volume=config.monitor_volume,
                    )
                    monitor.start()
                else:
                    if sounddevice is None:
                        sounddevice = _import_sounddevice()
                    monitor = SounddeviceMonitor(
                        sample_rate=config.sample_rate,
                        block_size=config.block_size,
                        output_device=config.monitor_output_device,
                        volume=config.monitor_volume,
                        queue_blocks=12,
                    )
                    monitor.start(sounddevice)
            if config.synthesis_mode == "speech_to_speech":
                processor = SpeechToSpeechProcessor(config.sample_rate, config)
            else:
                processor = MonkVoiceProcessor(config.sample_rate, config)

            with self._lock:
                self._state.input_backend = "pulse" if pulse_source is not None else "portaudio"
                self._state.input_name = pulse_source.description if pulse_source is not None else _sounddevice_input_name(sounddevice, config.input_device)
                self._state.virtual_mic_ready = route.source_present if config.virtual_mic else False
                self._state.virtual_sink_name = route.sink_name
                self._state.virtual_source_name = route.source_name
                self._state.monitor_ready = monitor is not None
                self._state.monitor_output_name = monitor.output_name if monitor is not None else None
                self._update_synthesis_state_locked(processor)

            self._mark_ready()

            def process_block(indata: np.ndarray, status: object | None = None) -> None:
                start = time.perf_counter()
                if status:
                    self._set_error(str(status))
                input_block = np.asarray(indata, dtype=np.float32)
                input_mono = _mono_samples(input_block)
                input_peak = float(np.max(np.abs(input_mono))) if input_mono.size else 0.0
                output, level = processor.process(input_block)
                ducking_gain = live_voice_ducking_gain()
                if ducking_gain < 1.0:
                    output = (output * ducking_gain).astype(np.float32, copy=False)
                wrote = True
                monitor_wrote = True
                if writer is not None:
                    wrote = writer.write(output.astype(np.float32, copy=False).tobytes())
                if monitor is not None:
                    monitor_signal = input_mono if config.monitor_mode == "input" else output
                    monitor_wrote = monitor.write(monitor_signal)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                with self._lock:
                    self._state.frames_processed += int(output.shape[0])
                    self._state.audio_level = float(level)
                    self._state.audio_meter_level = _smooth_meter_level(self._state.audio_meter_level, float(level))
                    self._state.audio_peak_level = _smooth_peak_level(self._state.audio_peak_level, input_peak)
                    self._state.callback_ms = float(elapsed_ms)
                    self._update_synthesis_state_locked(processor)
                    if not wrote:
                        self._state.dropped_blocks += 1
                    if not monitor_wrote:
                        self._state.monitor_dropped_blocks += 1

            if pulse_source is not None:
                pulse_reader = PulseSourceReader(
                    source_name=pulse_source.name,
                    sample_rate=config.sample_rate,
                    block_size=config.block_size,
                )
                pulse_reader.start()
                block_seconds = config.block_size / float(config.sample_rate)
                while not self._stop_event.is_set():
                    block = pulse_reader.read_block(timeout=block_seconds)
                    if block is not None:
                        process_block(block)
                    if writer is not None and writer.failed:
                        raise RuntimeError(writer.last_error or "PulseAudio writer stopped")
                    if monitor is not None and monitor.failed:
                        raise RuntimeError(monitor.last_error or "Headphone monitor stopped")
                    if pulse_reader.failed:
                        raise RuntimeError(pulse_reader.last_error or "Pulse source reader stopped")
            else:
                assert sounddevice is not None

                input_queue = queue.Queue(maxsize=3)
                input_drops = [0]
                def callback(indata, _frames, _time_info, status) -> None:
                    # The real-time callback copies only; DSP and output run on the owner worker.
                    item = (np.asarray(indata, dtype=np.float32).copy(), status)
                    try:
                        input_queue.put_nowait(item)
                    except queue.Full:
                        input_drops[0] += 1
                        try:
                            input_queue.get_nowait()
                            input_queue.put_nowait(item)
                        except (queue.Empty, queue.Full):
                            pass

                stream = sounddevice.InputStream(
                    samplerate=config.sample_rate,
                    blocksize=config.block_size,
                    channels=1,
                    dtype="float32",
                    device=config.input_device,
                    callback=callback,
                )
                with stream:
                    counted_drops = 0
                    while not self._stop_event.is_set():
                        try:
                            block, callback_status = input_queue.get(timeout=.1)
                            process_block(block, callback_status)
                            with self._lock:
                                self._state.dropped_blocks += input_drops[0] - counted_drops
                                counted_drops = input_drops[0]
                        except queue.Empty:
                            pass
                        if writer is not None and writer.failed:
                            raise RuntimeError(writer.last_error or "PulseAudio writer stopped")
                        if monitor is not None and monitor.failed:
                            raise RuntimeError(monitor.last_error or "Headphone monitor stopped")
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            if writer is not None:
                writer.close()
            if monitor is not None:
                monitor.close()
            if pulse_reader is not None:
                pulse_reader.close()
            if processor is not None:
                close = getattr(processor, "close", None)
                if close is not None:
                    close()
            route_repair_stop.set()
            if route_repair_thread is not None and route_repair_thread.is_alive():
                route_repair_thread.join()
            with self._lock:
                self._state.running = False
                self._state.monitor_ready = False

    def _set_error(self, error: str) -> None:
        with self._lock:
            self._state.last_error = error

    def _update_synthesis_state_locked(self, processor: Any) -> None:
        status_fn = getattr(processor, "status", None)
        if status_fn is None:
            self._state.synthesis_ready = False
            self._state.synthesis_latency_ms = 0.0
            self._state.synthesis_queue_ms = 0.0
            self._state.synthesis_rt_factor = 0.0
            self._state.synthesis_fallback = False
            self._state.synthesis_detail = None
            return
        status = status_fn()
        self._state.synthesis_ready = bool(status.ready)
        self._state.synthesis_latency_ms = float(status.latency_ms)
        self._state.synthesis_queue_ms = float(status.queue_ms)
        self._state.synthesis_rt_factor = float(status.rt_factor)
        self._state.synthesis_fallback = bool(status.fallback)
        self._state.synthesis_detail = status.detail


class MonkVoiceProcessor:
    def __init__(self, sample_rate: int, config: VoiceConfig) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self._privacy_mode = config.preset == "monk"
        self._lp_state = np.array([0.0], dtype=np.float32)
        self._hp_state = np.array([0.0], dtype=np.float32)
        self._pitch_min_delay = max(16, int(round(sample_rate * 0.018)))
        self._pitch_grain = max(64, int(round(sample_rate * 0.048)))
        self._pitch_ring = np.zeros(self._pitch_min_delay + self._pitch_grain + 4, dtype=np.float32)
        self._pitch_write = 0
        self._pitch_phase = 0.0
        self._privacy_sample_index = 0
        self._privacy_noise = np.random.default_rng()
        self._privacy_comb_a = _DelayLine(sample_rate, delay_seconds=0.011, max_seconds=0.04, feedback=0.18)
        self._privacy_comb_b = _DelayLine(sample_rate, delay_seconds=0.017, max_seconds=0.04, feedback=0.12)
        self._echo = _DelayLine(sample_rate, delay_seconds=0.18, max_seconds=0.5, feedback=0.26)
        self._reverbs = [
            _DelayLine(sample_rate, delay_seconds=0.029, max_seconds=0.08, feedback=0.48),
            _DelayLine(sample_rate, delay_seconds=0.037, max_seconds=0.08, feedback=0.42),
            _DelayLine(sample_rate, delay_seconds=0.053, max_seconds=0.08, feedback=0.36),
        ]

    def process(self, block: np.ndarray) -> tuple[np.ndarray, float]:
        if block.ndim == 2:
            mono = np.mean(block, axis=1, dtype=np.float32)
        else:
            mono = block.astype(np.float32, copy=False)

        level = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        if level < self.config.noise_gate:
            mono = mono * 0.08

        if self._privacy_mode:
            output = self._process_privacy_monk(mono, level)
            return np.clip(output, -0.98, 0.98).astype(np.float32), level

        shifted = self._pitch_shift_block(mono, self.config.pitch_semitones)
        body = self._lowpass(shifted, self.config.lowpass_hz)
        echo = self._echo.process(body) * self.config.echo
        reverb = sum(line.process(body) for line in self._reverbs) * (self.config.reverb / len(self._reverbs))

        wet = body + echo + reverb
        drive = 1.2 + self.config.depth * 1.5
        wet = np.tanh(wet * drive) / np.tanh(drive)
        output = mono * (1.0 - self.config.depth) + wet * self.config.depth
        output = output * self.config.gain
        return np.clip(output, -0.98, 0.98).astype(np.float32), level

    def _process_privacy_monk(self, mono: np.ndarray, level: float) -> np.ndarray:
        pitch_semitones = min(self.config.pitch_semitones, -7.0)
        shifted = self._pitch_shift_block(mono, pitch_semitones)
        body = self._highpass(shifted, 140.0)
        body = self._lowpass(body, min(self.config.lowpass_hz, 2400.0))

        comb_a = self._privacy_comb_a.process(body)
        comb_b = self._privacy_comb_b.process(body)
        body = body * 0.78 + comb_a * 0.18 - comb_b * 0.12
        body = self._privacy_modulate(body)

        echo_amount = max(self.config.echo, 0.2)
        reverb_amount = max(self.config.reverb, 0.42)
        echo = self._echo.process(body) * echo_amount
        reverb = sum(line.process(body) for line in self._reverbs) * (reverb_amount / len(self._reverbs))

        wet = body + echo + reverb
        wet = self._privacy_noise_floor(wet, level)
        drive = 2.7
        wet = np.tanh(wet * drive) / np.tanh(drive)
        return wet * self.config.gain

    def _pitch_shift_block(self, block: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return block.copy()

        factor = float(np.clip(2.0 ** (semitones / 12.0), 0.25, 2.0))
        phase_step = (1.0 - factor) / float(self._pitch_grain)
        output = np.empty_like(block, dtype=np.float32)
        for index, sample in enumerate(block.astype(np.float32, copy=False)):
            self._pitch_ring[self._pitch_write] = sample
            self._pitch_write = (self._pitch_write + 1) % len(self._pitch_ring)

            phase_a = self._pitch_phase % 1.0
            phase_b = (phase_a + 0.5) % 1.0
            delay_a = self._pitch_min_delay + phase_a * self._pitch_grain
            delay_b = self._pitch_min_delay + phase_b * self._pitch_grain
            window_a = 0.5 - 0.5 * np.cos(phase_a * 2.0 * np.pi)
            window_b = 0.5 - 0.5 * np.cos(phase_b * 2.0 * np.pi)
            output[index] = float(
                (
                    self._read_pitch_delay(delay_a) * window_a
                    + self._read_pitch_delay(delay_b) * window_b
                )
                / max(window_a + window_b, 1e-6)
            )

            self._pitch_phase = (self._pitch_phase + phase_step) % 1.0
        return output

    def _read_pitch_delay(self, delay_samples: float) -> float:
        read_pos = (self._pitch_write - delay_samples) % len(self._pitch_ring)
        left = int(np.floor(read_pos))
        right = (left + 1) % len(self._pitch_ring)
        fraction = read_pos - left
        return float(self._pitch_ring[left] * (1.0 - fraction) + self._pitch_ring[right] * fraction)

    def _highpass(self, block: np.ndarray, cutoff_hz: float) -> np.ndarray:
        try:
            from scipy import signal
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("scipy is required for voice filtering") from exc

        cutoff = float(np.clip(cutoff_hz, 20.0, self.sample_rate * 0.45))
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff)
        alpha = rc / (rc + dt)
        filtered, zf = signal.lfilter([alpha, -alpha], [1.0, -alpha], block, zi=self._hp_state)
        self._hp_state = np.asarray(zf, dtype=np.float32)
        return filtered.astype(np.float32, copy=False)

    def _privacy_modulate(self, block: np.ndarray) -> np.ndarray:
        indices = np.arange(len(block), dtype=np.float32) + float(self._privacy_sample_index)
        self._privacy_sample_index += len(block)
        slow = np.sin(2.0 * np.pi * 17.0 * indices / self.sample_rate)
        fast = np.sin(2.0 * np.pi * 43.0 * indices / self.sample_rate)
        modulation = 0.9 + 0.08 * slow + 0.035 * fast
        return (block * modulation.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    def _privacy_noise_floor(self, block: np.ndarray, level: float) -> np.ndarray:
        if level < self.config.noise_gate:
            return block
        noise_level = float(np.clip(level * 0.075, 0.0015, 0.012))
        noise = self._privacy_noise.normal(0.0, noise_level, size=block.shape).astype(np.float32)
        return block + noise

    def _lowpass(self, block: np.ndarray, cutoff_hz: float) -> np.ndarray:
        try:
            from scipy import signal
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("scipy is required for voice filtering") from exc

        cutoff = float(np.clip(cutoff_hz, 300.0, self.sample_rate * 0.45))
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff)
        alpha = dt / (rc + dt)
        filtered, zf = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], block, zi=self._lp_state)
        self._lp_state = np.asarray(zf, dtype=np.float32)
        return filtered.astype(np.float32, copy=False)


class _DelayLine:
    def __init__(self, sample_rate: int, *, delay_seconds: float, max_seconds: float, feedback: float) -> None:
        self._buffer = np.zeros(max(1, int(round(sample_rate * max_seconds))), dtype=np.float32)
        self._delay = max(1, min(len(self._buffer) - 1, int(round(sample_rate * delay_seconds))))
        self._feedback = float(np.clip(feedback, 0.0, 0.95))
        self._write = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        output = np.zeros_like(block, dtype=np.float32)
        for index, sample in enumerate(block):
            read = (self._write - self._delay) % len(self._buffer)
            delayed = self._buffer[read]
            output[index] = delayed
            self._buffer[self._write] = sample + delayed * self._feedback
            self._write = (self._write + 1) % len(self._buffer)
        return output


class PacatWriter:
    def __init__(self, *, sample_rate: int, channels: int, sink_name: str | None, queue_blocks: int) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._sink_name = sink_name
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_blocks)
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.failed = False
        self.last_error: str | None = None

    def start(self) -> None:
        pacat = shutil.which("pacat")
        if pacat is None:
            raise RuntimeError("pacat is not installed; install pulseaudio-utils or pipewire-pulse tools")
        command = [
            pacat,
            "--playback",
            f"--rate={self._sample_rate}",
            f"--channels={self._channels}",
            "--format=float32le",
            "--latency-msec=35",
        ]
        if self._sink_name:
            command.append(f"--device={self._sink_name}")
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, name="faceswap-voice-writer", daemon=True)
        self._thread.start()

    def write(self, payload: bytes, timeout: float | None = 0.0) -> bool:
        try:
            if timeout is None:
                self._queue.put(payload)
            elif timeout <= 0.0:
                self._queue.put_nowait(payload)
            else:
                self._queue.put(payload, timeout=timeout)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self._process is not None:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()

    def _run(self) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            try:
                self._process.stdin.write(payload)
                self._process.stdin.flush()
            except Exception as exc:
                self.failed = True
                self.last_error = f"Could not write to PulseAudio stream: {exc}"
                break


class SounddeviceMonitor:
    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        output_device: int | None,
        volume: float,
        queue_blocks: int,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._output_device = output_device
        self._volume = float(np.clip(volume, 0.0, 1.0))
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_blocks)
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._channels = 1
        self._output_name: str | None = None
        self._clock_lock = threading.Lock()
        self._submitted_seconds = 0.0
        self._observed_at = 0.0
        self._played_seconds = 0.0
        self.failed = False
        self.last_error: str | None = None

    @property
    def output_name(self) -> str | None:
        return self._output_name

    def start(self, sounddevice: Any) -> None:
        info = _sounddevice_output_info(sounddevice, self._output_device)
        max_channels = int(info.get("max_output_channels", 0))
        if max_channels <= 0:
            raise RuntimeError("Selected monitor output has no playback channels")
        self._channels = 2 if max_channels >= 2 else 1
        self._output_name = str(info.get("name") or "Default output")
        self._stream = sounddevice.OutputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=self._channels,
            dtype="float32",
            device=self._output_device,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._run, name="faceswap-voice-monitor", daemon=True)
        self._thread.start()

    def write(self, mono: np.ndarray) -> bool:
        try:
            self._queue.put_nowait(np.asarray(mono, dtype=np.float32).reshape(-1).copy())
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self._stream is not None:
            stop = getattr(self._stream, "stop", None)
            close = getattr(self._stream, "close", None)
            if stop is not None:
                stop()
            if close is not None:
                close()

    def _run(self) -> None:
        assert self._stream is not None
        while True:
            mono = self._queue.get()
            if mono is None:
                break
            try:
                self._stream.write(self._format_output(mono))
                with self._clock_lock:
                    self._submitted_seconds += len(mono) / self._sample_rate
                    self._played_seconds = self._submitted_seconds - float(self._stream.latency)
                    self._observed_at = time.monotonic()
            except Exception as exc:
                self.failed = True
                self.last_error = f"Could not write to headphone monitor: {exc}"
                break

    def position_seconds(self) -> float:
        with self._clock_lock:
            return max(0.0, min(self._submitted_seconds,
                self._played_seconds + time.monotonic() - self._observed_at))

    def _format_output(self, mono: np.ndarray) -> np.ndarray:
        payload = np.clip(mono * self._volume, -0.98, 0.98).astype(np.float32, copy=False)
        if self._channels == 1:
            return payload[:, None]
        return np.repeat(payload[:, None], self._channels, axis=1)


class PulseSourceReader:
    def __init__(self, *, source_name: str, sample_rate: int, block_size: int) -> None:
        self._source_name = source_name
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._process: subprocess.Popen[bytes] | None = None
        self._buffer = bytearray()
        self.failed = False
        self.last_error: str | None = None

    def start(self) -> None:
        parec = shutil.which("parec") or shutil.which("pacat")
        if parec is None:
            raise RuntimeError("parec is not installed; install pulseaudio-utils or pipewire-pulse tools")
        command = [
            parec,
            "--record",
            "--raw",
            f"--device={self._source_name}",
            f"--rate={self._sample_rate}",
            "--channels=1",
            "--format=float32le",
            "--latency-msec=35",
        ]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_block(self, *, timeout: float) -> np.ndarray | None:
        if self._process is None or self._process.stdout is None:
            self.failed = True
            self.last_error = "Pulse source reader is not running"
            return None
        if self._process.poll() is not None:
            self.failed = True
            self.last_error = self._stderr() or f"Pulse source reader stopped for {self._source_name}"
            return None

        block_bytes = self._block_size * 4
        if not self._fill_buffer(block_bytes, timeout):
            return None

        payload = bytes(self._buffer[:block_bytes])
        self._buffer = self._buffer[block_bytes:]
        return np.frombuffer(payload, dtype=np.float32).copy()

    def _fill_buffer(self, byte_count: int, timeout: float) -> bool:
        assert self._process is not None
        assert self._process.stdout is not None
        while len(self._buffer) < byte_count:
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
            if not ready:
                return False
            chunk = os.read(self._process.stdout.fileno(), byte_count - len(self._buffer))
            if not chunk:
                self.failed = True
                self.last_error = self._stderr() or f"Pulse source reader produced no audio for {self._source_name}"
                return False
            self._buffer.extend(chunk)
        return True

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _stderr(self) -> str | None:
        if self._process is None or self._process.stderr is None:
            return None
        try:
            data = self._process.stderr.read()
        except Exception:
            return None
        text = data.decode("utf-8", errors="replace").strip()
        return text or None


class PulseSinkMonitor:
    def __init__(self, *, sink_name: str, output_name: str, sample_rate: int, volume: float) -> None:
        self._sink_name = sink_name
        self._output_name = output_name
        self._sample_rate = sample_rate
        self._volume = float(np.clip(volume, 0.0, 1.0))
        self._writer: PacatWriter | None = None

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def failed(self) -> bool:
        return self._writer.failed if self._writer is not None else False

    @property
    def last_error(self) -> str | None:
        return self._writer.last_error if self._writer is not None else None

    def start(self) -> None:
        self._writer = PlaybackWriter(sample_rate=self._sample_rate, channels=1, sink_name=self._sink_name, queue_blocks=12)
        self._writer.start()

    def write(self, mono: np.ndarray) -> bool:
        if self._writer is None:
            return False
        payload = np.clip(np.asarray(mono, dtype=np.float32).reshape(-1) * self._volume, -0.98, 0.98)
        return self._writer.write(payload.astype(np.float32, copy=False).tobytes())

    def position_seconds(self) -> float:
        return self._writer.position_seconds() if self._writer else 0.0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def play_monitor_test_tone(
    *,
    output_device: int | None,
    volume: float,
    duration_ms: int = 650,
    frequency_hz: float = 880.0,
    sample_rate: int | None = None,
    sounddevice: Any | None = None,
) -> dict[str, object]:
    pulse_sink = _pulse_sink_for_output_device(output_device)
    if pulse_sink is not None:
        rate = int(sample_rate or round(pulse_sink.sample_rate))
        _play_pulse_test_tone(
            sink_name=pulse_sink.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            frequency_hz=frequency_hz,
            volume=volume,
        )
        return {
            "ok": True,
            "output_name": pulse_sink.description,
            "duration_ms": duration_ms,
            "sample_rate": rate,
        }

    sounddevice = sounddevice or _import_sounddevice()
    info = _sounddevice_output_info(sounddevice, output_device)
    max_channels = int(info.get("max_output_channels", 0))
    if max_channels <= 0:
        raise RuntimeError("Selected monitor output has no playback channels")

    output_name = str(info.get("name") or "Default output")
    rate = int(sample_rate or round(float(info.get("default_samplerate") or 48_000)))
    channels = 2 if max_channels >= 2 else 1
    tone = _test_tone(rate, duration_ms, frequency_hz, volume)
    payload = tone[:, None] if channels == 1 else np.repeat(tone[:, None], channels, axis=1)

    stream = sounddevice.OutputStream(
        samplerate=rate,
        blocksize=min(1024, max(64, len(tone))),
        channels=channels,
        dtype="float32",
        device=output_device,
    )
    try:
        stream.start()
        stream.write(payload)
        stream.stop()
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()

    return {
        "ok": True,
        "output_name": output_name,
        "duration_ms": duration_ms,
        "sample_rate": rate,
    }


def _play_pulse_test_tone(
    *,
    sink_name: str,
    sample_rate: int,
    duration_ms: int,
    frequency_hz: float,
    volume: float,
) -> None:
    pacat = shutil.which("pacat")
    if pacat is None:
        raise RuntimeError("pacat is not installed; install pulseaudio-utils or pipewire-pulse tools")

    tone = _test_tone(sample_rate, duration_ms, frequency_hz, volume)
    command = [
        pacat,
        "--playback",
        "--raw",
        f"--device={sink_name}",
        f"--rate={sample_rate}",
        "--channels=1",
        "--format=float32le",
        "--latency-msec=35",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, stderr = process.communicate(tone.astype(np.float32, copy=False).tobytes(), timeout=max(2.0, duration_ms / 1000.0 + 1.0))
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise RuntimeError("Pulse monitor test tone timed out") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "Pulse monitor test tone failed")


def _test_tone(sample_rate: int, duration_ms: int, frequency_hz: float, volume: float) -> np.ndarray:
    sample_count = max(1, int(round(sample_rate * duration_ms / 1000.0)))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    amplitude = 0.28 * float(np.clip(volume, 0.0, 1.0))
    tone = amplitude * np.sin(2.0 * np.pi * float(frequency_hz) * t)

    fade_samples = min(sample_count // 2, max(1, int(round(sample_rate * 0.015))))
    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    tone[:fade_samples] *= fade
    tone[-fade_samples:] *= fade[::-1]
    return tone.astype(np.float32, copy=False)


def _mono_samples(block: np.ndarray) -> np.ndarray:
    if block.ndim == 2:
        return np.mean(block, axis=1, dtype=np.float32)
    return block.astype(np.float32, copy=False).reshape(-1)


def _smooth_meter_level(previous: float, current: float) -> float:
    alpha = 0.45 if current > previous else 0.08
    return float(previous * (1.0 - alpha) + current * alpha)


def _smooth_peak_level(previous: float, current: float) -> float:
    if current >= previous:
        return float(current)
    return float(max(current, previous * 0.9))


def list_audio_devices() -> list[AudioDeviceInfo]:
    result: list[AudioDeviceInfo] = [
        AudioDeviceInfo(
            index=source.index,
            name=f"Pulse: {source.description}",
            hostapi="PulseAudio",
            max_input_channels=source.channels,
            max_output_channels=0,
            default_samplerate=source.sample_rate,
        )
        for source in _pulse_sources()
    ]
    result.extend(
        AudioDeviceInfo(
            index=sink.index,
            name=f"Pulse: {sink.description}",
            hostapi="PulseAudio",
            max_input_channels=0,
            max_output_channels=sink.channels,
            default_samplerate=sink.sample_rate,
        )
        for sink in _pulse_sinks()
    )

    try:
        sounddevice = _import_sounddevice()
        devices = sounddevice.query_devices()
        hostapis = sounddevice.query_hostapis()
    except Exception:
        return result

    for index, device in enumerate(devices):
        hostapi_index = int(device["hostapi"])
        hostapi = str(hostapis[hostapi_index]["name"]) if 0 <= hostapi_index < len(hostapis) else ""
        if platform.system() == "Windows" and hostapi != "Windows WASAPI":
            continue
        result.append(
            AudioDeviceInfo(
                index=index,
                name=str(device["name"]),
                hostapi=hostapi,
                max_input_channels=int(device["max_input_channels"]),
                max_output_channels=int(device["max_output_channels"]),
                default_samplerate=float(device["default_samplerate"]),
            )
        )
    return result


def ensure_virtual_mic_route() -> VoiceRouteStatus:
    if platform.system() == 'Windows':
        route = get_virtual_mic_route_status()
        if not route.sink_present:
            raise RuntimeError('Install VB-CABLE separately, restart, then select CABLE Input under Virtual mic.')
        return route
    pactl = shutil.which("pactl")
    if pactl is None:
        raise RuntimeError("pactl is not installed; install pulseaudio-utils or pipewire-pulse tools")

    status = get_virtual_mic_route_status()
    if not status.sink_present:
        _run_pactl(
            [
                "load-module",
                "module-null-sink",
                f"sink_name={VIRTUAL_SINK_NAME}",
                f"sink_properties=device.description={VIRTUAL_SINK_DESCRIPTION}",
            ]
        )
    status = get_virtual_mic_route_status()
    if not status.source_present:
        _run_pactl(
            [
                "load-module",
                "module-remap-source",
                f"master={VIRTUAL_SINK_NAME}.monitor",
                f"source_name={VIRTUAL_SOURCE_NAME}",
                f"source_properties=device.description={VIRTUAL_SOURCE_DESCRIPTION}",
            ]
        )
    return get_virtual_mic_route_status()


def get_virtual_mic_route_status() -> VoiceRouteStatus:
    if platform.system() == 'Windows':
        devices = list_audio_devices()
        return VoiceRouteStatus(sink_name='CABLE Input', source_name='CABLE Output',
            sink_present=any('cable input' in d.name.lower() and d.max_output_channels for d in devices),
            source_present=any('cable output' in d.name.lower() and d.max_input_channels for d in devices))
    sinks = _pactl_short("sinks")
    sources = _pactl_short("sources")
    return VoiceRouteStatus(
        sink_name=VIRTUAL_SINK_NAME,
        source_name=VIRTUAL_SOURCE_NAME,
        sink_present=any(VIRTUAL_SINK_NAME in line for line in sinks),
        source_present=any(VIRTUAL_SOURCE_NAME in line for line in sources),
        pactl_available=shutil.which("pactl") is not None,
        pacat_available=shutil.which("pacat") is not None,
    )


def repair_virtual_mic_capture_routes() -> int:
    source_index = _pulse_source_index(VIRTUAL_SOURCE_NAME)
    if source_index is None:
        return 0

    moved = 0
    for source_output in _pulse_source_outputs():
        if source_output.target_object != VIRTUAL_SOURCE_NAME:
            continue
        if source_output.source == source_index:
            continue
        try:
            _run_pactl(["move-source-output", str(source_output.index), VIRTUAL_SOURCE_NAME])
        except Exception:
            continue
        moved += 1
    return moved


def _repair_virtual_mic_routes_worker(stop_event: threading.Event) -> None:
    while not stop_event.wait(2.0):
        repair_virtual_mic_capture_routes()


def sounddevice_available() -> bool:
    try:
        _import_sounddevice()
    except Exception:
        return False
    return True


def portaudio_available() -> bool:
    if platform.system() == "Windows":
        return sounddevice_available()
    return ctypes.util.find_library("portaudio") is not None


def pulse_server_name() -> str | None:
    if platform.system() == "Windows":
        return None
    try:
        output = subprocess.check_output(["pactl", "info"], text=True, timeout=2)
    except Exception:
        return None
    for line in output.splitlines():
        if line.startswith("Server Name:"):
            return line.split(":", 1)[1].strip()
    return None


def _pulse_source_for_input_device(input_device: int | None) -> _PulseSource | None:
    if platform.system() == "Windows":
        return None
    if input_device is None or input_device > PULSE_SOURCE_INDEX_BASE or input_device <= PULSE_SINK_INDEX_BASE:
        return None
    for source in _pulse_sources():
        if source.index == input_device:
            return source
    raise RuntimeError("Selected Pulse input source is no longer available; refresh devices")


def _pulse_sink_for_output_device(output_device: int | None) -> _PulseSink | None:
    if platform.system() == "Windows":
        return None
    if output_device is None or output_device > PULSE_SINK_INDEX_BASE:
        return None
    for sink in _pulse_sinks():
        if sink.index == output_device:
            return sink
    raise RuntimeError("Selected Pulse output sink is no longer available; refresh devices")


def _pulse_sources() -> list[_PulseSource]:
    if platform.system() == "Windows":
        return []
    try:
        output = subprocess.check_output(["pactl", "list", "sources"], text=True, timeout=2)
    except Exception:
        return []
    return _parse_pulse_sources(output)


def _pulse_sinks() -> list[_PulseSink]:
    if platform.system() == "Windows":
        return []
    try:
        output = subprocess.check_output(["pactl", "list", "sinks"], text=True, timeout=2)
    except Exception:
        return []
    return _parse_pulse_sinks(output)


def _pulse_source_outputs() -> list[_PulseSourceOutput]:
    if platform.system() == "Windows":
        return []
    try:
        output = subprocess.check_output(["pactl", "list", "source-outputs"], text=True, timeout=2)
    except Exception:
        return []
    return _parse_pulse_source_outputs(output)


def _pulse_source_index(source_name: str) -> int | None:
    for line in _pactl_short("sources"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == source_name:
            return _parse_int(parts[0])
    return None


def _parse_pulse_sources(output: str) -> list[_PulseSource]:
    sources: list[_PulseSource] = []
    current: dict[str, str] = {}

    def finish() -> None:
        if not current:
            return
        name = current.get("name")
        if not name or name == VIRTUAL_SOURCE_NAME:
            return
        if current.get("monitor_of_sink", "n/a") != "n/a":
            return
        source_id = _parse_int(current.get("id"))
        if source_id is None:
            return
        channels, sample_rate = _parse_sample_spec(current.get("sample_specification", ""))
        sources.append(
            _PulseSource(
                index=PULSE_SOURCE_INDEX_BASE - source_id,
                name=name,
                description=current.get("description") or name,
                channels=channels,
                sample_rate=sample_rate,
            )
        )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Source #"):
            finish()
            current = {"id": line.removeprefix("Source #")}
        elif line.startswith("Name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Description:"):
            current["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("Sample Specification:"):
            current["sample_specification"] = line.split(":", 1)[1].strip()
        elif line.startswith("Monitor of Sink:"):
            current["monitor_of_sink"] = line.split(":", 1)[1].strip()
    finish()

    return sources


def _parse_pulse_sinks(output: str) -> list[_PulseSink]:
    sinks: list[_PulseSink] = []
    current: dict[str, str] = {}

    def finish() -> None:
        if not current:
            return
        name = current.get("name")
        if not name or name == VIRTUAL_SINK_NAME:
            return
        sink_id = _parse_int(current.get("id"))
        if sink_id is None:
            return
        channels, sample_rate = _parse_sample_spec(current.get("sample_specification", ""))
        sinks.append(
            _PulseSink(
                index=PULSE_SINK_INDEX_BASE - sink_id,
                name=name,
                description=current.get("description") or name,
                channels=channels,
                sample_rate=sample_rate,
            )
        )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Sink #"):
            finish()
            current = {"id": line.removeprefix("Sink #")}
        elif line.startswith("Name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Description:"):
            current["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("Sample Specification:"):
            current["sample_specification"] = line.split(":", 1)[1].strip()
    finish()

    return sinks


def _parse_pulse_source_outputs(output: str) -> list[_PulseSourceOutput]:
    source_outputs: list[_PulseSourceOutput] = []
    current: dict[str, str] = {}

    def finish() -> None:
        if not current:
            return
        index = _parse_int(current.get("id"))
        if index is None:
            return
        source_outputs.append(
            _PulseSourceOutput(
                index=index,
                source=_parse_int(current.get("source")),
                target_object=current.get("target_object"),
                application_name=current.get("application_name"),
                media_name=current.get("media_name"),
            )
        )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Source Output #"):
            finish()
            current = {"id": line.removeprefix("Source Output #")}
        elif line.startswith("Source:"):
            current["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("application.name ="):
            current["application_name"] = _strip_pactl_property_value(line.split("=", 1)[1])
        elif line.startswith("media.name ="):
            current["media_name"] = _strip_pactl_property_value(line.split("=", 1)[1])
        elif line.startswith("target.object ="):
            current["target_object"] = _strip_pactl_property_value(line.split("=", 1)[1])
    finish()

    return source_outputs


def _strip_pactl_property_value(value: str) -> str:
    return value.strip().strip('"')


def _parse_sample_spec(value: str) -> tuple[int, float]:
    channels = 1
    sample_rate = 48_000.0
    for part in value.split():
        if part.endswith("ch"):
            channels = _parse_int(part[:-2]) or channels
        elif part.endswith("Hz"):
            sample_rate = float(_parse_int(part[:-2]) or int(sample_rate))
    return channels, sample_rate


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _import_sounddevice() -> Any:
    try:
        import sounddevice
    except OSError as exc:
        raise RuntimeError("PortAudio library is missing; install libportaudio2 for sounddevice") from exc
    except Exception as exc:
        raise RuntimeError(f"sounddevice is not available: {exc}") from exc
    return sounddevice


def _sounddevice_output_info(sounddevice: Any, device: int | None) -> dict[str, Any]:
    try:
        if device is None:
            return dict(sounddevice.query_devices(kind="output"))
        return dict(sounddevice.query_devices(device, "output"))
    except Exception as exc:
        label = "default output" if device is None else f"output device {device}"
        raise RuntimeError(f"Could not open {label}: {exc}") from exc


def _sounddevice_input_name(sounddevice: Any | None, device: int | None) -> str | None:
    if sounddevice is None:
        return None
    try:
        if device is None:
            info = sounddevice.query_devices(kind="input")
        else:
            info = sounddevice.query_devices(device, "input")
    except Exception:
        return None
    return str(info.get("name") or "Default input")


def _pactl_short(kind: str) -> list[str]:
    if platform.system() == "Windows":
        return []
    try:
        output = subprocess.check_output(["pactl", "list", "short", kind], text=True, timeout=2)
    except Exception:
        return []
    return output.splitlines()


def _run_pactl(arguments: list[str]) -> None:
    try:
        subprocess.check_output(["pactl", *arguments], stderr=subprocess.STDOUT, text=True, timeout=5)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.output.strip() or f"pactl {' '.join(arguments)} failed") from exc


class WindowsCableWriter:
    def __init__(self, config):
        devices = list_audio_devices()
        device = next((d for d in devices if d.index == config.virtual_output_device), None)
        if device is None or not device.max_output_channels or 'cable input' not in device.name.lower():
            raise RuntimeError('Select the installed CABLE Input playback endpoint for virtual voice output')
        self.output = SounddeviceMonitor(sample_rate=config.sample_rate, block_size=config.block_size,
            output_device=device.index, volume=1, queue_blocks=12)

    @property
    def failed(self):
        return self.output.failed

    @property
    def last_error(self):
        return self.output.last_error

    def start(self):
        self.output.start(_import_sounddevice())

    def write(self, payload):
        return self.output.write(np.frombuffer(payload, dtype=np.float32))

    def close(self):
        self.output.close()
