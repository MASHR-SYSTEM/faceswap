from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import VoiceConfig
from .settings import get_settings


@dataclass
class SpeechSynthesisStatus:
    ready: bool = False
    latency_ms: float = 0.0
    queue_ms: float = 0.0
    rt_factor: float = 0.0
    fallback: bool = False
    detail: str | None = None


class SpeechSynthesisEngine(Protocol):
    def start(self) -> None: ...

    def process(self, mono: np.ndarray) -> np.ndarray: ...

    def status(self) -> SpeechSynthesisStatus: ...

    def close(self) -> None: ...


class SpeechToSpeechProcessor:
    def __init__(self, sample_rate: int, config: VoiceConfig) -> None:
        self.sample_rate = sample_rate
        self.config = config
        try:
            self._engine: SpeechSynthesisEngine = SeedVcEngine(sample_rate, config)
            self._engine.start()
        except Exception as exc:
            self._engine = LowLatencyPresetSpeechEngine(
                sample_rate,
                config,
                fallback_detail=f"Seed-VC adapter unavailable: {exc}",
            )
            self._engine.start()

    def process(self, block: np.ndarray) -> tuple[np.ndarray, float]:
        mono = _mono_samples(block)
        level = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if mono.size else 0.0
        if level < self.config.noise_gate:
            mono = mono * 0.05
        output = self._engine.process(mono)
        return np.clip(output, -0.98, 0.98).astype(np.float32), level

    def status(self) -> SpeechSynthesisStatus:
        return self._engine.status()

    def close(self) -> None:
        self._engine.close()


class SeedVcEngine:
    def __init__(self, sample_rate: int, config: VoiceConfig) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self._adapter: Any | None = None
        self._status = SpeechSynthesisStatus(detail="Seed-VC adapter not started")

    def start(self) -> None:
        adapter_path = get_settings().models_dir / "speech/seed-vc/realtime_adapter.py"
        if not adapter_path.exists():
            raise RuntimeError(f"missing {adapter_path}")

        spec = importlib.util.spec_from_file_location("faceswap_seedvc_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        create_engine = getattr(module, "create_engine", None)
        if create_engine is None:
            raise RuntimeError("adapter must expose create_engine(model_dir, target, quality, sample_rate)")

        self._adapter = create_engine(
            model_dir=str(adapter_path.parent),
            target=self.config.synthesis_target,
            quality=self.config.synthesis_quality,
            sample_rate=self.sample_rate,
        )
        start = getattr(self._adapter, "start", None)
        if start is not None:
            start()
        self._status = SpeechSynthesisStatus(ready=True, detail="Seed-VC adapter ready")

    def process(self, mono: np.ndarray) -> np.ndarray:
        if self._adapter is None:
            raise RuntimeError("Seed-VC adapter is not started")

        block_start = time.perf_counter()
        process = getattr(self._adapter, "process", None)
        if process is None:
            raise RuntimeError("Seed-VC adapter must expose process(mono)")
        output = np.asarray(process(mono.astype(np.float32, copy=False)), dtype=np.float32)
        if output.shape != mono.shape:
            output = _fit_length(output, len(mono))
        elapsed_ms = (time.perf_counter() - block_start) * 1000.0
        self._status.latency_ms = float(
            getattr(self._adapter, "latency_ms", elapsed_ms + _block_ms(len(mono), self.sample_rate)) or 0.0
        )
        self._status.rt_factor = float(
            getattr(self._adapter, "rt_factor", elapsed_ms / max(_block_ms(len(mono), self.sample_rate), 1e-6)) or 0.0
        )
        self._status.queue_ms = float(getattr(self._adapter, "queue_ms", 0.0) or 0.0)
        self._status.detail = getattr(self._adapter, "detail", self._status.detail)
        return output

    def status(self) -> SpeechSynthesisStatus:
        if self._adapter is not None:
            self._status.queue_ms = float(getattr(self._adapter, "queue_ms", self._status.queue_ms) or 0.0)
            self._status.latency_ms = float(getattr(self._adapter, "latency_ms", self._status.latency_ms) or 0.0)
            self._status.rt_factor = float(getattr(self._adapter, "rt_factor", self._status.rt_factor) or 0.0)
            self._status.detail = getattr(self._adapter, "detail", self._status.detail)
        return replace(self._status)

    def close(self) -> None:
        if self._adapter is not None:
            close = getattr(self._adapter, "close", None)
            if close is not None:
                close()
        self._adapter = None


class LowLatencyPresetSpeechEngine:
    _TARGETS = {
        "monk": {"pitch": -7.5, "lowpass": 2300.0, "drive": 2.8, "mod": 0.1, "noise": 0.075},
        "shadow": {"pitch": -5.0, "lowpass": 1900.0, "drive": 3.1, "mod": 0.14, "noise": 0.09},
        "bright": {"pitch": 2.0, "lowpass": 3600.0, "drive": 2.2, "mod": 0.08, "noise": 0.055},
    }

    def __init__(self, sample_rate: int, config: VoiceConfig, *, fallback_detail: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self._profile = self._TARGETS.get(config.synthesis_target, self._TARGETS["monk"])
        self._status = SpeechSynthesisStatus(
            ready=True,
            fallback=True,
            detail=fallback_detail or "Using local low-latency synthesizer",
        )
        self._pitch = _RealtimePitchShifter(sample_rate)
        self._lp_state = np.array([0.0], dtype=np.float32)
        self._hp_state = np.zeros(2, dtype=np.float32)
        self._comb_a = _DelayLine(sample_rate, delay_seconds=0.009, max_seconds=0.04, feedback=0.14)
        self._comb_b = _DelayLine(sample_rate, delay_seconds=0.016, max_seconds=0.04, feedback=0.11)
        self._sample_index = 0
        self._noise = np.random.default_rng()

    def start(self) -> None:
        self._status.ready = True

    def process(self, mono: np.ndarray) -> np.ndarray:
        block_start = time.perf_counter()
        mono = mono.astype(np.float32, copy=False)
        level = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if mono.size else 0.0
        pitch = float(self._profile["pitch"])
        shifted = self._pitch.process(mono, pitch)
        body = _highpass(shifted, self.sample_rate, 120.0, self._hp_state)
        body = _lowpass(body, self.sample_rate, float(self._profile["lowpass"]), self._lp_state)
        body = body * 0.78 + self._comb_a.process(body) * 0.18 - self._comb_b.process(body) * 0.12
        body = self._modulate(body, float(self._profile["mod"]))
        body = body + self._noise.normal(
            0.0,
            float(np.clip(level * float(self._profile["noise"]), 0.0008, 0.011)),
            size=body.shape,
        ).astype(np.float32)
        drive = float(self._profile["drive"])
        output = np.tanh(body * drive) / np.tanh(drive)
        output = output * self.config.gain

        elapsed_ms = (time.perf_counter() - block_start) * 1000.0
        block_ms = _block_ms(len(mono), self.sample_rate)
        self._status.latency_ms = block_ms + self._pitch.algorithmic_delay_ms
        self._status.rt_factor = elapsed_ms / max(block_ms, 1e-6)
        self._status.queue_ms = 0.0
        return output.astype(np.float32, copy=False)

    def status(self) -> SpeechSynthesisStatus:
        return replace(self._status)

    def close(self) -> None:
        self._status.ready = False

    def _modulate(self, block: np.ndarray, amount: float) -> np.ndarray:
        indices = np.arange(len(block), dtype=np.float32) + float(self._sample_index)
        self._sample_index += len(block)
        slow = np.sin(2.0 * np.pi * 13.0 * indices / self.sample_rate)
        fast = np.sin(2.0 * np.pi * 37.0 * indices / self.sample_rate)
        modulation = 1.0 + amount * slow + amount * 0.45 * fast
        return (block * modulation.astype(np.float32, copy=False)).astype(np.float32, copy=False)


class _RealtimePitchShifter:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._min_delay = max(16, int(round(sample_rate * 0.014)))
        self._grain = max(64, int(round(sample_rate * 0.036)))
        self._ring = np.zeros(self._min_delay + self._grain + 4, dtype=np.float32)
        self._write = 0
        self._phase = 0.0

    @property
    def algorithmic_delay_ms(self) -> float:
        return (self._min_delay + self._grain) * 1000.0 / float(self.sample_rate)

    def process(self, block: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            semitones = 0.6
        factor = float(np.clip(2.0 ** (semitones / 12.0), 0.35, 1.8))
        phase_step = (1.0 - factor) / float(self._grain)
        output = np.empty_like(block, dtype=np.float32)
        for index, sample in enumerate(block.astype(np.float32, copy=False)):
            self._ring[self._write] = sample
            self._write = (self._write + 1) % len(self._ring)

            phase_a = self._phase % 1.0
            phase_b = (phase_a + 0.5) % 1.0
            delay_a = self._min_delay + phase_a * self._grain
            delay_b = self._min_delay + phase_b * self._grain
            window_a = 0.5 - 0.5 * np.cos(phase_a * 2.0 * np.pi)
            window_b = 0.5 - 0.5 * np.cos(phase_b * 2.0 * np.pi)
            output[index] = float(
                (self._read_delay(delay_a) * window_a + self._read_delay(delay_b) * window_b)
                / max(window_a + window_b, 1e-6)
            )
            self._phase = (self._phase + phase_step) % 1.0
        return output

    def _read_delay(self, delay_samples: float) -> float:
        read_pos = (self._write - delay_samples) % len(self._ring)
        left = int(np.floor(read_pos))
        right = (left + 1) % len(self._ring)
        fraction = read_pos - left
        return float(self._ring[left] * (1.0 - fraction) + self._ring[right] * fraction)


class _DelayLine:
    def __init__(self, sample_rate: int, *, delay_seconds: float, max_seconds: float, feedback: float) -> None:
        self._buffer = np.zeros(max(2, int(round(sample_rate * max_seconds))), dtype=np.float32)
        self._delay = max(1, min(len(self._buffer) - 1, int(round(sample_rate * delay_seconds))))
        self._feedback = float(feedback)
        self._index = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        output = np.empty_like(block, dtype=np.float32)
        for i, sample in enumerate(block.astype(np.float32, copy=False)):
            read_index = (self._index - self._delay) % len(self._buffer)
            delayed = self._buffer[read_index]
            output[i] = delayed
            self._buffer[self._index] = sample + delayed * self._feedback
            self._index = (self._index + 1) % len(self._buffer)
        return output


def _mono_samples(block: np.ndarray) -> np.ndarray:
    if block.ndim == 2:
        return np.mean(block, axis=1, dtype=np.float32)
    return block.astype(np.float32, copy=False)


def _lowpass(block: np.ndarray, sample_rate: int, cutoff_hz: float, state: np.ndarray) -> np.ndarray:
    cutoff = float(np.clip(cutoff_hz, 300.0, sample_rate * 0.45))
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    y = float(state[0])
    output = np.empty_like(block, dtype=np.float32)
    for index, sample in enumerate(block.astype(np.float32, copy=False)):
        y = y + alpha * (float(sample) - y)
        output[index] = y
    state[0] = y
    return output


def _highpass(block: np.ndarray, sample_rate: int, cutoff_hz: float, state: np.ndarray) -> np.ndarray:
    cutoff = float(np.clip(cutoff_hz, 20.0, sample_rate * 0.45))
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * np.pi * cutoff)
    alpha = rc / (rc + dt)
    previous_input = float(state[0])
    previous_output = float(state[1])
    output = np.empty_like(block, dtype=np.float32)
    for index, sample in enumerate(block.astype(np.float32, copy=False)):
        sample_float = float(sample)
        previous_output = alpha * (previous_output + sample_float - previous_input)
        previous_input = sample_float
        output[index] = previous_output
    state[0] = previous_input
    state[1] = previous_output
    return output


def _fit_length(block: np.ndarray, length: int) -> np.ndarray:
    if len(block) == length:
        return block.astype(np.float32, copy=False)
    if len(block) > length:
        return block[:length].astype(np.float32, copy=False)
    output = np.zeros(length, dtype=np.float32)
    output[: len(block)] = block.astype(np.float32, copy=False)
    return output


def _block_ms(length: int, sample_rate: int) -> float:
    return length * 1000.0 / float(sample_rate)
