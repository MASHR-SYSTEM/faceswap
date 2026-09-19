from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .detectors import FaceBox


@dataclass(frozen=True)
class MouthAnchor:
    cx: float
    cy: float
    width: float
    height: float

    def smoothed(self, previous: "MouthAnchor | None", smoothing: float) -> "MouthAnchor":
        if previous is None:
            return self
        alpha = 1.0 - smoothing
        return MouthAnchor(
            cx=previous.cx * smoothing + self.cx * alpha,
            cy=previous.cy * smoothing + self.cy * alpha,
            width=previous.width * smoothing + self.width * alpha,
            height=previous.height * smoothing + self.height * alpha,
        )


class LipSyncController:
    def __init__(self, *, window_ms: int = 20) -> None:
        self._lock = threading.RLock()
        self._window_ms = window_ms
        self._window_samples = 960
        self._sample_rate = 48_000
        self._start_time: float | None = None
        self._playback_clock: Callable[[], float] | None = None
        self._delay_seconds = 0.11
        self._amount = 1.0
        self._smoothing = 0.35
        self._enabled = True
        self._active = False
        self._envelope: list[float] = []
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_value = 0.0
        self._last_index = -1

    def begin(
        self,
        *,
        sample_rate: int,
        delay_ms: int,
        amount: float,
        smoothing: float,
        enabled: bool,
    ) -> None:
        with self._lock:
            self._sample_rate = int(sample_rate)
            self._window_samples = max(1, int(round(self._sample_rate * self._window_ms / 1000.0)))
            self._start_time = None
            self._playback_clock = None
            self._delay_seconds = delay_ms / 1000.0
            self._amount = float(np.clip(amount, 0.0, 2.0))
            self._smoothing = float(np.clip(smoothing, 0.0, 0.95))
            self._enabled = bool(enabled)
            self._active = bool(enabled)
            self._envelope = []
            self._pending = np.zeros(0, dtype=np.float32)
            self._last_value = 0.0
            self._last_index = -1

    def start_playback(self, clock: Callable[[], float] | None = None) -> None:
        with self._lock:
            self._playback_clock = clock
            self._start_time = time.monotonic()

    def append_audio(self, samples: np.ndarray) -> None:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        with self._lock:
            if not self._active:
                return
            block = np.concatenate([self._pending, mono]) if self._pending.size else mono
            window_count = block.size // self._window_samples
            if window_count:
                windows = block[: window_count * self._window_samples].reshape(window_count, self._window_samples)
                rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
                self._envelope.extend(_rms_to_mouth_value(rms).tolist())
            self._pending = block[window_count * self._window_samples :].copy()

    def finish_audio(self) -> None:
        with self._lock:
            if self._pending.size:
                rms = float(np.sqrt(np.mean(np.square(self._pending, dtype=np.float64))))
                self._envelope.append(float(_rms_to_mouth_value(np.asarray([rms], dtype=np.float64))[0]))
                self._pending = np.zeros(0, dtype=np.float32)

    def end(self) -> None:
        with self._lock:
            self._active = False
            self._last_value = 0.0
            self._last_index = -1

    def value(self, now: float | None = None) -> float:
        with self._lock:
            if not self._active or not self._enabled or self._start_time is None:
                self._last_value = 0.0
                return 0.0

            current_time = time.monotonic() if now is None else now
            position = self._playback_clock() if self._playback_clock else current_time - self._start_time
            audio_seconds = position - self._delay_seconds
            if audio_seconds < 0.0:
                target = 0.0
                index = -1
            else:
                index = int((audio_seconds * self._sample_rate) / self._window_samples)
                target = self._envelope[index] if 0 <= index < len(self._envelope) else 0.0

            if index == self._last_index:
                return float(np.clip(self._last_value * self._amount, 0.0, 2.0))
            smoothing = self._smoothing
            if index != self._last_index and target > self._last_value:
                smoothing = min(smoothing, 0.25)
            self._last_value = self._last_value * smoothing + target * (1.0 - smoothing)
            self._last_index = index
            return float(np.clip(self._last_value * self._amount, 0.0, 2.0))

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active


def apply_mouth_animation(
    frame_bgr: np.ndarray,
    face_box: FaceBox | None,
    mouth_value: float,
    anchor: MouthAnchor | None = None,
) -> np.ndarray:
    aperture = float(np.clip(mouth_value, 0.0, 2.0))
    if face_box is None or aperture < 0.025:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    face = face_box.expanded(1.0, width, height, 0.0)
    if anchor is None:
        anchor = MouthAnchor(
            cx=face.x + face.w * 0.5,
            cy=face.y + face.h * 0.72,
            width=face.w * 0.42,
            height=face.h * 0.14,
        )

    mouth_w = max(8, int(round(anchor.width * (1.18 + 0.14 * min(aperture, 1.4)))))
    mouth_h = max(5, int(round(anchor.height * (1.0 + 0.95 * min(aperture, 1.5)))))
    cx = int(round(anchor.cx))
    cy = int(round(anchor.cy + anchor.height * 0.1 * min(aperture, 1.0)))
    x1 = max(0, cx - mouth_w // 2)
    x2 = min(width, cx + mouth_w // 2)
    y1 = max(0, cy - mouth_h // 2)
    y2 = min(height, cy + mouth_h // 2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr

    output = frame_bgr.copy()
    roi = output[y1:y2, x1:x2].astype(np.float32)
    roi_h, roi_w = roi.shape[:2]
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    center = (max(1, roi_w // 2), max(1, roi_h // 2))
    axes = (max(1, int(roi_w * 0.48)), max(1, int(roi_h * 0.42)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.8, roi_w * 0.035), sigmaY=max(0.8, roi_h * 0.08))
    if float(mask.max()) > 0.0:
        mask /= float(mask.max())

    warped = _mouth_open_warp(roi, aperture)
    blend = np.clip(mask * (0.42 + 0.22 * min(aperture, 1.0)), 0.0, 0.72)[:, :, None]
    morphed = roi * (1.0 - blend) + warped * blend

    shadow_mask = _mouth_shadow_mask(roi_h, roi_w, center, axes)
    shadow = np.clip(shadow_mask * 0.11 * min(aperture, 1.0), 0.0, 0.11)[:, :, None]
    morphed = morphed * (1.0 - shadow)

    output[y1:y2, x1:x2] = np.clip(morphed, 0, 255).astype(np.uint8)
    return output


def _mouth_open_warp(roi: np.ndarray, aperture: float) -> np.ndarray:
    height, width = roi.shape[:2]
    if height < 8 or width < 8:
        return roi
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    normalized_y = y_coords / max(1.0, height - 1.0)
    normalized_x = x_coords / max(1.0, width - 1.0)
    center_weight = np.clip(np.sin(np.clip(normalized_x, 0.0, 1.0) * np.pi), 0.0, 1.0) ** 0.75
    vertical_weight = np.clip(np.sin(np.clip(normalized_y, 0.0, 1.0) * np.pi), 0.0, 1.0) ** 0.9
    lower_pull = np.clip((normalized_y - 0.45) / 0.55, 0.0, 1.0)
    upper_pull = np.clip((0.55 - normalized_y) / 0.55, 0.0, 1.0)
    open_shift = min(aperture, 1.5) * height * 0.22 * center_weight * vertical_weight
    map_y = y_coords.copy()
    map_y += open_shift * upper_pull
    map_y -= open_shift * lower_pull
    map_y = np.clip(map_y, 0.0, height - 1.0).astype(np.float32)

    horizontal_pull = (normalized_x - 0.5) * min(aperture, 1.2) * width * 0.045 * vertical_weight
    map_x = np.clip(x_coords - horizontal_pull, 0.0, width - 1.0).astype(np.float32)
    return cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _mouth_shadow_mask(
    height: int,
    width: int,
    center: tuple[int, int],
    axes: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    inner_axes = (max(1, int(axes[0] * 0.62)), max(1, int(axes[1] * 0.32)))
    inner_center = (center[0], min(height - 1, center[1] + max(1, int(axes[1] * 0.08))))
    cv2.ellipse(mask, inner_center, inner_axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.8, width * 0.04), sigmaY=max(0.8, height * 0.06))
    peak = float(mask.max())
    return mask / peak if peak > 0.0 else mask


def _rms_to_mouth_value(rms: np.ndarray) -> np.ndarray:
    values = (np.asarray(rms, dtype=np.float64) - 0.012) * 11.5
    return np.clip(values, 0.0, 1.0).astype(np.float32)
