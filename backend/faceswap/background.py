from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import EffectConfig


@dataclass
class BackgroundTimings:
    frame_ms: float = 0.0
    segment_ms: float = 0.0


class VirtualBackgroundEffect:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._configured_key: tuple[object, ...] | None = None
        self._segmenter: Any | None = None
        self._mp: Any | None = None
        self._background_bgr: np.ndarray | None = None
        self._last_alpha: np.ndarray | None = None
        self._resized_background: np.ndarray | None = None
        self._resized_shape: tuple[int, int] | None = None
        self._retry_at = 0.0
        self._ready = False
        self._last_error: str | None = None
        self._timings = BackgroundTimings()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def timings(self) -> BackgroundTimings:
        return self._timings

    def close(self) -> None:
        self._reset(None)

    def apply(
        self,
        frame_bgr: np.ndarray,
        config: EffectConfig,
        *,
        background_path: Path | None,
        mask_source_bgr: np.ndarray | None = None,
    ) -> np.ndarray:
        if not config.background_enabled:
            self._last_alpha = None
            self._timings = BackgroundTimings()
            return frame_bgr

        self.configure(background_path)
        if not self.ready or self._segmenter is None or self._background_bgr is None or self._mp is None:
            return frame_bgr

        total_start = time.perf_counter()
        try:
            source = mask_source_bgr if mask_source_bgr is not None else frame_bgr
            segment_start = time.perf_counter()
            alpha = self._foreground_alpha(source, config)
            self._timings.segment_ms = _elapsed_ms(segment_start)

            shape = frame_bgr.shape[:2]
            if self._resized_shape != shape:
                self._resized_background = _resize_cover(self._background_bgr, shape[1], shape[0])
                self._resized_shape = shape
            background = self._resized_background
            strength = float(np.clip(config.background_strength, 0.0, 1.0))
            if strength < 1.0:
                background = cv2.addWeighted(background, strength, frame_bgr, 1.0 - strength, 0.0)

            alpha_3 = alpha[:, :, None]
            output = frame_bgr.astype(np.float32) * alpha_3 + background.astype(np.float32) * (1.0 - alpha_3)
            self._timings.frame_ms = _elapsed_ms(total_start)
            self._last_error = None
            return np.clip(output, 0, 255).astype(np.uint8)
        except Exception as exc:  # pragma: no cover - depends on optional MediaPipe runtime
            self._last_error = f"Background segmentation failed: {exc}"
            self._timings.frame_ms = _elapsed_ms(total_start)
            return frame_bgr

    def configure(self, background_path: Path | None) -> None:
        key = (_file_key(self._model_path), _file_key(background_path) if background_path else None)
        if self._configured_key == key and (self.ready or time.monotonic() < self._retry_at):
            return

        self._reset(None)
        self._configured_key = key
        self._retry_at = time.monotonic() + 5.0
        if background_path is None:
            self._reset("Background replacement needs a background image")
            return
        if not self._model_path.exists():
            self._reset(f"Background segmentation model does not exist: {self._model_path}")
            return
        if not background_path.exists():
            self._reset(f"Background image does not exist: {background_path}")
            return

        try:
            self._load(background_path)
        except Exception as exc:  # pragma: no cover - depends on optional MediaPipe runtime
            self._reset(f"Background runtime failed to load: {exc}")

    def _load(self, background_path: Path) -> None:
        mp, vision = _import_mediapipe()
        options = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        background = cv2.imread(str(background_path))
        if background is None:
            raise RuntimeError(f"Could not read background image: {background_path}")

        self._mp = mp
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._background_bgr = background
        self._ready = True
        self._last_error = None

    def _foreground_alpha(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment(image)

        if result.confidence_masks:
            mask_index = 1 if len(result.confidence_masks) > 1 else 0
            alpha = np.asarray(result.confidence_masks[mask_index].numpy_view(), dtype=np.float32).squeeze()
        elif result.category_mask is not None:
            alpha = np.asarray(result.category_mask.numpy_view(), dtype=np.float32).squeeze() / 255.0
        else:
            raise RuntimeError("MediaPipe returned no segmentation mask")

        if alpha.shape[:2] != frame_bgr.shape[:2]:
            alpha = cv2.resize(alpha, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        threshold = float(np.clip(config.background_threshold, 0.0, 0.98))
        alpha = np.clip((alpha - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0)

        if self._last_alpha is not None and self._last_alpha.shape == alpha.shape:
            smoothing = float(np.clip(config.background_smoothing, 0.0, 0.98))
            alpha = self._last_alpha * smoothing + alpha * (1.0 - smoothing)
        self._last_alpha = alpha

        return cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)

    def _reset(self, error: str | None) -> None:
        if self._segmenter is not None:
            close = getattr(self._segmenter, "close", None)
            if close is not None:
                close()
        self._segmenter = None
        self._mp = None
        self._background_bgr = None
        self._resized_background = None
        self._resized_shape = None
        self._last_alpha = None
        self._ready = False
        self._last_error = error
        self._timings = BackgroundTimings()


def _import_mediapipe() -> tuple[Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mediapipe is not installed") from exc
    return mp, vision


def _resize_cover(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image_bgr.shape[:2]
    if source_width == width and source_height == height:
        return image_bgr

    scale = max(width / source_width, height / source_height)
    resized_width = max(width, int(round(source_width * scale)))
    resized_height = max(height, int(round(source_height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    x = max(0, (resized_width - width) // 2)
    y = max(0, (resized_height - height) // 2)
    return np.ascontiguousarray(resized[y : y + height, x : x + width])


def _file_key(path: Path) -> tuple[str, int | None, int | None]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
