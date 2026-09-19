from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detectors import FaceBox
from .schemas import EffectConfig


class TargetHeadEffect:
    def __init__(self) -> None:
        self._path: str | None = None
        self._image_bgra: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._image_bgra is not None

    def load(self, path: str | None) -> None:
        if not path:
            self._path = None
            self._image_bgra = None
            return
        if self._path == path and self._image_bgra is not None:
            return

        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Target image does not exist: {path}")

        rgba = Image.open(image_path).convert("RGBA")
        self._image_bgra = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
        self._path = path

    def apply(self, frame_bgr: np.ndarray, face: FaceBox, config: EffectConfig) -> np.ndarray:
        if self._image_bgra is None:
            return frame_bgr

        box = face.expanded(config.scale, frame_bgr.shape[1], frame_bgr.shape[0], config.y_offset)
        overlay = cv2.resize(self._image_bgra, (box.w, box.h), interpolation=cv2.INTER_AREA)

        rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0
        alpha *= config.strength

        ellipse_mask = np.zeros((box.h, box.w), dtype=np.float32)
        cv2.ellipse(
            ellipse_mask,
            (box.w // 2, box.h // 2),
            (max(1, int(box.w * 0.45)), max(1, int(box.h * 0.50))),
            0,
            0,
            360,
            1,
            -1,
        )
        ellipse_mask = cv2.GaussianBlur(ellipse_mask, (0, 0), sigmaX=max(3, box.w * 0.035))
        alpha = np.clip(alpha * ellipse_mask, 0.0, 1.0)[:, :, None]

        roi = frame_bgr[box.y : box.y + box.h, box.x : box.x + box.w]
        corrected = _match_color(rgb, roi)
        blended = corrected.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
        frame_bgr[box.y : box.y + box.h, box.x : box.x + box.w] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame_bgr


def apply_cartoon(frame_bgr: np.ndarray, strength: float) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=80, sigmaSpace=80)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 8)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(smooth, edges_bgr)
    return cv2.addWeighted(frame_bgr, 1.0 - strength, cartoon, strength, 0)


def apply_privacy_blur(frame_bgr: np.ndarray, face: FaceBox, strength: float) -> np.ndarray:
    box = face.expanded(1.25, frame_bgr.shape[1], frame_bgr.shape[0], -0.02)
    roi = frame_bgr[box.y : box.y + box.h, box.x : box.x + box.w]
    k = max(15, int(min(box.w, box.h) * 0.22) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame_bgr[box.y : box.y + box.h, box.x : box.x + box.w] = cv2.addWeighted(
        roi,
        1.0 - strength,
        blurred,
        strength,
        0,
    )
    return frame_bgr


def draw_debug(frame_bgr: np.ndarray, face: FaceBox | None) -> np.ndarray:
    if face is None:
        cv2.putText(frame_bgr, "NO FACE", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 64, 255), 2)
        return frame_bgr
    cv2.rectangle(frame_bgr, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 160), 2)
    cv2.putText(frame_bgr, "FACE LOCK", (face.x, max(24, face.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 160), 2)
    return frame_bgr


def _match_color(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    source = source_bgr.astype(np.float32)
    target = target_bgr.astype(np.float32)
    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1.0)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = np.maximum(target_std.reshape(1, 1, 3), 1.0)
    matched = (source - source_mean) * (target_std / source_std) + target_mean
    return np.clip(matched, 0, 255).astype(np.uint8)
