from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .settings import get_settings


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def expanded(self, scale: float, max_width: int, max_height: int, y_offset: float = 0.0) -> "FaceBox":
        cx = self.x + self.w / 2
        cy = self.y + self.h / 2 + self.h * y_offset
        nw = self.w * scale
        nh = self.h * scale
        x = int(max(0, cx - nw / 2))
        y = int(max(0, cy - nh / 2))
        right = int(min(max_width, cx + nw / 2))
        bottom = int(min(max_height, cy + nh / 2))
        return FaceBox(x=x, y=y, w=max(1, right - x), h=max(1, bottom - y))

    def smoothed(self, previous: "FaceBox | None", smoothing: float) -> "FaceBox":
        if previous is None:
            return self
        alpha = 1.0 - smoothing
        return FaceBox(
            x=int(previous.x * smoothing + self.x * alpha),
            y=int(previous.y * smoothing + self.y * alpha),
            w=int(previous.w * smoothing + self.w * alpha),
            h=int(previous.h * smoothing + self.h * alpha),
        )


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade at {cascade_path}")
        self._insightface_app: Any | None = None
        self._insightface_failed = False

    def close(self):
        self._insightface_app = None
        self._insightface_failed = False

    def detect_largest(self, frame_bgr: np.ndarray) -> FaceBox | None:
        insightface_face = self._detect_with_insightface(frame_bgr)
        if insightface_face is not None:
            return insightface_face

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(64, 64),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        boxes = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        return max(boxes, key=lambda box: box.area)

    def _detect_with_insightface(self, frame_bgr: np.ndarray) -> FaceBox | None:
        if self._insightface_failed:
            return None
        if self._insightface_app is None and not self._load_insightface():
            return None
        try:
            faces = self._insightface_app.get(frame_bgr) if self._insightface_app is not None else []
        except Exception:
            self._insightface_failed = True
            return None
        if not faces:
            return None
        return max((_face_box_from_insightface(face) for face in faces), key=lambda box: box.area)

    def _load_insightface(self) -> bool:
        settings = get_settings()
        insightface_root = settings.models_dir / "insightface"
        buffalo_dir = insightface_root / "models" / "buffalo_l"
        if not buffalo_dir.exists():
            self._insightface_failed = True
            return False
        try:
            import insightface
            import onnxruntime as ort
        except Exception:
            self._insightface_failed = True
            return False

        available = set(str(provider) for provider in ort.get_available_providers())
        providers: list[Any] = []
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "gpu_mem_limit": str(1024 * 1024 * 1024),
                    },
                )
            )
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            self._insightface_failed = True
            return False

        try:
            app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=str(insightface_root),
                providers=providers,
                allowed_modules=["detection"],
            )
            app.prepare(ctx_id=0 if "CUDAExecutionProvider" in available else -1, det_size=(320, 320))
        except Exception:
            self._insightface_failed = True
            return False
        self._insightface_app = app
        return True


def _face_box_from_insightface(face: Any) -> FaceBox:
    x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox[:4]]
    return FaceBox(x=max(0, x1), y=max(0, y1), w=max(1, x2 - x1), h=max(1, y2 - y1))
