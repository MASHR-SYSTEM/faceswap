"""Face-swap adapters. Register trusted adapters at startup, before serving requests.

Factories must be lightweight; model loading belongs in configure(). No optional
neural dependency is imported until the selected adapter needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol
import re

import numpy as np

if TYPE_CHECKING:
    from .detectors import FaceBox
    from .lipsync import MouthAnchor
    from .schemas import EffectConfig


class SwapTimings(Protocol):
    frame_ms: float
    detect_ms: float
    swap_ms: float


class FaceSwapBackend(Protocol):
    """BGR uint8 frames in/out at unchanged dimensions; owns its model resources.

    configure/apply report recoverable errors through last_error and ready.
    apply returns the input frame when unavailable. close releases resources and
    permits later configure calls. Face/mouth coordinates are in output pixels.
    """
    @property
    def ready(self) -> bool: ...
    @property
    def target_ready(self) -> bool: ...
    @property
    def active_provider(self) -> str | None: ...
    @property
    def last_error(self) -> str | None: ...
    @property
    def timings(self) -> SwapTimings: ...
    @property
    def last_face(self) -> FaceBox | None: ...
    @property
    def last_mouth_anchor(self) -> MouthAnchor | None: ...

    def configure(self, *, model_path: Path, target_image_path: Path | None,
                  provider: str, precision: str = "fp32") -> None: ...
    def apply(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class BackendSpec:
    id: str
    label: str
    default_model: str
    factory: Callable[[Path, Path], FaceSwapBackend]


_BACKENDS: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.id):
        raise ValueError("Backend IDs must use lowercase letters, digits and underscores")
    if spec.id in _BACKENDS:
        raise ValueError(f"Face-swap backend already registered: {spec.id}")
    _BACKENDS[spec.id] = spec


def get_backend(backend_id: str) -> BackendSpec:
    try:
        return _BACKENDS[backend_id]
    except KeyError:
        raise ValueError(f"Unknown face-swap backend: {backend_id}") from None


def backend_catalog() -> list[dict[str, str]]:
    return [{"id": spec.id, "label": spec.label, "default_model": spec.default_model}
            for spec in _BACKENDS.values()]


def _inswapper(models_dir: Path, cache_dir: Path) -> FaceSwapBackend:
    from .neural import NeuralFaceSwapEffect
    return NeuralFaceSwapEffect(models_dir, cache_dir)


register_backend(BackendSpec("inswapper", "InSwapper", "inswapper_128.onnx", _inswapper))
