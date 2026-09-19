from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .background import VirtualBackgroundEffect
from .detectors import FaceBox, FaceDetector
from .effects import TargetHeadEffect, apply_cartoon, apply_privacy_blur
from .lipsync import MouthAnchor, apply_mouth_animation
from .neural import NeuralFaceSwapEffect
from .schemas import EffectConfig
from .settings import get_settings


class FrameProcessor:
    def __init__(self) -> None:
        settings = get_settings()
        self._detector = FaceDetector()
        self._previous_face: FaceBox | None = None
        self._missed_face_frames = 0
        self._max_missed_face_frames = 6
        self._frame_index = 0
        self._detect_every_n_frames = 3
        self._target_head = TargetHeadEffect()
        self._neural_faceswap = NeuralFaceSwapEffect(settings.models_dir, settings.neural_cache_dir)
        self._background = VirtualBackgroundEffect(settings.background_model_path)
        self._default_model_path = settings.models_dir / "inswapper_128.onnx"
        self._default_target_path = settings.assets_dir / "aging-cyber-monk-target.png"
        self._default_background_path = settings.assets_dir / "ordo-basement-cyber-warehouse-photoreal.png"
        self._last_mode = "passthrough"
        self._background_active = False
        self._face_locked = False
        self._lip_sync_driver: Callable[[], float] | None = None

    def set_lip_sync_driver(self, driver: Callable[[], float] | None) -> None:
        self._lip_sync_driver = driver

    @property
    def target_ready(self) -> bool:
        return self._target_head.ready or (self._neural_active and self._neural_faceswap.target_ready)

    @property
    def model_ready(self) -> bool:
        return self._neural_active and self._neural_faceswap.ready

    @property
    def model_provider(self) -> str | None:
        return self._neural_faceswap.active_provider if self._neural_active else None

    @property
    def model_latency_ms(self) -> float:
        return self._neural_faceswap.timings.frame_ms if self._neural_active else 0.0

    @property
    def model_detect_ms(self) -> float:
        return self._neural_faceswap.timings.detect_ms if self._neural_active else 0.0

    @property
    def model_swap_ms(self) -> float:
        return self._neural_faceswap.timings.swap_ms if self._neural_active else 0.0

    @property
    def model_error(self) -> str | None:
        return self._neural_faceswap.last_error if self._neural_active else None

    @property
    def background_ready(self) -> bool:
        return self._background_active and self._background.ready

    @property
    def background_latency_ms(self) -> float:
        return self._background.timings.frame_ms if self._background_active else 0.0

    @property
    def background_segment_ms(self) -> float:
        return self._background.timings.segment_ms if self._background_active else 0.0

    @property
    def background_error(self) -> str | None:
        return self._background.last_error if self._background_active else None

    @property
    def face_locked(self) -> bool:
        return self._face_locked

    @property
    def _neural_active(self) -> bool:
        return self._last_mode == "onnx_faceswap"

    def process(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
        previous_mode = self._last_mode
        self._last_mode = config.mode
        self._background_active = config.background_enabled
        self._face_locked = False
        self._frame_index += 1
        if previous_mode != config.mode:
            self._previous_face = None
            self._missed_face_frames = 0
        if config.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        mask_source = frame_bgr
        if config.mode == "onnx_faceswap":
            settings = get_settings()
            model_path = _resolve_model_path(config.model_path, self._default_model_path, settings.models_dir)
            target_path = _resolve_target_path(config.target_image_path, self._default_target_path)
            background_path = _resolve_target_path(config.background_path, self._default_background_path)

            self._neural_faceswap.configure(
                model_path=model_path,
                target_image_path=target_path,
                provider=config.provider,
                precision=config.precision,
            )
            output = self._neural_faceswap.apply(frame_bgr, config)
            self._face_locked = self._neural_faceswap.last_face is not None
            output = self._apply_background(output, config, background_path, mask_source)
            output = self._apply_lip_sync(
                output,
                self._neural_faceswap.last_face,
                self._neural_faceswap.last_mouth_anchor,
            )
            return output

        if config.mode == "cartoon":
            output = apply_cartoon(frame_bgr, config.strength)
            output = self._apply_background(
                output,
                config,
                _resolve_target_path(config.background_path, self._default_background_path),
                mask_source,
            )
            return output

        output = frame_bgr
        face: FaceBox | None = None
        if config.mode in {"privacy_blur", "target_head"}:
            face = self._tracked_face(frame_bgr, config)

        if face is not None:
            if config.mode == "privacy_blur":
                output = apply_privacy_blur(output, face, config.strength)
            elif config.mode == "target_head":
                target_path = _resolve_target_path(config.target_image_path, self._default_target_path)
                self._target_head.load(str(target_path) if target_path else None)
                output = self._target_head.apply(output, face, config)

        output = self._apply_background(
            output,
            config,
            _resolve_target_path(config.background_path, self._default_background_path),
            mask_source,
        )
        output = self._apply_lip_sync(output, face, None)
        return output

    def _tracked_face(self, frame_bgr: np.ndarray, config: EffectConfig) -> FaceBox | None:
        should_detect = (
            self._previous_face is None
            or self._missed_face_frames > 0
            or self._frame_index % self._detect_every_n_frames == 0
        )
        if not should_detect and self._previous_face is not None:
            self._face_locked = True
            return self._previous_face

        detected_face = self._detector.detect_largest(frame_bgr)
        self._face_locked = detected_face is not None or self._previous_face is not None
        if detected_face is not None:
            face = detected_face.smoothed(self._previous_face, config.smoothing)
            self._previous_face = face
            self._missed_face_frames = 0
            return face

        if self._previous_face is not None and self._missed_face_frames < self._max_missed_face_frames:
            self._missed_face_frames += 1
            return self._previous_face

        self._previous_face = None
        self._missed_face_frames = 0
        self._face_locked = False
        return None

    def _apply_background(
        self,
        output_bgr: np.ndarray,
        config: EffectConfig,
        background_path: Path | None,
        mask_source_bgr: np.ndarray,
    ) -> np.ndarray:
        return self._background.apply(
            output_bgr,
            config,
            background_path=background_path,
            mask_source_bgr=mask_source_bgr,
        )

    def close(self) -> None:
        self._background.close()
        self._detector.close()
        self._neural_faceswap.close()
        self._previous_face = None
        self._missed_face_frames = 0

    def _apply_lip_sync(
        self,
        output_bgr: np.ndarray,
        face: FaceBox | None,
        mouth_anchor: MouthAnchor | None,
    ) -> np.ndarray:
        if self._lip_sync_driver is None:
            return output_bgr
        return apply_mouth_animation(output_bgr, face, self._lip_sync_driver(), mouth_anchor)


def _resolve_model_path(value: str | None, default: Path, models_dir: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == models_dir.name:
        return models_dir.joinpath(*path.parts[1:])
    return models_dir / path


def _resolve_target_path(value: str | None, default: Path) -> Path | None:
    if not value:
        return default if default.exists() else None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'assets':
        return get_settings().assets_dir.joinpath(*path.parts[1:])
    return get_settings().project_root / path
