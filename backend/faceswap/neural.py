from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from types import SimpleNamespace

import cv2
import numpy as np

from .detectors import FaceBox
from .lipsync import MouthAnchor
from .schemas import EffectConfig
from .swapping import swap_frame

NeuralProviderPreference = Literal["auto", "tensorrt", "cuda", "cpu"]


@dataclass(frozen=True)
class NeuralRuntimeConfig:
    model_path: Path
    target_image_path: Path
    provider: NeuralProviderPreference
    models_dir: Path
    cache_dir: Path
    det_size: tuple[int, int] = (320, 320)
    precision: str = "fp32"

    @property
    def key(self) -> tuple[object, ...]:
        return (
            _file_key(self.model_path),
            _file_key(self.target_image_path),
            self.provider,
            str(self.models_dir.resolve()),
            str(self.cache_dir.resolve()),
            self.det_size,
            self.precision,
        )


@dataclass
class NeuralTimings:
    frame_ms: float = 0.0
    detect_ms: float = 0.0
    swap_ms: float = 0.0
    inference_ms: float = 0.0
    controls_ms: float = 0.0


@dataclass
class NeuralRuntimeInfo:
    ready: bool = False
    active_provider: str | None = None
    available_providers: tuple[str, ...] = ()
    last_error: str | None = None
    timings: NeuralTimings = field(default_factory=NeuralTimings)
    last_face: FaceBox | None = None
    last_mouth_anchor: MouthAnchor | None = None


class NeuralFaceSwapEffect:
    def __init__(self, models_dir: Path, cache_dir: Path) -> None:
        self._models_dir = models_dir
        self._cache_dir = cache_dir
        self._configured_key: tuple[object, ...] | None = None
        self._runtime_key = None
        self._retry_at = 0.0
        self._source_latent = None
        self._face_app: Any | None = None
        self._swapper: Any | None = None
        self._source_face: Any | None = None
        self._previous_face_box: FaceBox | None = None
        self._previous_mouth_anchor: MouthAnchor | None = None
        self._previous_candidate: np.ndarray | None = None
        self._info = NeuralRuntimeInfo()

    @property
    def ready(self) -> bool:
        return self._info.ready

    @property
    def target_ready(self) -> bool:
        return self._source_face is not None

    @property
    def last_error(self) -> str | None:
        return self._info.last_error

    @property
    def active_provider(self) -> str | None:
        return self._info.active_provider

    @property
    def timings(self) -> NeuralTimings:
        return self._info.timings

    @property
    def last_face(self) -> FaceBox | None:
        return self._info.last_face

    @property
    def last_mouth_anchor(self) -> MouthAnchor | None:
        return self._info.last_mouth_anchor

    def configure(
        self,
        *,
        model_path: Path,
        target_image_path: Path | None,
        provider: NeuralProviderPreference,
        precision: str = "fp32",
    ) -> None:
        if target_image_path is None:
            self._reset("Neural swap needs a target image with a visible face")
            return

        config = NeuralRuntimeConfig(
            model_path=model_path,
            target_image_path=target_image_path,
            provider=provider,
            models_dir=self._models_dir,
            cache_dir=self._cache_dir,
            precision=precision,
        )
        key = config.key
        runtime_key = (key[0], *key[2:])
        if self._configured_key == key and (self.ready or time.monotonic() < self._retry_at):
            return
        self._configured_key = key
        self._retry_at = time.monotonic() + 5
        self._reset_live_state()
        try:
            if not config.model_path.exists():
                raise RuntimeError(f"Neural model does not exist: {config.model_path}")
            if not config.target_image_path.exists():
                raise RuntimeError(f"Target image does not exist: {config.target_image_path}")
            if (self._runtime_key == runtime_key and self._face_app is not None
                    and self._swapper is not None):
                self._set_source(config)
                self._info.ready = True
                self._info.last_error = None
            else:
                self._reset(None)
                self._load(config)
                self._runtime_key = runtime_key
        except Exception as exc:
            self._source_face = None
            self._source_latent = None
            self._info.ready = False
            self._info.last_error = f"Neural runtime failed to load: {exc}"

    def apply(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
        if not self.ready or self._face_app is None or self._swapper is None or self._source_face is None:
            self._reset_live_state()
            return frame_bgr

        self._info.timings = NeuralTimings()
        total_start = time.perf_counter()
        try:
            detect_start = time.perf_counter()
            bboxes, keypoints = self._face_app.det_model.detect(frame_bgr, max_num=0)
            faces = [SimpleNamespace(bbox=box[:4], kps=keypoints[index], det_score=box[4])
                     for index, box in enumerate(bboxes)]
            self._info.timings.detect_ms = _elapsed_ms(detect_start)
            if not faces:
                self._info.last_face = None
                self._info.last_mouth_anchor = None
                self._reset_live_state()
                self._info.timings.swap_ms = 0.0
                self._info.timings.frame_ms = _elapsed_ms(total_start)
                return frame_bgr

            target_face = _select_live_face(faces, self._info.last_face)
            current_face_box = _face_box_from_insightface(target_face)
            face_box = _smooth_face_box(current_face_box, self._previous_face_box, config.smoothing)
            self._previous_face_box = face_box
            self._info.last_face = face_box
            mouth_anchor = _mouth_anchor_from_insightface(target_face, face_box)
            mouth_anchor = mouth_anchor.smoothed(
                self._previous_mouth_anchor,
                float(np.clip(config.smoothing, 0.0, 0.92)),
            )
            self._previous_mouth_anchor = mouth_anchor
            self._info.last_mouth_anchor = mouth_anchor

            swap_start = time.perf_counter()
            raw_output = swap_frame(self._swapper, frame_bgr, target_face, self._source_latent, self._info.timings)
            controls_start = time.perf_counter()
            output, self._previous_candidate = _apply_live_controls(
                frame_bgr, raw_output, face_box, config, self._previous_candidate)
            self._info.timings.controls_ms = _elapsed_ms(controls_start)
            self._info.timings.swap_ms = _elapsed_ms(swap_start)
            self._info.timings.frame_ms = _elapsed_ms(total_start)
            self._info.last_error = None
            return output
        except Exception as exc:  # pragma: no cover - depends on optional GPU stack/model internals
            self._info.last_error = f"Neural inference failed: {exc}"
            self._info.last_face = None
            self._info.last_mouth_anchor = None
            self._reset_live_state()
            self._info.timings.frame_ms = _elapsed_ms(total_start)
            return frame_bgr

    def _load(self, config: NeuralRuntimeConfig) -> None:
        insightface, ort = _import_neural_dependencies()
        _preload_onnxruntime_dlls(ort)
        available = tuple(ort.get_available_providers())
        self._info.available_providers = available

        from .runtime_models import artifact_directory
        directory = artifact_directory(config.model_path, config.cache_dir, config.precision, ort.__version__)
        provider_specs = build_provider_specs(config.provider, available, directory, config.precision)
        if not provider_specs:
            raise RuntimeError("No ONNX Runtime execution providers are available")

        try:
            self._load_with_providers(insightface, config, provider_specs)
        except Exception:
            if not _contains_provider(provider_specs, "TensorrtExecutionProvider"):
                raise
            fallback_specs = build_provider_specs("cuda", available, directory, config.precision)
            if not fallback_specs:
                raise
            self._load_with_providers(insightface, config, fallback_specs)
            provider_specs = fallback_specs

        self._info.ready = True
        self._info.last_error = None
        self._info.active_provider = _active_provider_from_model(self._swapper) or _first_provider_name(provider_specs)

    def _load_with_providers(self, insightface: Any, config: NeuralRuntimeConfig, provider_specs: list[Any]) -> None:
        insightface_root = config.models_dir / "insightface"
        buffalo_dir = insightface_root / "models" / "buffalo_l"
        if not buffalo_dir.exists():
            raise RuntimeError(
                "InsightFace buffalo_l assets are missing. Place them under "
                f"{buffalo_dir} before starting neural swap."
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Load only the two required models, not every buffalo_l model before filtering.
        context = 0 if _uses_gpu(provider_specs) else -1
        detector = insightface.model_zoo.get_model(str(buffalo_dir / 'det_10g.onnx'), providers=provider_specs)
        recognizer = insightface.model_zoo.get_model(str(buffalo_dir / 'w600k_r50.onnx'), providers=provider_specs)
        detector.prepare(ctx_id=context, input_size=config.det_size)
        recognizer.prepare(ctx_id=context)
        def source_faces(image):
            from insightface.app.common import Face
            boxes, points = detector.detect(image, max_num=0)
            if not len(boxes):
                return []
            index = max(range(len(boxes)), key=lambda i: _insightface_bbox_area(boxes[i]))
            face = Face(bbox=boxes[index, :4], kps=points[index], det_score=boxes[index, 4])
            recognizer.get(image, face)
            return [face]
        face_app = SimpleNamespace(det_model=detector, get=source_faces)

        self._face_app = face_app
        from insightface.model_zoo.inswapper import INSwapper
        import onnxruntime as ort
        from .runtime_models import artifact_directory, fp16_model
        runtime_path = config.model_path
        if config.precision == 'fp16' and not _contains_provider(provider_specs, 'TensorrtExecutionProvider'):
            directory = artifact_directory(config.model_path, config.cache_dir, config.precision, ort.__version__)
            runtime_path = fp16_model(config.model_path, directory)
        session = ort.InferenceSession(str(runtime_path), providers=provider_specs)
        self._swapper = INSwapper(model_file=str(config.model_path), session=session)
        self._set_source(config)

    def _set_source(self, config):
        target = cv2.imread(str(config.target_image_path))
        if target is None:
            raise RuntimeError(f"Could not read target image: {config.target_image_path}")
        faces = self._face_app.get(target)
        if not faces:
            raise RuntimeError("Target image does not contain a detectable face")
        self._source_face = _largest_insightface_face(faces)
        latent = np.dot(self._source_face.normed_embedding.reshape((1, -1)), self._swapper.emap)
        self._source_latent = latent / np.linalg.norm(latent)

    def reset_tracking(self):
        self._reset_live_state()
        self._info.last_face = None
        self._info.last_mouth_anchor = None

    def close(self):
        self._reset(None)
        self._runtime_key = None
        self._configured_key = None

    def _reset(self, error: str | None) -> None:
        self._face_app = None
        self._swapper = None
        self._source_face = None
        self._source_latent = None
        self._reset_live_state()
        self._info.ready = False
        self._info.active_provider = None
        self._info.last_error = error
        self._info.last_face = None
        self._info.last_mouth_anchor = None
        self._info.timings = NeuralTimings()

    def _reset_live_state(self) -> None:
        self._previous_face_box = None
        self._previous_mouth_anchor = None
        self._previous_candidate = None


def build_provider_specs(
    preference: NeuralProviderPreference,
    available: tuple[str, ...] | list[str],
    cache_dir: Path,
    precision: str = "fp32",
) -> list[Any]:
    available_set = set(available)
    if preference == "cpu":
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in available_set else []

    specs: list[Any] = []
    if preference in {"auto", "tensorrt"} and "TensorrtExecutionProvider" in available_set:
        specs.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": "0",
                    "trt_fp16_enable": str(precision == "fp16"),
                    "trt_engine_cache_enable": "True",
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": "True",
                    "trt_timing_cache_path": str(cache_dir),
                },
            )
        )
    if preference in {"auto", "tensorrt", "cuda"} and "CUDAExecutionProvider" in available_set:
        specs.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "gpu_mem_limit": str(6 * 1024 * 1024 * 1024),
                },
            )
        )
    if "CPUExecutionProvider" in available_set:
        specs.append("CPUExecutionProvider")
    return specs


def _import_neural_dependencies() -> tuple[Any, Any]:
    try:
        import insightface
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("insightface is not installed") from exc

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("onnxruntime-gpu is not installed") from exc
    ort.set_default_logger_severity(3)
    return insightface, ort


def _preload_onnxruntime_dlls(ort: Any) -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except TypeError:
        preload()
    except Exception:
        return


def _smooth_face_box(current: FaceBox, previous: FaceBox | None, smoothing: float) -> FaceBox:
    return current.smoothed(previous, float(np.clip(smoothing, 0.0, 0.98))) if previous is not None else current


def _apply_live_controls(
    original_bgr: np.ndarray,
    swapped_bgr: np.ndarray,
    face_box: FaceBox,
    config: EffectConfig,
    previous_output: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = original_bgr.shape[:2]
    boxes = _control_mask_boxes((height, width), face_box, config.scale, config.y_offset)
    if not boxes:
        return original_bgr.copy(), None
    # Sharpen's sigma=1.2 needs neighbouring pixels; 16px exceeds its kernel radius.
    x, y = max(0, min(b.x for b in boxes) - 16), max(0, min(b.y for b in boxes) - 16)
    right = min(width, max(b.x + b.w for b in boxes) + 16)
    bottom = min(height, max(b.y + b.h for b in boxes) + 16)
    mask = np.zeros((bottom - y, right - x), dtype=np.float32)
    _fill_control_mask(mask, boxes, config.edge_feather, x, y)
    previous = (previous_output[y:bottom, x:right] if previous_output is not None
                and previous_output.shape == original_bgr.shape else None)
    output = original_bgr.copy()
    original_roi = original_bgr[y:bottom, x:right]
    candidate = _full_strength_region(
        original_roi, swapped_bgr[y:bottom, x:right], mask, config, previous)
    output[y:bottom, x:right] = _blend_strength(original_roi, candidate, config.strength)
    history = None
    if config.temporal_smoothing > 0:
        history = original_bgr.copy()
        history[y:bottom, x:right] = np.clip(candidate, 0, 255).astype(np.uint8)
    return output, history


def _apply_controls_region(original_bgr, swapped_bgr, mask, config, previous_output):
    candidate = _full_strength_region(original_bgr, swapped_bgr, mask, config, previous_output)
    return _blend_strength(original_bgr, candidate, config.strength)


def _blend_strength(original, candidate, strength):
    # Apply strength last: temporal history must never retain a previous strength.
    strength = float(np.clip(strength, 0.0, 1.0))
    output = original.astype(np.float32) + strength * (candidate - original.astype(np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)


def _full_strength_region(original_bgr, swapped_bgr, mask, config, previous_output):
    output = swapped_bgr.copy()

    color_match = float(np.clip(config.color_match, 0.0, 1.0))
    if color_match > 0.0:
        output = _blend_color_matched(output, original_bgr, mask, color_match)

    sharpen = float(np.clip(config.sharpen, 0.0, 1.0))
    if sharpen > 0.0:
        output = _blend_sharpened(output, mask, sharpen)

    alpha = mask[:, :, None]
    output = original_bgr.astype(np.float32) * (1.0 - alpha) + output.astype(np.float32) * alpha

    temporal_smoothing = float(np.clip(config.temporal_smoothing, 0.0, 0.95))
    if temporal_smoothing > 0.0 and previous_output is not None and previous_output.shape == output.shape:
        temporal_alpha = (mask * temporal_smoothing)[:, :, None]
        output = output * (1.0 - temporal_alpha) + previous_output.astype(np.float32) * temporal_alpha

    return output


def _face_alpha_mask(
    frame_shape: tuple[int, int],
    face_box: FaceBox,
    scale: float,
    y_offset: float,
    edge_feather: float,
) -> np.ndarray:
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.float32)
    boxes = _control_mask_boxes(frame_shape, face_box, scale, y_offset)
    _fill_control_mask(mask, boxes, edge_feather)
    return mask


def _control_mask_boxes(frame_shape, face_box, scale, y_offset):
    height, width = frame_shape
    scale = float(np.clip(scale, .6, 2))
    offset = float(np.clip(y_offset, -.5, .5))
    # Moving the mask must not remove the existing forehead/chin coverage.
    # Union the centred and shifted masks, retaining bounded ROI processing.
    offsets = (0.0,) if offset == 0 else (0.0, offset)
    boxes = [face_box.expanded(scale, width, height, value) for value in offsets]
    return [box for box in boxes if box.x < width and box.y < height]


def _fill_control_mask(mask, boxes, edge_feather, x=0, y=0):
    for box in boxes:
        region = mask[box.y-y:box.y-y+box.h, box.x-x:box.x-x+box.w]
        np.maximum(region, _box_alpha_mask(box, edge_feather), out=region)


def _box_alpha_mask(box: FaceBox, edge_feather: float) -> np.ndarray:
    roi = np.zeros((box.h, box.w), dtype=np.float32)
    center = (max(1, box.w // 2), max(1, box.h // 2))
    axes = (max(1, int(box.w * 0.5)), max(1, int(box.h * 0.55)))
    cv2.ellipse(roi, center, axes, 0, 0, 360, 1.0, -1)

    feather = float(np.clip(edge_feather, 0.0, 1.0))
    if feather > 0.0:
        sigma = max(1.0, min(box.w, box.h) * (0.015 + feather * 0.08))
        roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
        peak = float(roi.max())
        if peak > 0.0:
            roi /= peak

    return np.clip(roi, 0.0, 1.0)


def _blend_color_matched(source_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    mask_u8 = (mask > 0.05).astype(np.uint8) * 255
    if int(mask_u8.sum()) == 0:
        return source_bgr

    source = source_bgr.astype(np.float32)
    target = target_bgr.astype(np.float32)
    source_mean, source_std = cv2.meanStdDev(source, mask=mask_u8)
    target_mean, target_std = cv2.meanStdDev(target, mask=mask_u8)
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1.0)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = np.maximum(target_std.reshape(1, 1, 3), 1.0)

    matched = (source - source_mean) * (target_std / source_std) + target_mean
    blend = (mask * amount)[:, :, None]
    output = source * (1.0 - blend) + matched * blend
    return np.clip(output, 0, 255).astype(np.uint8)


def _blend_sharpened(frame_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    frame = frame_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(frame, 1.0 + amount * 1.4, blurred, -amount * 1.4, 0.0)
    blend = (mask * amount)[:, :, None]
    output = frame * (1.0 - blend) + sharpened * blend
    return np.clip(output, 0, 255).astype(np.uint8)


def _largest_insightface_face(faces: list[Any]) -> Any:
    return max(faces, key=lambda face: _insightface_bbox_area(face.bbox))


def _face_box_from_insightface(face: Any) -> FaceBox:
    x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox[:4]]
    return FaceBox(x=max(0, x1), y=max(0, y1), w=max(1, x2 - x1), h=max(1, y2 - y1))


def _mouth_anchor_from_insightface(face: Any, face_box: FaceBox) -> MouthAnchor:
    kps = getattr(face, "kps", None)
    if kps is not None:
        points = np.asarray(kps, dtype=np.float32)
        if points.shape[0] >= 5 and points.shape[1] >= 2:
            left = points[3, :2]
            right = points[4, :2]
            distance = float(np.linalg.norm(right - left))
            if np.isfinite(distance) and distance >= 4.0:
                center = (left + right) * 0.5
                return MouthAnchor(
                    cx=float(center[0]),
                    cy=float(center[1] + max(distance * 0.08, face_box.h * 0.015)),
                    width=max(distance * 1.85, face_box.w * 0.28),
                    height=max(distance * 0.72, face_box.h * 0.1),
                )

    return MouthAnchor(
        cx=face_box.x + face_box.w * 0.5,
        cy=face_box.y + face_box.h * 0.72,
        width=face_box.w * 0.42,
        height=face_box.h * 0.14,
    )


def _insightface_bbox_area(bbox: Any) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _file_key(path: Path) -> tuple[str, int | None, int | None]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _first_provider_name(provider_specs: list[Any]) -> str | None:
    if not provider_specs:
        return None
    first = provider_specs[0]
    if isinstance(first, tuple):
        return str(first[0])
    return str(first)


def _contains_provider(provider_specs: list[Any], name: str) -> bool:
    return any((provider[0] if isinstance(provider, tuple) else provider) == name for provider in provider_specs)


def _uses_gpu(provider_specs: list[Any]) -> bool:
    return any(
        provider in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
        for provider in [item[0] if isinstance(item, tuple) else item for item in provider_specs]
    )


def _active_provider_from_model(model: Any) -> str | None:
    session = getattr(model, "session", None)
    if session is None:
        return None
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        return None
    providers = get_providers()
    return str(providers[0]) if providers else None


def _select_live_face(faces, previous):
    if previous is None:
        return _largest_insightface_face(faces)
    def overlap(face):
        box = _face_box_from_insightface(face)
        intersection = max(0, min(box.x+box.w, previous.x+previous.w)-max(box.x, previous.x)) * max(0, min(box.y+box.h, previous.y+previous.h)-max(box.y, previous.y))
        return intersection / max(1, box.area + previous.area - intersection)
    candidate = max(faces, key=overlap)
    return candidate if overlap(candidate) >= .15 else _largest_insightface_face(faces)
