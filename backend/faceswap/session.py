from __future__ import annotations

from .lifecycle import WorkerLifecycle, serialized
from .frames import LatestCapture, LatestOutput

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .camera import list_camera_devices, capture_backend
from .processor import FrameProcessor
from .settings import get_settings
from .lifecycle import SessionBusy
from .schemas import EffectConfig, SessionStartRequest, SessionStatus


@dataclass
class _RuntimeState:
    active_capture: dict | None = None
    virtual_camera_state: str = "disabled"
    phase: str = "idle"
    generation: int = 0
    running: bool = False
    source_index: int | None = None
    width: int | None = None
    height: int | None = None
    fps_target: int | None = None
    fps_actual: float = 0.0
    frames_processed: int = 0
    dropped_frames: int = 0
    frame_age_ms: float = 0.0
    capture_sequence: int = 0
    effect_revision: int = 0
    preview_ready: bool = False
    virtual_camera: bool = False
    virtual_camera_ready: bool = False
    active_effect: str = "passthrough"
    last_error: str | None = None
    model_ready: bool = False
    model_provider: str | None = None
    model_latency_ms: float = 0.0
    model_detect_ms: float = 0.0
    model_swap_ms: float = 0.0
    target_ready: bool = False
    face_locked: bool = False
    background_ready: bool = False
    background_latency_ms: float = 0.0
    background_segment_ms: float = 0.0
    background_error: str | None = None


class VideoSession(WorkerLifecycle):
    def __init__(self, config_path=None) -> None:
        self._init_lifecycle()
        self._lock = threading.RLock()
        self._preview_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._state = _RuntimeState()
        self._config_path = config_path
        self._effect = EffectConfig()
        if config_path is not None and config_path.exists():
            try:
                self._effect = EffectConfig.model_validate_json(config_path.read_text())
            except (ValueError, OSError):
                pass
        self._processor = FrameProcessor()

    def set_lip_sync_driver(self, driver: Callable[[], float] | None) -> None:
        self._processor.set_lip_sync_driver(driver)

    @serialized
    def start(self, request: SessionStartRequest) -> SessionStatus:
        source_index = _resolve_source_index(request.source_index, request.source_id)
        active_capture = request.model_dump(exclude={"effect"})
        self.stop()
        request = request.model_copy(update={"source_index": source_index})
        with self._lock:
            self._require_stopped()
            self._latest_jpeg = None
            self.update_effect(request.effect)
            self._state = _RuntimeState(
                active_capture=active_capture,
                running=True,
                source_index=request.source_index,
                width=request.width,
                height=request.height,
                fps_target=request.fps,
                virtual_camera=request.virtual_camera,
                active_effect=request.effect.mode,
                effect_revision=self._state.effect_revision,
            )
            self._launch(request, "faceswap-video")
            return self.status()

    def update_effect(self, effect: EffectConfig, expected_revision: int | None = None) -> SessionStatus:
        with self._lock:
            if expected_revision is not None and expected_revision != self._state.effect_revision:
                raise SessionBusy('Configuration changed; refresh before applying another edit')
            if self._config_path is not None:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(dir=self._config_path.parent)
                try:
                    with os.fdopen(fd, 'w') as stream:
                        stream.write(effect.model_dump_json(indent=2))
                    os.replace(name, self._config_path)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
            self._state.effect_revision += 1
            self._effect = effect
            self._state.active_effect = effect.mode
            self._state.last_error = None
            return self.status()

    @serialized
    def stop(self) -> SessionStatus:
        if self._stop_worker():
            with self._preview_condition:
                self._latest_jpeg = None
                self._state.preview_ready = False
                self._state.virtual_camera_ready = False
                self._state.virtual_camera_state = "disabled"
                self._preview_condition.notify_all()
        return self.status()

    def status(self) -> SessionStatus:
        with self._lock:
            return SessionStatus(**self._state.__dict__)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _current_effect(self) -> EffectConfig:
        with self._lock:
            if hasattr(self._effect, "model_copy"):
                return self._effect.model_copy(deep=True)
            return self._effect.copy(deep=True)

    def mjpeg_stream(self):
        last_frame = None
        while True:
            with self._preview_condition:
                self._preview_condition.wait_for(
                    lambda: self._latest_jpeg is not last_frame
                    or not self._state.running
                    or self._stop_event.is_set(),
                    timeout=1.0,
                )
                if not self._state.running or self._stop_event.is_set():
                    break
                frame = self._latest_jpeg
            if frame is None or frame is last_frame:
                continue
            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

    def _run(self, request: SessionStartRequest) -> None:
        capture: cv2.VideoCapture | None = None
        virtual_camera = None
        reader = None
        output_worker = None
        try:
            capture = cv2.VideoCapture(request.source_index, capture_backend())
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, request.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, request.height)
            if request.fps is not None:
                capture.set(cv2.CAP_PROP_FPS, request.fps)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {request.source_index}; it may be busy or inaccessible. No other camera was selected.")

            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or request.width
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or request.height
            detected_fps = capture.get(cv2.CAP_PROP_FPS)
            virtual_fps = request.fps or (int(round(detected_fps)) if detected_fps and detected_fps > 0 else 30)

            if request.virtual_camera:
                virtual_camera = self._open_virtual_camera(actual_width, actual_height, virtual_fps)

                output_worker = LatestOutput(virtual_camera)
            reader = LatestCapture(capture).start()
            last_sequence = 0
            self._mark_ready()
            tick_start = time.monotonic()
            fps_window_start = tick_start
            fps_window_frames = 0

            while not self._stop_event.is_set():
                item = reader.next(last_sequence)
                if item is None:
                    continue
                sequence, captured_at, frame = item
                with self._lock:
                    self._state.dropped_frames += max(0, sequence - last_sequence - 1)
                    self._state.capture_sequence = sequence
                last_sequence = sequence

                effect = self._current_effect()
                processed = self._processor.process(frame, effect)
                fps_window_frames += 1

                if output_worker is not None:
                    output_worker.publish(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                    with self._lock:
                        self._state.virtual_camera_state = virtual_camera.state
                        self._state.virtual_camera_ready = virtual_camera.state == "ready"

                ok, encoded = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                if ok:
                    with self._preview_condition:
                        self._latest_jpeg = encoded.tobytes()
                        self._state.preview_ready = True
                        self._state.frames_processed += 1
                        self._state.frame_age_ms = (time.monotonic() - captured_at) * 1000
                        self._state.active_effect = effect.mode
                        self._state.model_ready = self._processor.model_ready
                        self._state.model_provider = self._processor.model_provider
                        self._state.model_latency_ms = self._processor.model_latency_ms
                        self._state.model_detect_ms = self._processor.model_detect_ms
                        self._state.model_swap_ms = self._processor.model_swap_ms
                        self._state.target_ready = self._processor.target_ready
                        self._state.face_locked = self._processor.face_locked
                        self._state.background_ready = self._processor.background_ready
                        self._state.background_latency_ms = self._processor.background_latency_ms
                        self._state.background_segment_ms = self._processor.background_segment_ms
                        self._state.background_error = self._processor.background_error
                        if self._processor.model_error:
                            self._state.last_error = self._processor.model_error
                        elif self._processor.background_error:
                            self._state.last_error = self._processor.background_error
                        else:
                            self._state.last_error = None
                        now = time.monotonic()
                        if now - fps_window_start >= 1.0:
                            self._state.fps_actual = fps_window_frames / (now - fps_window_start)
                            fps_window_frames = 0
                            fps_window_start = now
                        self._preview_condition.notify_all()
        except Exception as exc:
            with self._preview_condition:
                self._state.last_error = str(exc)
                self._state.running = False
                self._preview_condition.notify_all()
        finally:
            if reader is not None:
                reader.close()
            elif capture is not None:
                capture.release()
            if output_worker is not None:
                output_worker.close()
            elif virtual_camera is not None:
                virtual_camera.close()
            self._processor.close()
            with self._lock:
                self._state.running = False
                self._state.virtual_camera_ready = False
                self._state.virtual_camera_state = "disabled"

    def _open_virtual_camera(self, width: int, height: int, fps: int):
        from .output import open_output
        return open_output(width, height, fps, get_settings().virtual_camera_device)


def _resolve_source_index(requested_index: int | None, source_id: str | None = None) -> int:
    devices = list_camera_devices()
    if source_id is not None:
        selected = next((d for d in devices if d.device_id == source_id), None)
        if selected is None:
            raise ValueError('The selected camera is unavailable. Reconnect it or choose another camera.')
        return selected.index
    if requested_index is None:
        standard = next((device for device in devices if device.kind == "standard"), None)
        if standard is None:
            raise ValueError("No standard webcam found. Connect it or explicitly select another camera.")
        return standard.index
    if not any(device.index == requested_index for device in devices):
        raise ValueError(f"Selected camera {requested_index} is unavailable. Refresh the camera list and select a device.")
    return requested_index
