from __future__ import annotations

import io
import json
import queue
import time
import tempfile
from .gpu_jobs import gpu_coordinator, JobCancelled
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps

from .detectors import FaceBox, FaceDetector
from .lipsync import apply_mouth_animation
from .schemas import AvatarProfile, AvatarRenderJob, AvatarRenderRequest, VoiceConfig
from .speech import SpeechToSpeechProcessor
from .tts import (
    FallbackSpeechEngine,
    KokoroTextToSpeechEngine,
    _declick_audio_chunk,
    _mono_float32,
    _resample_audio,
)


@dataclass(slots=True)
class MuseTalkConfig:
    root: Path | None = None
    python: Path | None = None
    ffmpeg_path: str | None = None
    version: str = "v15"
    unet_model_path: Path | None = None
    unet_config_path: Path | None = None
    bbox_shift: int = 0


class AvatarStudio:
    def __init__(
        self,
        assets_dir: Path,
        *,
        renderer: str = "local",
        musetalk_config: MuseTalkConfig | None = None,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.profiles_dir = self.assets_dir / "avatar_profiles"
        self.outputs_dir = self.assets_dir / "avatar_outputs"
        self.voice_presets_dir = self.assets_dir / "voice_presets"
        self._renderer = renderer.strip().lower() or "local"
        self._musetalk_config = musetalk_config or MuseTalkConfig()
        self._lock = threading.RLock()
        self._jobs: dict[str, AvatarRenderJob] = {}
        self._detector: FaceDetector | None = None
        self._tts_engine = KokoroTextToSpeechEngine()
        self._fallback_tts_engine = FallbackSpeechEngine()
        self._ensure_dirs()
        self._queue = queue.Queue(maxsize=8)
        self._queue_thread = None
        self._shutdown = threading.Event()
        self._cancelled = set()
        self._active_job = None
        # Queued requests cannot be safely replayed without their original payload.
        for manifest in self.outputs_dir.glob('*/job.json'):
            try:
                job = AvatarRenderJob(**json.loads(manifest.read_text()))
                if job.status in {'queued', 'running', 'cancelling'}:
                    self._store_job(job.model_copy(update={'status': 'failed',
                        'error': 'Application restarted before this job completed', 'completed_at': _now_iso()}))
            except (ValueError, OSError):
                continue

    @property
    def renderer_name(self) -> str:
        return self._renderer if self._renderer in {"local", "musetalk"} else "local"

    def musetalk_configured(self) -> bool:
        status, _detail = self.musetalk_status()
        return status == "ready"

    def musetalk_status(self) -> tuple[str, str]:
        try:
            _musetalk_command_parts(self._musetalk_config)
        except JobCancelled:
            raise
        except Exception as exc:
            detail = str(exc)
            return _classify_musetalk_error(detail), detail
        return "ready", "MuseTalk renderer configured"

    async def create_profile(
        self,
        *,
        display_name: str,
        target_image: UploadFile,
        voice_reference: UploadFile,
        camera_reference: UploadFile | None,
        consent_confirmed: bool,
    ) -> AvatarProfile:
        name = display_name.strip() or "Avatar"
        if not consent_confirmed:
            raise ValueError("Consent confirmation is required before creating an avatar profile.")

        target_bytes = await target_image.read()
        voice_bytes = await voice_reference.read()
        if not target_bytes:
            raise ValueError("Target image is empty.")
        if not voice_bytes:
            raise ValueError("Voice reference is empty.")

        profile_id = f"{_slugify(name)}-{uuid.uuid4().hex[:10]}"
        profile_dir = self.profiles_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=False)

        target_path = profile_dir / "target.png"
        _write_normalized_image(target_path, target_bytes)

        voice_path = self.voice_presets_dir / f"{profile_id}.wav"
        _write_voice_reference_wav(voice_path, voice_bytes)
        (profile_dir / "voice_reference.wav").write_bytes(voice_path.read_bytes())

        camera_path: Path | None = None
        if camera_reference is not None and camera_reference.filename:
            camera_bytes = await camera_reference.read()
            if camera_bytes:
                camera_path = profile_dir / f"camera_reference{_safe_extension(camera_reference.filename)}"
                camera_path.write_bytes(camera_bytes)

        profile = AvatarProfile(
            id=profile_id,
            display_name=name,
            target_image_path=_workspace_path(target_path),
            voice_reference_path=_workspace_path(voice_path),
            camera_reference_path=_workspace_path(camera_path) if camera_path is not None else None,
            created_at=_now_iso(),
            consent_confirmed=True,
        )
        (profile_dir / "profile.json").write_text(_model_json(profile), encoding="utf-8")
        return profile

    def list_profiles(self) -> list[AvatarProfile]:
        self._ensure_dirs()
        profiles: list[AvatarProfile] = []
        for manifest in self.profiles_dir.glob("*/profile.json"):
            try:
                profiles.append(AvatarProfile(**json.loads(manifest.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return sorted(profiles, key=lambda profile: profile.created_at, reverse=True)

    def get_profile(self, profile_id: str) -> AvatarProfile:
        profile_path = self._profile_dir(profile_id) / "profile.json"
        if not profile_path.exists():
            raise KeyError(profile_id)
        return AvatarProfile(**json.loads(profile_path.read_text(encoding="utf-8")))

    def submit_render(self, request: AvatarRenderRequest) -> AvatarRenderJob:
        profile = self.get_profile(request.profile_id)
        job_id = uuid.uuid4().hex
        job = AvatarRenderJob(
            id=job_id,
            profile_id=profile.id,
            status="queued",
            progress=0.0,
            text_preview=_preview_text(request.text),
            created_at=_now_iso(),
            detail="Queued",
        )
        with self._lock:
            if self._shutdown.is_set():
                raise ValueError('Avatar studio is shutting down')
            if self._queue.full():
                raise ValueError('Render queue is full (maximum eight waiting jobs)')
            self._store_job(job)
            self._queue.put_nowait((job_id, request))
            if self._queue_thread is None or not self._queue_thread.is_alive():
                self._queue_thread = threading.Thread(target=self._queue_worker, name='avatar-queue', daemon=True)
                self._queue_thread.start()
        return job

    def _is_cancelled(self, job_id):
        return self._shutdown.is_set() or job_id in self._cancelled

    def _queue_worker(self):
        while not self._shutdown.is_set():
            try:
                job_id, request = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                with gpu_coordinator.render(lambda: self._is_cancelled(job_id)):
                    self._active_job = job_id
                    try:
                        self._run_job(job_id, request)
                    finally:
                        self._tts_engine.close()
            except JobCancelled:
                self._update_job(job_id, status='cancelled', completed_at=_now_iso(), detail='Cancelled')
            finally:
                self._active_job = None
                self._queue.task_done()

    def cancel(self, job_id):
        with self._lock:
            job = self.get_job(job_id)
            if job.status in {'succeeded', 'failed', 'cancelled'}:
                return job
            self._cancelled.add(job_id)
            return self._update_job(job_id,
                status='cancelling' if self._active_job == job_id else 'cancelled', detail='Cancellation requested')

    def close(self):
        self._shutdown.set()
        for job in self.list_jobs():
            if job.status in {'queued', 'running'}:
                self.cancel(job.id)
        if self._queue_thread:
            self._queue_thread.join(timeout=5)

    def _check_cancelled(self, job_id):
        if self._is_cancelled(job_id):
            raise JobCancelled('Render cancelled')

    def _run_render_command(self, command, *, timeout, **kwargs):
        job_id = self._active_job
        with tempfile.TemporaryFile(mode='w+t') as stdout, tempfile.TemporaryFile(mode='w+t') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **kwargs)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if job_id:
                        self._check_cancelled(job_id)
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    time.sleep(.1)
                stdout.seek(0)
                stderr.seek(0)
                result = subprocess.CompletedProcess(command, process.returncode, stdout.read(), stderr.read())
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
                return result
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def list_jobs(self) -> list[AvatarRenderJob]:
        self._ensure_dirs()
        jobs: list[AvatarRenderJob] = []
        with self._lock:
            known = set(self._jobs)
            jobs.extend(self._jobs.values())
        for manifest in self.outputs_dir.glob("*/job.json"):
            job_id = manifest.parent.name
            if job_id in known:
                continue
            try:
                jobs.append(AvatarRenderJob(**json.loads(manifest.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def get_job(self, job_id: str) -> AvatarRenderJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
        manifest = self._job_dir(job_id) / "job.json"
        if not manifest.exists():
            raise KeyError(job_id)
        job = AvatarRenderJob(**json.loads(manifest.read_text(encoding="utf-8")))
        with self._lock:
            self._jobs[job_id] = job
        return job

    def job_video_path(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        if not job.video_path:
            raise FileNotFoundError(job_id)
        path = Path(job.video_path)
        if not path.exists():
            raise FileNotFoundError(job_id)
        return path

    def job_audio_path(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        if not job.audio_path:
            raise FileNotFoundError(job_id)
        path = Path(job.audio_path)
        if not path.exists():
            raise FileNotFoundError(job_id)
        return path

    def _run_job(self, job_id: str, request: AvatarRenderRequest) -> None:
        try:
            profile = self.get_profile(request.profile_id)
            job_dir = self._job_dir(job_id)
            audio_path = job_dir / "speech.wav"
            video_path = job_dir / "avatar.mp4"

            self._update_job(job_id, status="running", progress=0.08, started_at=_now_iso(), detail="Generating speech")
            samples, sample_rate, speech_detail = self._synthesize_audio(profile, request)
            _write_wav(audio_path, samples, sample_rate)
            self._update_job(
                job_id,
                progress=0.48,
                audio_path=_workspace_path(audio_path),
                audio_url=f"/api/avatar/jobs/{job_id}/audio",
                detail=speech_detail,
            )

            self._update_job(job_id, progress=0.55, detail="Rendering avatar video")
            renderer_detail = self._render_avatar_video(
                profile=profile,
                target_path=Path(profile.target_image_path),
                audio_samples=samples,
                sample_rate=sample_rate,
                fps=request.fps,
                audio_path=audio_path,
                video_path=video_path,
                job_id=job_id,
            )
            self._update_job(
                job_id,
                status="succeeded",
                progress=1.0,
                completed_at=_now_iso(),
                video_path=_workspace_path(video_path),
                video_url=f"/api/avatar/jobs/{job_id}/video",
                detail=f"Render complete; {speech_detail}; {renderer_detail}",
            )
        except JobCancelled:
            self._update_job(job_id, status='cancelled', completed_at=_now_iso(), detail='Cancelled')
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                completed_at=_now_iso(),
                error=str(exc),
                detail="Render failed",
            )

    def _synthesize_audio(
        self,
        profile: AvatarProfile,
        request: AvatarRenderRequest,
    ) -> tuple[np.ndarray, int, str]:
        sample_rate = 48_000
        detail = "Kokoro TTS"
        chunks: list[np.ndarray] = []
        try:
            generator = self._tts_engine.generate(request.text, voice=request.voice, speed=request.speed)
            for source_rate, audio in generator:
                if self._active_job:
                    self._check_cancelled(self._active_job)
                source = _declick_audio_chunk(_mono_float32(audio), source_rate)
                chunks.append(_resample_audio(source, source_rate, sample_rate))
        except Exception as exc:
            detail = f"Fallback TTS: {exc}"
            chunks.clear()
            for source_rate, audio in self._fallback_tts_engine.generate(request.text, voice=request.voice, speed=request.speed):
                source = _declick_audio_chunk(_mono_float32(audio), source_rate)
                chunks.append(_resample_audio(source, source_rate, sample_rate))

        samples = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros(sample_rate, dtype=np.float32)
        samples = _pad_audio(samples, sample_rate)
        if request.synthesis_enabled:
            source_samples = samples
            try:
                converted, synthesis_detail = self._convert_voice(samples, sample_rate, profile.id, request.synthesis_quality)
            except Exception as exc:
                synthesis_detail = f"Voice conversion failed, using original TTS: {exc}"
            else:
                if _rms(source_samples) > 0.001 and _rms(converted) < 0.0005:
                    synthesis_detail = "Voice conversion produced silence, using original TTS"
                else:
                    samples = converted
            detail = f"{detail}; {synthesis_detail}"
        return np.clip(samples, -0.98, 0.98).astype(np.float32, copy=False), sample_rate, detail

    def _convert_voice(
        self,
        samples: np.ndarray,
        sample_rate: int,
        profile_id: str,
        quality: str,
    ) -> tuple[np.ndarray, str]:
        block_size = 512 if quality == "low_latency" else 1024
        processor = SpeechToSpeechProcessor(
            sample_rate,
            VoiceConfig(
                enabled=True,
                preset="monk",
                synthesis_mode="speech_to_speech",
                synthesis_target=profile_id,
                synthesis_quality=quality,
                input_device=None,
                virtual_mic=False,
                monitor_enabled=False,
                monitor_output_device=None,
                monitor_volume=0.7,
                monitor_mode="processed",
                sample_rate=sample_rate,
                block_size=block_size,
                noise_gate=0.0,
                gain=1.0,
            ),
        )
        output_chunks: list[np.ndarray] = []
        try:
            for start in range(0, len(samples), block_size):
                if self._active_job:
                    self._check_cancelled(self._active_job)
                block = samples[start : start + block_size]
                if block.size == 0:
                    continue
                processed, _level = processor.process(block)
                output_chunks.append(processed[: block.size])
            status = processor.status()
            detail = status.detail or "Voice conversion ready"
            if status.fallback:
                detail = f"Voice conversion fallback: {detail}"
            else:
                detail = f"Voice conversion ready: {detail}"
            return np.concatenate(output_chunks).astype(np.float32, copy=False), detail
        finally:
            processor.close()

    def _render_avatar_video(
        self,
        *,
        profile: AvatarProfile,
        target_path: Path,
        audio_samples: np.ndarray,
        sample_rate: int,
        fps: int,
        audio_path: Path,
        video_path: Path,
        job_id: str,
    ) -> str:
        if self.renderer_name == "musetalk":
            return self._render_video_with_musetalk(
                profile=profile,
                target_path=target_path,
                audio_samples=audio_samples,
                sample_rate=sample_rate,
                audio_path=audio_path,
                video_path=video_path,
                job_id=job_id,
            )
        self._render_video_locally(
            target_path=target_path,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            fps=fps,
            audio_path=audio_path,
            video_path=video_path,
            job_id=job_id,
        )
        return "local renderer"

    def _render_video_with_musetalk(
        self,
        *,
        profile: AvatarProfile,
        target_path: Path,
        audio_samples: np.ndarray,
        sample_rate: int,
        audio_path: Path,
        video_path: Path,
        job_id: str,
    ) -> str:
        command_parts = _musetalk_command_parts(self._musetalk_config)
        root = command_parts["root"]
        python = command_parts["python"]
        unet_model = command_parts["unet_model"]
        unet_config = command_parts["unet_config"]
        ffmpeg_path = command_parts["ffmpeg_path"]
        ffmpeg_executable = command_parts["ffmpeg_executable"]
        version = command_parts["version"]

        musetalk_job_dir = self._job_dir(job_id) / "musetalk"
        result_dir = musetalk_job_dir / "results"
        config_path = musetalk_job_dir / "inference.yaml"
        source_video_path = musetalk_job_dir / "source-25fps.mp4"
        command_path = musetalk_job_dir / "command.json"
        stdout_path = musetalk_job_dir / "stdout.log"
        stderr_path = musetalk_job_dir / "stderr.log"
        musetalk_job_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        source_duration = max(1.0, len(audio_samples) / float(sample_rate))
        _create_still_video(
            target_path=target_path,
            output_path=source_video_path,
            duration_seconds=source_duration,
            ffmpeg_executable=ffmpeg_executable,
        )

        config_path.write_text(
            "\n".join(
                [
                    "avatar_0:",
                    f"  video_path: {_yaml_quote(str(source_video_path.resolve()))}",
                    f"  audio_path: {_yaml_quote(str(audio_path.resolve()))}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        command = [
            python,
            "-m",
            "scripts.inference",
            "--inference_config",
            str(config_path.resolve()),
            "--result_dir",
            str(result_dir.resolve()),
            "--unet_model_path",
            str(unet_model.resolve()),
            "--unet_config",
            str(unet_config.resolve()),
            "--version",
            version,
            "--ffmpeg_path",
            ffmpeg_path,
            "--batch_size",
            "1",
            "--use_float16",
        ]
        if self._musetalk_config.bbox_shift:
            command.extend(["--bbox_shift", str(self._musetalk_config.bbox_shift)])

        self._update_job(job_id, progress=0.62, detail="Running MuseTalk lip-sync renderer")
        before = _now_timestamp()
        musetalk_env = os.environ.copy()
        musetalk_env.pop("CUDA_VISIBLE_DEVICES", None)
        musetalk_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        command_path.write_text(
            json.dumps(
                {
                    "cwd": str(root),
                    "command": command,
                    "env": {
                        "PYTORCH_CUDA_ALLOC_CONF": musetalk_env.get("PYTORCH_CUDA_ALLOC_CONF"),
                        "CUDA_VISIBLE_DEVICES": musetalk_env.get("CUDA_VISIBLE_DEVICES"),
                    },
                    "source_video_path": str(source_video_path.resolve()),
                    "audio_path": str(audio_path.resolve()),
                    "profile_id": profile.id,
                    "started_at": _now_iso(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            completed = self._run_render_command(
                command,
                cwd=str(root),
                env=musetalk_env,
                timeout=1800,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"MuseTalk failed: {detail[-1200:] or exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MuseTalk timed out after 30 minutes") from exc

        output = _latest_mp4(result_dir, after_timestamp=before)
        if output is None:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"MuseTalk finished without producing an MP4. {detail[-1200:]}")
        shutil.copy2(output, video_path)
        self._update_job(job_id, progress=0.96, detail="MuseTalk render complete")
        return "MuseTalk renderer"

    def _render_video_locally(
        self,
        *,
        target_path: Path,
        audio_samples: np.ndarray,
        sample_rate: int,
        fps: int,
        audio_path: Path,
        video_path: Path,
        job_id: str,
    ) -> None:
        image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not load avatar target image: {target_path}")
        base = _resize_even(image, max_dimension=720)
        height, width = base.shape[:2]
        face_box = self._detect_face(base) or _fallback_face_box(width, height)
        silent_path = video_path.with_name("avatar-silent.mp4")
        writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 video writer.")

        duration = max(1.0, len(audio_samples) / float(sample_rate))
        frame_count = max(1, int(math.ceil(duration * fps)))
        try:
            for frame_index in range(frame_count):
                self._check_cancelled(job_id)
                start = int(frame_index * sample_rate / fps)
                end = int((frame_index + 1) * sample_rate / fps)
                mouth = _mouth_value(audio_samples[start:end])
                frame = apply_mouth_animation(base.copy(), face_box, mouth)
                writer.write(frame)
                if frame_index % max(1, fps) == 0:
                    self._update_job(job_id, progress=min(0.95, 0.55 + 0.38 * frame_index / frame_count))
        finally:
            writer.release()

        if not _mux_audio(silent_path, audio_path, video_path):
            shutil.copy2(silent_path, video_path)

    def _detect_face(self, frame_bgr: np.ndarray) -> FaceBox | None:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector.detect_largest(frame_bgr)

    def _ensure_dirs(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.voice_presets_dir.mkdir(parents=True, exist_ok=True)

    def _store_job(self, job: AvatarRenderJob) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._write_job(job)

    def _update_job(self, job_id: str, **updates: Any) -> AvatarRenderJob:
        if updates.get('status') not in {'cancelled', 'failed', 'cancelling'}:
            self._check_cancelled(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self.get_job(job_id)
            data = _model_dump(job)
            data.update(updates)
            next_job = AvatarRenderJob(**data)
            self._jobs[job_id] = next_job
            self._write_job(next_job)
            return next_job

    def _write_job(self, job: AvatarRenderJob) -> None:
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        temporary = job_dir / "job.json.tmp"
        temporary.write_text(_model_json(job), encoding="utf-8")
        temporary.replace(job_dir / "job.json")

    def _profile_dir(self, profile_id: str) -> Path:
        if not _SAFE_ID_RE.fullmatch(profile_id):
            raise KeyError(profile_id)
        return self.profiles_dir / profile_id

    def _job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError(job_id)
        return self.outputs_dir / job_id


_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,96}")


def _write_normalized_image(path: Path, payload: bytes) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width < 96 or image.height < 96:
                raise ValueError("Target image must be at least 96x96.")
            image.save(path, format="PNG")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Target image must be a readable image file.") from exc


def _validate_wav_bytes(payload: bytes) -> None:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise ValueError("Voice reference WAV contains no samples.")
            if wav_file.getnchannels() < 1:
                raise ValueError("Voice reference WAV must have at least one channel.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Voice reference must be a WAV file for local voice cloning.") from exc


def _write_voice_reference_wav(path: Path, payload: bytes) -> None:
    try:
        _validate_wav_bytes(payload)
        path.write_bytes(payload)
        return
    except ValueError as wav_error:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("Voice reference must be WAV, or ffmpeg must be installed for browser recordings.") from wav_error

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-f",
        "wav",
        str(path),
    ]
    try:
        subprocess.run(command, input=payload, check=True, timeout=90)
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise ValueError("Converted voice reference contains no samples.")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ValueError("Could not convert voice reference to WAV. Upload a WAV file or record again.") from exc


def _resize_even(frame: np.ndarray, *, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / float(max(width, height)))
    next_width = max(2, int(round(width * scale)) // 2 * 2)
    next_height = max(2, int(round(height * scale)) // 2 * 2)
    if next_width == width and next_height == height:
        return frame
    return cv2.resize(frame, (next_width, next_height), interpolation=cv2.INTER_AREA)


def _fallback_face_box(width: int, height: int) -> FaceBox:
    size = int(min(width, height) * 0.46)
    return FaceBox(x=(width - size) // 2, y=int(height * 0.24), w=size, h=size)


def _mouth_value(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    rms = _rms(samples)
    openness = (rms - 0.008) / 0.085
    return float(np.clip(openness, 0.0, 1.0))


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def _pad_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    silence = np.zeros(int(round(sample_rate * 0.16)), dtype=np.float32)
    return np.concatenate([silence, samples.astype(np.float32, copy=False), silence])


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -0.98, 0.98) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=180)
    except Exception:
        return False
    return output_path.exists()


def _musetalk_command_parts(config: MuseTalkConfig) -> dict[str, Any]:
    if config.root is None:
        raise RuntimeError("Set FACESWAP_MUSETALK_ROOT to the MuseTalk checkout.")
    root = config.root.expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"MuseTalk root does not exist: {root}")
    if not (root / "scripts" / "inference.py").exists():
        raise RuntimeError(f"MuseTalk inference script not found under: {root}")

    python = _musetalk_python(config, root)
    version = _musetalk_version(config.version)
    unet_model = _resolve_musetalk_path(config.unet_model_path, root) or _default_musetalk_unet_model(root, version)
    unet_config = _resolve_musetalk_path(config.unet_config_path, root) or _default_musetalk_unet_config(root, version)
    if not unet_model.exists():
        raise RuntimeError(f"MuseTalk UNet model not found: {unet_model}")
    if not unet_config.exists():
        raise RuntimeError(f"MuseTalk UNet config not found: {unet_config}")

    ffmpeg_path, ffmpeg_executable = _resolve_ffmpeg(config.ffmpeg_path)
    if not ffmpeg_executable:
        raise RuntimeError("Set FACESWAP_MUSETALK_FFMPEG_PATH or install ffmpeg.")

    return {
        "root": root,
        "python": python,
        "version": version,
        "unet_model": unet_model.expanduser().resolve(),
        "unet_config": unet_config.expanduser().resolve(),
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_executable": ffmpeg_executable,
    }


def _musetalk_python(config: MuseTalkConfig, root: Path) -> str:
    if config.python is not None:
        python = config.python.expanduser()
        if not python.exists():
            raise RuntimeError(f"MuseTalk Python does not exist: {python}")
        return str(python.absolute())
    for candidate in (root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate.absolute())
    python_path = shutil.which("python")
    if python_path:
        return python_path
    raise RuntimeError("Set FACESWAP_MUSETALK_PYTHON to the MuseTalk Python executable.")


def _resolve_musetalk_path(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (root / expanded).resolve()


def _resolve_ffmpeg(value: str | None) -> tuple[str | None, str | None]:
    if value:
        path = Path(value).expanduser()
        if path.is_dir():
            executable = path / "ffmpeg"
            return str(path.resolve()), str(executable.resolve()) if executable.exists() else None
        return str(path.resolve()), str(path.resolve()) if path.exists() else None
    executable = shutil.which("ffmpeg")
    return executable, executable


def _musetalk_version(version: str) -> str:
    normalized = version.strip().lower().replace(".", "")
    if normalized in {"v15", "15"}:
        return "v15"
    if normalized in {"v1", "1"}:
        return "v1"
    raise RuntimeError("FACESWAP_MUSETALK_VERSION must be v15 or v1.")


def _default_musetalk_unet_model(root: Path, version: str) -> Path:
    if version == "v15":
        return root / "models" / "musetalkV15" / "unet.pth"
    return root / "models" / "musetalk" / "pytorch_model.bin"


def _default_musetalk_unet_config(root: Path, version: str) -> Path:
    if version == "v15":
        return root / "models" / "musetalkV15" / "musetalk.json"
    return root / "models" / "musetalk" / "musetalk.json"


def _latest_mp4(root: Path, *, after_timestamp: float) -> Path | None:
    candidates = [
        path
        for path in root.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= after_timestamp - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _create_still_video(
    *,
    target_path: Path,
    output_path: Path,
    duration_seconds: float,
    ffmpeg_executable: str,
) -> None:
    command = [
        ffmpeg_executable,
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        str(target_path.resolve()),
        "-t",
        f"{duration_seconds:.3f}",
        "-vf",
        "fps=25,scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(output_path.resolve()),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Could not create MuseTalk source video: {detail[-800:] or exc}") from exc


def _classify_musetalk_error(detail: str) -> str:
    lowered = detail.lower()
    if "musetalk root" in lowered or "faceswap_musetalk_root" in lowered or "inference script" in lowered:
        return "not_installed"
    if "unet model" in lowered or "unet config" in lowered or "not found" in lowered and "models" in lowered:
        return "missing_weights"
    if "python" in lowered or "ffmpeg" in lowered:
        return "env_invalid"
    return "last_error"


def _yaml_quote(value: str) -> str:
    return json.dumps(value)


def _now_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return suffix
    return ".bin"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:42] or "avatar"


def _preview_text(value: str) -> str:
    clean = " ".join(value.strip().split())
    return clean[:117] + "..." if len(clean) > 120 else clean


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_json(model: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
    return model.json(indent=2)
