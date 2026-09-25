from contextlib import asynccontextmanager
import asyncio
import os
import io
import platform
import tempfile
import uuid
import shutil
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from .lifecycle import SessionBusy
from . import __version__
from PIL import Image

from .camera import list_camera_devices, list_v4l2loopback_devices
from .avatar import AvatarStudio, MuseTalkConfig
from .schemas import (
    AssetUploadResponse,
    AudioDeviceInfo,
    AvatarProfile,
    AvatarRenderJob,
    AvatarRenderRequest,
    CapabilityStatus,
    DeviceInfo,
    SessionEffectUpdateRequest,
    SessionStartRequest,
    SessionStatus,
    SetupInstallRequest,
    TTSSpeakRequest,
    TTSStatus,
    TTSVoiceInfo,
    VoiceMonitorTestRequest,
    VoiceMonitorTestResponse,
    VoiceRouteStatus,
    VoiceStartRequest,
    VoiceStatus,
)
from .model_setup import SetupManager
from .session import VideoSession
from .settings import get_settings, PACKAGED, ROOT
from .tts import TTSSession, kokoro_available
from .voice import (
    VoiceSession,
    ensure_virtual_mic_route,
    get_virtual_mic_route_status,
    list_audio_devices,
    play_monitor_test_tone,
    portaudio_available,
    pulse_server_name,
    sounddevice_available,
)

settings = get_settings()
settings.assets_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.neural_cache_dir.mkdir(parents=True, exist_ok=True)
settings.background_model_path.parent.mkdir(parents=True, exist_ok=True)
setup_manager = SetupManager(settings.models_dir, settings.config_path.parent / "setup.json")

if PACKAGED:
    for source in (ROOT / 'assets').glob('*.png'):
        destination = settings.assets_dir / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
    if not settings.config_path.exists():
        from .schemas import EffectConfig
        settings.config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.config_path.write_text(EffectConfig(
            background_enabled=True,
            background_path="assets/ordo-basement-cyber-warehouse-photoreal.png",
        ).model_dump_json())

session = VideoSession(config_path=settings.config_path)
voice_session = VoiceSession()
tts_session = TTSSession()
avatar_studio = AvatarStudio(
    settings.assets_dir,
    renderer=settings.avatar_renderer,
    musetalk_config=MuseTalkConfig(
        root=settings.musetalk_root,
        python=settings.musetalk_python,
        ffmpeg_path=settings.musetalk_ffmpeg_path,
        version=settings.musetalk_version,
        unet_model_path=settings.musetalk_unet_model_path,
        unet_config_path=settings.musetalk_unet_config_path,
        bbox_shift=settings.musetalk_bbox_shift,
    ),
)
session.set_lip_sync_driver(tts_session.lip_value)
@asynccontextmanager
async def lifespan(_app):
    yield
    await asyncio.gather(*(asyncio.to_thread(worker.stop) for worker in
                           (session, voice_session, tts_session)))
    await asyncio.to_thread(avatar_studio.close)


app = FastAPI(title="FaceSwap Local", version=__version__, lifespan=lifespan)


@app.exception_handler(SessionBusy)
async def session_busy(_request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status", response_model=SessionStatus)
def get_status() -> SessionStatus:
    return session.status()


@app.get("/api/capabilities", response_model=CapabilityStatus)
def get_capabilities() -> CapabilityStatus:
    onnx_providers = _onnx_providers()
    default_model_path = settings.models_dir / "inswapper_128.onnx"
    default_target_path = settings.assets_dir / "aging-cyber-monk-target.png"
    default_background_path = settings.assets_dir / "ordo-basement-cyber-warehouse-photoreal.png"
    voice_route = get_virtual_mic_route_status()
    musetalk_status, musetalk_detail = avatar_studio.musetalk_status()
    return CapabilityStatus(
        target_presets=[str(p.name) for p in settings.assets_dir.glob("*.png") if p.name in {"aging-cyber-monk-target.png", "default-target.png", "optimus.png", "mashr-female.png"}],
        platform=platform.system(),
        virtual_camera_backend="unitycapture" if platform.system() == "Windows" else "v4l2loopback",
        voice_modes=["dsp"] if platform.system() == "Windows" else ["dsp", "speech_to_speech"],
        opencv=cv2.__version__,
        cuda_devices=_opencv_cuda_devices(),
        pyvirtualcam=False,
        v4l2loopback_devices=list_v4l2loopback_devices(),
        models_dir=str(settings.models_dir.resolve()),
        assets_dir=str(settings.assets_dir.resolve()),
        onnxruntime=bool(onnx_providers),
        onnx_providers=onnx_providers,
        insightface=_can_import("insightface"),
        tensorrt=_can_import("tensorrt") or _can_import("tensorrt_lean") or "TensorrtExecutionProvider" in onnx_providers,
        mediapipe=_can_import("mediapipe"),
        sounddevice=sounddevice_available(),
        portaudio=portaudio_available(),
        kokoro=kokoro_available(),
        pactl=shutil.which("pactl") is not None,
        pacat=shutil.which("pacat") is not None,
        pulse_server=pulse_server_name(),
        voice_virtual_sink_present=voice_route.sink_present,
        voice_virtual_source_present=voice_route.source_present,
        default_model_path=str(default_model_path.resolve()),
        default_model_present=default_model_path.exists(),
        default_target_path=str(default_target_path.resolve()),
        default_target_present=default_target_path.exists(),
        default_background_path=str(default_background_path.resolve()),
        default_background_present=default_background_path.exists(),
        background_model_path=str(settings.background_model_path.resolve()),
        background_model_present=settings.background_model_path.exists(),
        neural_cache_dir=str(settings.neural_cache_dir.resolve()),
        avatar_renderer=avatar_studio.renderer_name,
        musetalk_configured=musetalk_status == "ready",
        musetalk_status=musetalk_status,
        musetalk_root=str(settings.musetalk_root.resolve()) if settings.musetalk_root else None,
        musetalk_detail=musetalk_detail,
    )


@app.get("/api/devices", response_model=list[DeviceInfo])
def get_devices() -> list[DeviceInfo]:
    return list_camera_devices()


@app.get("/api/setup")
def get_setup():
    return setup_manager.state()


@app.post("/api/setup/install", status_code=202)
def install_setup(request: SetupInstallRequest):
    try:
        return setup_manager.start(accept_terms=request.accept_terms, component_ids=request.components)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/setup/jobs/{job_id}")
def get_setup_job(job_id: str):
    job = setup_manager.job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Setup job not found")
    return job


@app.delete("/api/setup/jobs/{job_id}")
def cancel_setup_job(job_id: str):
    job = setup_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Setup job not found")
    return job


@app.post("/api/setup/verify")
def verify_setup():
    return setup_manager.verify()


@app.post("/api/setup/complete")
def complete_setup():
    return setup_manager.complete()


@app.get("/api/session/config")
def session_configuration():
    with session._lock:
        return {"effect": session._current_effect(), "revision": session.status().effect_revision}


@app.post("/api/session/start", response_model=SessionStatus)
def start_session(request: SessionStartRequest) -> SessionStatus:
    setup = setup_manager.state()
    missing = []
    by_id = {item["id"]: item for item in setup["components"]}
    effect_was_selected = "effect" in request.model_fields_set
    if effect_was_selected and request.effect.mode == "onnx_faceswap":
        missing.extend(item for item in ("inswapper", "buffalo_l") if not by_id[item]["ready"])
    if effect_was_selected and request.effect.background_enabled and not by_id["background"]["ready"]:
        missing.append("background")
    if missing:
        return JSONResponse(status_code=409, content={"code": "setup_required", "components": missing,
            "message": "Download the selected optional models before using this effect."})
    try:
        return session.start(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/session/effect", response_model=SessionStatus)
def update_session_effect(request: SessionEffectUpdateRequest) -> SessionStatus:
    return session.update_effect(request.effect, request.expected_revision)


@app.post("/api/session/stop", response_model=SessionStatus)
def stop_session() -> SessionStatus:
    return session.stop()


@app.post("/api/session/stop-all")
async def stop_all():
    async def stop_one(name, worker):
        try:
            status = await asyncio.to_thread(worker.stop)
            return name, {"status": status.model_dump(), "error": None}
        except Exception as exc:
            return name, {"status": worker.status().model_dump(), "error": str(exc)}
    return dict(await asyncio.gather(stop_one("video", session), stop_one("voice", voice_session),
                                     stop_one("tts", tts_session)))


@app.get("/api/voice/status", response_model=VoiceStatus)
def get_voice_status() -> VoiceStatus:
    return voice_session.status()


@app.get("/api/voice/devices", response_model=list[AudioDeviceInfo])
def get_audio_devices() -> list[AudioDeviceInfo]:
    return list_audio_devices()


@app.post("/api/voice/start", response_model=VoiceStatus)
def start_voice(request: VoiceStartRequest) -> VoiceStatus:
    return voice_session.start(request.voice)


@app.post("/api/voice/stop", response_model=VoiceStatus)
def stop_voice() -> VoiceStatus:
    return voice_session.stop()


@app.post("/api/voice/monitor/test-tone", response_model=VoiceMonitorTestResponse)
def test_voice_monitor(request: VoiceMonitorTestRequest) -> VoiceMonitorTestResponse:
    try:
        result = play_monitor_test_tone(
            output_device=request.output_device,
            volume=request.volume,
            duration_ms=request.duration_ms,
            frequency_hz=request.frequency_hz,
            sample_rate=request.sample_rate,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VoiceMonitorTestResponse(**result)


@app.get("/api/tts/status", response_model=TTSStatus)
def get_tts_status() -> TTSStatus:
    return tts_session.status()


@app.get("/api/tts/voices", response_model=list[TTSVoiceInfo])
def get_tts_voices() -> list[TTSVoiceInfo]:
    return tts_session.voices()


@app.post("/api/tts/speak", response_model=TTSStatus)
def speak_tts(request: TTSSpeakRequest) -> TTSStatus:
    try:
        return tts_session.speak(request)
    except SessionBusy:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tts/stop", response_model=TTSStatus)
def stop_tts() -> TTSStatus:
    return tts_session.stop()


@app.get("/api/voice/virtual-mic", response_model=VoiceRouteStatus)
def get_voice_route() -> VoiceRouteStatus:
    return get_virtual_mic_route_status()


@app.post("/api/voice/virtual-mic/setup", response_model=VoiceRouteStatus)
def setup_voice_route() -> VoiceRouteStatus:
    return ensure_virtual_mic_route()


@app.post("/api/assets/target", response_model=AssetUploadResponse)
async def upload_target(file: UploadFile = File(...)) -> AssetUploadResponse:
    return await _save_uploaded_image(file, "target", "Target asset")


@app.post("/api/assets/background", response_model=AssetUploadResponse)
async def upload_background(file: UploadFile = File(...)) -> AssetUploadResponse:
    return await _save_uploaded_image(file, "background", "Background asset")


@app.get("/api/avatar/profiles", response_model=list[AvatarProfile])
def list_avatar_profiles() -> list[AvatarProfile]:
    return avatar_studio.list_profiles()


@app.post("/api/avatar/profiles", response_model=AvatarProfile)
async def create_avatar_profile(
    display_name: str = Form(...),
    consent_confirmed: bool = Form(...),
    target_image: UploadFile = File(...),
    voice_reference: UploadFile = File(...),
    camera_reference: UploadFile | None = File(None),
) -> AvatarProfile:
    try:
        return await avatar_studio.create_profile(
            display_name=display_name,
            target_image=target_image,
            voice_reference=voice_reference,
            camera_reference=camera_reference,
            consent_confirmed=consent_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/avatar/jobs", response_model=list[AvatarRenderJob])
def list_avatar_jobs() -> list[AvatarRenderJob]:
    return avatar_studio.list_jobs()


@app.post("/api/avatar/jobs", response_model=AvatarRenderJob)
def create_avatar_job(request: AvatarRenderRequest) -> AvatarRenderJob:
    try:
        return avatar_studio.submit_render(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Avatar profile not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/avatar/jobs/{job_id}", response_model=AvatarRenderJob)
def get_avatar_job(job_id: str) -> AvatarRenderJob:
    try:
        return avatar_studio.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Avatar job not found") from exc


@app.post("/api/avatar/jobs/{job_id}/cancel", response_model=AvatarRenderJob)
def cancel_avatar_job(job_id: str):
    try:
        return avatar_studio.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Avatar job not found") from exc


@app.get("/api/avatar/jobs/{job_id}/video")
def get_avatar_video(job_id: str) -> FileResponse:
    try:
        path = avatar_studio.job_video_path(job_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Avatar video not found") from exc
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/api/avatar/jobs/{job_id}/audio")
def get_avatar_audio(job_id: str) -> FileResponse:
    try:
        path = avatar_studio.job_audio_path(job_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Avatar audio not found") from exc
    return FileResponse(path, media_type="audio/wav", filename=f"{job_id}.wav")


async def _save_uploaded_image(file: UploadFile, basename: str, label: str) -> AssetUploadResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{label} must be an image")

    data = await file.read(15 * 1024 * 1024 + 1)
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"{label} is too large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('Image dimensions exceed limit')
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            normalized = image.convert('RGB')
            encoded = io.BytesIO()
            normalized.save(encoded, format='PNG')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {label.lower()}") from exc
    destination = settings.assets_dir / f"{basename}-{uuid.uuid4().hex}.png"
    fd, temporary = tempfile.mkstemp(dir=settings.assets_dir)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded.getvalue())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return AssetUploadResponse(path=str(destination), width=width, height=height)


@app.get('/api/assets/background/thumbnail')
def current_background_thumbnail():
    selected = session._current_effect().background_path
    if not selected:
        raise HTTPException(status_code=404, detail='No background selected')
    path = Path(selected)
    if not path.is_absolute():
        path = settings.assets_dir.joinpath(*path.parts[1:]) if path.parts[0] == "assets" else settings.project_root / path
    path = path.resolve()
    if not path.is_relative_to(settings.assets_dir.resolve()):
        raise HTTPException(status_code=404, detail='Background preview unavailable')
    try:
        with Image.open(path) as image:
            image.thumbnail((320, 180))
            encoded = io.BytesIO()
            image.convert('RGB').save(encoded, format='JPEG')
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail='Background preview unavailable')
    return Response(content=encoded.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.get("/preview.jpg")
def latest_preview() -> Response:
    frame = session.latest_jpeg()
    if frame is None:
        raise HTTPException(status_code=404, detail="No preview frame is available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/preview.mjpeg")
def preview_mjpeg() -> StreamingResponse:
    return StreamingResponse(
        session.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


def _can_import(module: str) -> bool:
    try:
        __import__(module)
    except Exception:
        return False
    return True


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return []
    try:
        return [str(provider) for provider in ort.get_available_providers()]
    except Exception:
        return []


def _opencv_cuda_devices() -> int:
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


# Mounted after API routes so API requests never fall through to the UI.
if settings.frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run("faceswap.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
