from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .swap_backends import backend_catalog, get_backend
from . import __version__


EffectMode = Literal["passthrough", "cartoon", "privacy_blur", "target_head", "onnx_faceswap"]
NeuralProvider = Literal["auto", "tensorrt", "cuda", "cpu"]
VoicePreset = Literal["monk", "custom"]
VoiceMonitorMode = Literal["processed", "input"]
VoiceSynthesisMode = Literal["dsp", "speech_to_speech"]
VoiceSynthesisQuality = Literal["low_latency", "balanced", "quality"]


class EffectConfig(BaseModel):
    swap_backend: str = "inswapper"

    @field_validator("swap_backend")
    @classmethod
    def validate_swap_backend(cls, value: str) -> str:
        get_backend(value)
        return value

    mode: EffectMode = "onnx_faceswap"
    strength: float = Field(default=0.82, ge=0.0, le=1.0)
    smoothing: float = Field(default=0.5, ge=0.0, le=0.98)
    scale: float = Field(default=1.22, ge=0.6, le=2.0)
    y_offset: float = Field(default=0.5, ge=-0.5, le=0.5)
    mirror: bool = True
    debug: bool = False
    target_image_path: str | None = "assets/aging-cyber-monk-target.png"
    model_path: str | None = None
    provider: NeuralProvider = "cuda"
    precision: Literal["fp32", "fp16"] = "fp32"
    edge_feather: float = Field(default=0.35, ge=0.0, le=1.0)
    color_match: float = Field(default=0.25, ge=0.0, le=1.0)
    sharpen: float = Field(default=0.2, ge=0.0, le=1.0)
    temporal_smoothing: float = Field(default=0.4, ge=0.0, le=0.95)
    background_enabled: bool = False
    background_path: str | None = None
    background_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    background_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    background_smoothing: float = Field(default=0.0, ge=0.0, le=0.98)


class SessionStartRequest(BaseModel):
    source_id: str | None = None
    source_index: int | None = Field(default=None, ge=0, le=255)
    width: int = Field(default=640, ge=320, le=3840)
    height: int = Field(default=480, ge=240, le=2160)
    fps: int | None = Field(default=None, ge=5, le=60)
    virtual_camera: bool = False
    effect: EffectConfig = Field(default_factory=EffectConfig)


class SessionEffectUpdateRequest(BaseModel):
    expected_revision: int | None = None
    effect: EffectConfig = Field(default_factory=EffectConfig)


class SetupInstallRequest(BaseModel):
    accept_terms: bool = False
    components: list[Literal["inswapper", "buffalo_l", "background"]] | None = None


class VoiceConfig(BaseModel):
    enabled: bool = True
    preset: VoicePreset = "monk"
    synthesis_mode: VoiceSynthesisMode = "dsp"
    synthesis_target: str = "monk"
    synthesis_quality: VoiceSynthesisQuality = "low_latency"
    input_device: int | None = None
    virtual_output_device: int | None = None
    virtual_mic: bool = True
    monitor_enabled: bool = False
    monitor_output_device: int | None = None
    monitor_volume: float = Field(default=0.7, ge=0.0, le=1.0)
    monitor_mode: VoiceMonitorMode = "processed"
    sample_rate: int = Field(default=48000, ge=16000, le=96000)
    block_size: int = Field(default=512, ge=256, le=4096)
    pitch_semitones: float = Field(default=-7.0, ge=-12.0, le=4.0)
    depth: float = Field(default=1.0, ge=0.0, le=1.0)
    reverb: float = Field(default=0.42, ge=0.0, le=1.0)
    echo: float = Field(default=0.2, ge=0.0, le=1.0)
    lowpass_hz: float = Field(default=2400.0, ge=400.0, le=12000.0)
    noise_gate: float = Field(default=0.004, ge=0.0, le=0.1)
    gain: float = Field(default=1.1, ge=0.1, le=3.0)


class VoiceStartRequest(BaseModel):
    voice: VoiceConfig = Field(default_factory=VoiceConfig)


class VoiceMonitorTestRequest(BaseModel):
    output_device: int | None = None
    volume: float = Field(default=0.7, ge=0.0, le=1.0)
    duration_ms: int = Field(default=650, ge=100, le=3000)
    frequency_hz: float = Field(default=880.0, ge=80.0, le=4000.0)
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)


class VoiceMonitorTestResponse(BaseModel):
    ok: bool = True
    output_name: str
    duration_ms: int
    sample_rate: int


class TTSVoiceInfo(BaseModel):
    name: str
    label: str
    language: str = "en-us"


class TTSSpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    voice: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=1.7)
    synthesis_target: str = "monk"
    synthesis_quality: VoiceSynthesisQuality = "low_latency"
    synthesis_enabled: bool = True
    interrupt: bool = True
    virtual_mic: bool = True
    monitor_enabled: bool = False
    monitor_output_device: int | None = None
    monitor_volume: float = Field(default=0.7, ge=0.0, le=1.0)
    sample_rate: int = Field(default=48000, ge=16000, le=96000)
    mute_live_mic: bool = True
    lip_sync_enabled: bool = True
    lip_amount: float = Field(default=1.0, ge=0.0, le=2.0)
    lip_smoothing: float = Field(default=0.35, ge=0.0, le=0.95)
    lip_delay_ms: int = Field(default=110, ge=-300, le=600)


class TTSStatus(BaseModel):
    phase: Literal["idle", "starting", "running", "stopping", "error"] = "idle"
    generation: int = 0
    running: bool = False
    active: bool = False
    ready: bool = False
    fallback: bool = False
    engine: str = "kokoro"
    voice: str = "af_heart"
    synthesis_target: str = "monk"
    synthesis_quality: str = "low_latency"
    synthesis_ready: bool = False
    synthesis_latency_ms: float = 0.0
    synthesis_queue_ms: float = 0.0
    synthesis_rt_factor: float = 0.0
    synthesis_fallback: bool = False
    synthesis_detail: str | None = None
    text_preview: str | None = None
    sample_rate: int = 48000
    generated_seconds: float = 0.0
    playback_seconds: float = 0.0
    audio_level: float = 0.0
    mouth_open: float = 0.0
    lip_sync_enabled: bool = True
    lip_delay_ms: int = 110
    mute_live_mic: bool = True
    virtual_mic: bool = True
    virtual_mic_ready: bool = False
    monitor_enabled: bool = False
    monitor_ready: bool = False
    monitor_output_device: int | None = None
    monitor_output_name: str | None = None
    last_error: str | None = None
    detail: str | None = None


class VoiceStatus(BaseModel):
    active_config: VoiceConfig | None = None
    phase: Literal["idle", "starting", "running", "stopping", "error"] = "idle"
    generation: int = 0
    running: bool = False
    enabled: bool = False
    preset: str = "monk"
    synthesis_mode: str = "dsp"
    synthesis_target: str = "monk"
    synthesis_quality: str = "low_latency"
    synthesis_ready: bool = False
    synthesis_latency_ms: float = 0.0
    synthesis_queue_ms: float = 0.0
    synthesis_rt_factor: float = 0.0
    synthesis_fallback: bool = False
    synthesis_detail: str | None = None
    sample_rate: int = 48000
    block_size: int = 512
    input_device: int | None = None
    input_backend: str | None = None
    input_name: str | None = None
    virtual_mic: bool = True
    virtual_mic_ready: bool = False
    virtual_sink_name: str = "faceswap_voice_sink"
    virtual_source_name: str = "faceswap_voice_mic"
    monitor_enabled: bool = False
    monitor_ready: bool = False
    monitor_output_device: int | None = None
    monitor_output_name: str | None = None
    monitor_volume: float = 0.7
    monitor_mode: str = "processed"
    frames_processed: int = 0
    audio_level: float = 0.0
    audio_meter_level: float = 0.0
    audio_peak_level: float = 0.0
    callback_ms: float = 0.0
    dropped_blocks: int = 0
    monitor_dropped_blocks: int = 0
    last_error: str | None = None


class VoiceRouteStatus(BaseModel):
    sink_name: str
    source_name: str
    sink_present: bool = False
    source_present: bool = False
    pactl_available: bool = False
    pacat_available: bool = False


class AudioDeviceInfo(BaseModel):
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


class SessionStatus(BaseModel):
    active_capture: dict | None = None
    virtual_camera_state: str = "disabled"
    phase: Literal["idle", "starting", "running", "stopping", "error"] = "idle"
    generation: int = 0
    running: bool
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
    active_effect: EffectMode = "passthrough"
    last_error: str | None = None
    model_ready: bool = False
    model_provider: str | None = None
    model_latency_ms: float = 0.0
    model_detect_ms: float = 0.0
    model_swap_ms: float = 0.0
    model_inference_ms: float = 0.0
    model_controls_ms: float = 0.0
    target_ready: bool = False
    face_locked: bool = False
    background_ready: bool = False
    background_latency_ms: float = 0.0
    background_segment_ms: float = 0.0
    background_error: str | None = None
    capture_backend: str | None = None
    capture_format: str | None = None


class DeviceInfo(BaseModel):
    device_id: str | None = None
    index: int
    label: str
    kind: Literal["standard", "infrared", "virtual"] = "standard"
    is_default: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None


class AssetUploadResponse(BaseModel):
    path: str
    width: int
    height: int


class CapabilityStatus(BaseModel):
    swap_backends: list[dict[str, str]] = Field(default_factory=backend_catalog)
    target_presets: list[str] = []
    platform: str = "Linux"
    virtual_camera_backend: str = "v4l2loopback"
    voice_modes: list[str] = ["dsp", "speech_to_speech"]
    application: str = "mashr-faceswap"
    version: str = __version__
    opencv: str
    cuda_devices: int
    pyvirtualcam: bool
    v4l2loopback_devices: list[str]
    models_dir: str
    assets_dir: str
    onnxruntime: bool
    onnx_providers: list[str]
    insightface: bool
    tensorrt: bool
    mediapipe: bool
    sounddevice: bool
    portaudio: bool
    kokoro: bool
    pactl: bool
    pacat: bool
    pulse_server: str | None = None
    voice_virtual_sink_present: bool
    voice_virtual_source_present: bool
    default_model_path: str
    default_model_present: bool
    default_target_path: str
    default_target_present: bool
    default_background_path: str
    default_background_present: bool
    background_model_path: str
    background_model_present: bool
    neural_cache_dir: str
    avatar_renderer: str = "local"
    musetalk_configured: bool = False
    musetalk_status: str = "not_installed"
    musetalk_root: str | None = None
    musetalk_detail: str | None = None


class AvatarProfile(BaseModel):
    id: str
    display_name: str
    target_image_path: str
    voice_reference_path: str
    camera_reference_path: str | None = None
    created_at: str
    consent_confirmed: bool


class AvatarRenderRequest(BaseModel):
    profile_id: str
    text: str = Field(..., min_length=1, max_length=8000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=1.6)
    synthesis_enabled: bool = True
    synthesis_quality: Literal["low_latency", "balanced", "quality"] = "balanced"
    fps: int = Field(24, ge=12, le=30)


class AvatarRenderJob(BaseModel):
    id: str
    profile_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
    progress: float = 0.0
    text_preview: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    audio_path: str | None = None
    video_path: str | None = None
    audio_url: str | None = None
    video_url: str | None = None
    error: str | None = None
    detail: str | None = None
