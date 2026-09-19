export type EffectMode = "passthrough" | "cartoon" | "privacy_blur" | "target_head" | "onnx_faceswap";
export type NeuralProvider = "auto" | "tensorrt" | "cuda" | "cpu";
export type VoicePreset = "monk" | "custom";
export type VoiceMonitorMode = "processed" | "input";
export type VoiceSynthesisMode = "dsp" | "speech_to_speech";
export type VoiceSynthesisQuality = "low_latency" | "balanced" | "quality";

export interface TTSVoiceInfo {
  name: string;
  label: string;
  language: string;
}

export interface DeviceInfo {
  device_id?: string | null;
  index: number;
  label: string;
  kind: "standard" | "infrared" | "virtual";
  is_default: boolean;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
}

export interface AudioDeviceInfo {
  index: number;
  name: string;
  hostapi: string;
  max_input_channels: number;
  max_output_channels: number;
  default_samplerate: number;
}

export interface EffectConfig {
  mode: EffectMode;
  strength: number;
  smoothing: number;
  scale: number;
  y_offset: number;
  mirror: boolean;
  debug: boolean;
  target_image_path?: string | null;
  model_path?: string | null;
  provider: NeuralProvider;
  precision: "fp32" | "fp16";
  edge_feather: number;
  color_match: number;
  sharpen: number;
  temporal_smoothing: number;
  background_enabled: boolean;
  background_path?: string | null;
  background_strength: number;
  background_threshold: number;
  background_smoothing: number;
}

export interface SessionStartRequest {
  source_id?: string | null;
  source_index: number | null;
  width: number;
  height: number;
  fps?: number | null;
  virtual_camera: boolean;
  effect: EffectConfig;
}

export interface SessionEffectUpdateRequest {
  expected_revision?: number;
  effect: EffectConfig;
}

export interface VoiceConfig {
  virtual_output_device?: number | null;
  enabled: boolean;
  preset: VoicePreset;
  synthesis_mode: VoiceSynthesisMode;
  synthesis_target: string;
  synthesis_quality: VoiceSynthesisQuality;
  input_device?: number | null;
  virtual_mic: boolean;
  monitor_enabled: boolean;
  monitor_output_device?: number | null;
  monitor_volume: number;
  monitor_mode: VoiceMonitorMode;
  sample_rate: number;
  block_size: number;
  pitch_semitones: number;
  depth: number;
  reverb: number;
  echo: number;
  lowpass_hz: number;
  noise_gate: number;
  gain: number;
}

export interface VoiceStartRequest {
  voice: VoiceConfig;
}

export interface VoiceMonitorTestRequest {
  output_device?: number | null;
  volume: number;
  duration_ms?: number;
  frequency_hz?: number;
  sample_rate?: number | null;
}

export interface VoiceMonitorTestResponse {
  ok: boolean;
  output_name: string;
  duration_ms: number;
  sample_rate: number;
}

export interface TTSSpeakRequest {
  text: string;
  voice: string;
  speed: number;
  synthesis_target: string;
  synthesis_quality: VoiceSynthesisQuality;
  synthesis_enabled: boolean;
  interrupt: boolean;
  virtual_mic: boolean;
  monitor_enabled: boolean;
  monitor_output_device?: number | null;
  monitor_volume: number;
  sample_rate: number;
  mute_live_mic: boolean;
  lip_sync_enabled: boolean;
  lip_amount: number;
  lip_smoothing: number;
  lip_delay_ms: number;
}

export interface TTSStatus {
  running: boolean;
  active: boolean;
  ready: boolean;
  fallback: boolean;
  engine: string;
  voice: string;
  synthesis_target: string;
  synthesis_quality: string;
  synthesis_ready: boolean;
  synthesis_latency_ms: number;
  synthesis_queue_ms: number;
  synthesis_rt_factor: number;
  synthesis_fallback: boolean;
  synthesis_detail?: string | null;
  text_preview?: string | null;
  sample_rate: number;
  generated_seconds: number;
  playback_seconds: number;
  audio_level: number;
  mouth_open: number;
  lip_sync_enabled: boolean;
  lip_delay_ms: number;
  mute_live_mic: boolean;
  virtual_mic: boolean;
  virtual_mic_ready: boolean;
  monitor_enabled: boolean;
  monitor_ready: boolean;
  monitor_output_device?: number | null;
  monitor_output_name?: string | null;
  last_error?: string | null;
  detail?: string | null;
}

export interface VoiceStatus {
  phase?: "idle" | "starting" | "running" | "stopping" | "error";
  generation?: number;
  active_config?: VoiceConfig | null;
  running: boolean;
  enabled: boolean;
  preset: string;
  synthesis_mode: string;
  synthesis_target: string;
  synthesis_quality: string;
  synthesis_ready: boolean;
  synthesis_latency_ms: number;
  synthesis_queue_ms: number;
  synthesis_rt_factor: number;
  synthesis_fallback: boolean;
  synthesis_detail?: string | null;
  sample_rate: number;
  block_size: number;
  input_device?: number | null;
  input_backend?: string | null;
  input_name?: string | null;
  virtual_mic: boolean;
  virtual_mic_ready: boolean;
  virtual_sink_name: string;
  virtual_source_name: string;
  monitor_enabled: boolean;
  monitor_ready: boolean;
  monitor_output_device?: number | null;
  monitor_output_name?: string | null;
  monitor_volume: number;
  monitor_mode: string;
  frames_processed: number;
  audio_level: number;
  audio_meter_level: number;
  audio_peak_level: number;
  callback_ms: number;
  dropped_blocks: number;
  monitor_dropped_blocks: number;
  last_error?: string | null;
}

export interface VoiceRouteStatus {
  sink_name: string;
  source_name: string;
  sink_present: boolean;
  source_present: boolean;
  pactl_available: boolean;
  pacat_available: boolean;
}

export interface SessionStatus {
  generation?: number;
  active_capture?: Omit<SessionStartRequest, "effect"> | null;
  virtual_camera_state?: string;
  phase?: "idle" | "starting" | "running" | "stopping" | "error";
  effect_revision: number;
  frame_age_ms: number;
  dropped_frames: number;
  running: boolean;
  source_index?: number | null;
  width?: number | null;
  height?: number | null;
  fps_target?: number | null;
  fps_actual: number;
  frames_processed: number;
  preview_ready: boolean;
  virtual_camera: boolean;
  virtual_camera_ready: boolean;
  active_effect: EffectMode;
  last_error?: string | null;
  model_ready: boolean;
  model_provider?: string | null;
  model_latency_ms: number;
  model_detect_ms: number;
  model_swap_ms: number;
  target_ready: boolean;
  face_locked: boolean;
  background_ready: boolean;
  background_latency_ms: number;
  background_segment_ms: number;
  background_error?: string | null;
}

export interface CapabilityStatus {
  target_presets?: string[];
  platform?: string;
  voice_modes?: string[];
  virtual_camera_backend?: string;
  opencv: string;
  cuda_devices: number;
  pyvirtualcam: boolean;
  v4l2loopback_devices: string[];
  models_dir: string;
  assets_dir: string;
  onnxruntime: boolean;
  onnx_providers: string[];
  insightface: boolean;
  tensorrt: boolean;
  mediapipe: boolean;
  sounddevice: boolean;
  portaudio: boolean;
  kokoro: boolean;
  pactl: boolean;
  pacat: boolean;
  pulse_server?: string | null;
  voice_virtual_sink_present: boolean;
  voice_virtual_source_present: boolean;
  default_model_path: string;
  default_model_present: boolean;
  default_target_path: string;
  default_target_present: boolean;
  default_background_path: string;
  default_background_present: boolean;
  background_model_path: string;
  background_model_present: boolean;
  neural_cache_dir: string;
  avatar_renderer: string;
  musetalk_configured: boolean;
  musetalk_status: string;
  musetalk_root?: string | null;
  musetalk_detail?: string | null;
}

export interface AssetUploadResponse {
  path: string;
  width: number;
  height: number;
}

export type AvatarJobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelling" | "cancelled";

export interface AvatarProfile {
  id: string;
  display_name: string;
  target_image_path: string;
  voice_reference_path: string;
  camera_reference_path?: string | null;
  created_at: string;
  consent_confirmed: boolean;
}

export interface AvatarRenderRequest {
  profile_id: string;
  text: string;
  voice: string;
  speed: number;
  synthesis_enabled: boolean;
  synthesis_quality: VoiceSynthesisQuality;
  fps: number;
}

export interface AvatarRenderJob {
  id: string;
  profile_id: string;
  status: AvatarJobStatus;
  progress: number;
  text_preview: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  audio_path?: string | null;
  video_path?: string | null;
  audio_url?: string | null;
  video_url?: string | null;
  error?: string | null;
  detail?: string | null;
}
