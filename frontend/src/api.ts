import type {
  EffectConfig,
  AssetUploadResponse,
  AvatarProfile,
  AvatarRenderJob,
  AvatarRenderRequest,
  AudioDeviceInfo,
  CapabilityStatus,
  DeviceInfo,
  SessionEffectUpdateRequest,
  SessionStartRequest,
  SessionStatus,
  TTSSpeakRequest,
  TTSStatus,
  TTSVoiceInfo,
  VoiceMonitorTestRequest,
  VoiceMonitorTestResponse,
  VoiceRouteStatus,
  VoiceStartRequest,
  VoiceStatus
  , SetupStatus
  , SetupJob
  , DiagnosticLog
} from "./types";

async function jsonRequest<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: options?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.message ?? (typeof payload.detail === "string" ? payload.detail : payload.detail?.message) ?? detail;
    } catch {
      // Keep HTTP status detail.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  diagnostics: () => jsonRequest<DiagnosticLog>("/api/diagnostics"),
  clearDiagnostics: () => jsonRequest<DiagnosticLog>("/api/diagnostics", { method: "DELETE" }),
  setup: () => jsonRequest<SetupStatus>("/api/setup"),
  installSetup: (acceptTerms: boolean) => jsonRequest<SetupJob>("/api/setup/install", {
    method: "POST", body: JSON.stringify({ accept_terms: acceptTerms })
  }),
  setupJob: (id: string) => jsonRequest<SetupJob>(`/api/setup/jobs/${id}`),
  cancelSetup: (id: string) => jsonRequest<SetupJob>(`/api/setup/jobs/${id}`, { method: "DELETE" }),
  verifySetup: () => jsonRequest<SetupStatus>("/api/setup/verify", { method: "POST" }),
  completeSetup: () => jsonRequest<SetupStatus>("/api/setup/complete", { method: "POST" }),
  effect: () => jsonRequest<{ effect: EffectConfig; revision: number }>("/api/session/config"),
  cancelAvatar: (id: string) => jsonRequest<AvatarRenderJob>(`/api/avatar/jobs/${id}/cancel`, { method: "POST" }),
  status: () => jsonRequest<SessionStatus>("/api/status"),
  voiceStatus: () => jsonRequest<VoiceStatus>("/api/voice/status"),
  ttsStatus: () => jsonRequest<TTSStatus>("/api/tts/status"),
  ttsVoices: () => jsonRequest<TTSVoiceInfo[]>("/api/tts/voices"),
  capabilities: () => jsonRequest<CapabilityStatus>("/api/capabilities"),
  devices: () => jsonRequest<DeviceInfo[]>("/api/devices"),
  audioDevices: () => jsonRequest<AudioDeviceInfo[]>("/api/voice/devices"),
  voiceRoute: () => jsonRequest<VoiceRouteStatus>("/api/voice/virtual-mic"),
  start: (payload: SessionStartRequest) =>
    jsonRequest<SessionStatus>("/api/session/start", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  stopAll: () => jsonRequest<Record<string, { status: SessionStatus | VoiceStatus; error: string | null }>>("/api/session/stop-all", { method: "POST" }),
  stop: () => jsonRequest<SessionStatus>("/api/session/stop", { method: "POST" }),
  updateEffect: (payload: SessionEffectUpdateRequest) =>
    jsonRequest<SessionStatus>("/api/session/effect", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  startVoice: (payload: VoiceStartRequest) =>
    jsonRequest<VoiceStatus>("/api/voice/start", {
      method: "POST",
      body: JSON.stringify(payload)
  }),
  stopVoice: () => jsonRequest<VoiceStatus>("/api/voice/stop", { method: "POST" }),
  speakTts: (payload: TTSSpeakRequest) =>
    jsonRequest<TTSStatus>("/api/tts/speak", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  stopTts: () => jsonRequest<TTSStatus>("/api/tts/stop", { method: "POST" }),
  testMonitorTone: (payload: VoiceMonitorTestRequest) =>
    jsonRequest<VoiceMonitorTestResponse>("/api/voice/monitor/test-tone", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  setupVoiceRoute: () => jsonRequest<VoiceRouteStatus>("/api/voice/virtual-mic/setup", { method: "POST" }),
  avatarProfiles: () => jsonRequest<AvatarProfile[]>("/api/avatar/profiles"),
  createAvatarProfile: (payload: {
    displayName: string;
    targetImage: File;
    voiceReference: File;
    cameraReference?: File | null;
    consentConfirmed: boolean;
  }) => {
    const data = new FormData();
    data.append("display_name", payload.displayName);
    data.append("consent_confirmed", String(payload.consentConfirmed));
    data.append("target_image", payload.targetImage);
    data.append("voice_reference", payload.voiceReference);
    if (payload.cameraReference) {
      data.append("camera_reference", payload.cameraReference);
    }
    return jsonRequest<AvatarProfile>("/api/avatar/profiles", {
      method: "POST",
      body: data
    });
  },
  avatarJobs: () => jsonRequest<AvatarRenderJob[]>("/api/avatar/jobs"),
  createAvatarJob: (payload: AvatarRenderRequest) =>
    jsonRequest<AvatarRenderJob>("/api/avatar/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  avatarJob: (jobId: string) => jsonRequest<AvatarRenderJob>(`/api/avatar/jobs/${jobId}`),
  avatarVideoUrl: (jobId: string) => `/api/avatar/jobs/${jobId}/video`,
  avatarAudioUrl: (jobId: string) => `/api/avatar/jobs/${jobId}/audio`,
  uploadTarget: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return jsonRequest<AssetUploadResponse>("/api/assets/target", {
      method: "POST",
      body: data
    });
  },
  uploadBackground: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return jsonRequest<AssetUploadResponse>("/api/assets/background", {
      method: "POST",
      body: data
    });
  }
};
