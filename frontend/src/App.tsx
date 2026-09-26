import { useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  CircleAlert,
  Cpu,
  Gauge,
  Headphones,
  ImagePlus,
  Mic,
  Pause,
  Play,
  Radio,
  RefreshCcw,
  SlidersHorizontal,
  Video
} from "lucide-react";
import { api } from "./api";
import type {
  AudioDeviceInfo,
  AvatarProfile,
  AvatarRenderJob,
  CapabilityStatus,
  DeviceInfo,
  EffectConfig,
  SessionStatus,
  VoiceConfig,
  VoiceRouteStatus,
  VoiceStatus
} from "./types";

const defaultEffect: EffectConfig = {
  mode: "onnx_faceswap",
  strength: 0.82,
  smoothing: 0.5,
  scale: 1.22,
  y_offset: 0.5,
  mirror: true,
  debug: false,
  target_image_path: "assets/aging-cyber-monk-target.png",
  model_path: null,
  swap_backend: "inswapper",
  provider: "cuda",
  precision: "fp32",
  edge_feather: 0.35,
  color_match: 0.5,
  sharpen: 0.2,
  temporal_smoothing: 0.4,
  background_enabled: true,
  background_path: "assets/ordo-basement-cyber-warehouse-photoreal.png",
  background_strength: 1,
  background_threshold: 1,
  background_smoothing: 0
};

const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;
// Temporarily hide the studio and suspend its background requests.
const AVATAR_STUDIO_ENABLED = false;

const targetOptions: Array<{
  label: string;
  path: string;
  effectOverrides?: Partial<Pick<EffectConfig, "strength" | "scale" | "y_offset" | "background_enabled">>;
}> = [
  { label: "Aging cyber monk", path: "assets/aging-cyber-monk-target.png" },
  { label: "Jack", path: "assets/optimus.png" },
  { label: "Jane", path: "assets/mashr-female.png" },
  { label: "Default target", path: "assets/default-target.png" }
];

const defaultVoice: VoiceConfig = {
  enabled: true,
  preset: "monk",
  synthesis_mode: "dsp",
  synthesis_target: "monk",
  synthesis_quality: "low_latency",
  input_device: null,
  virtual_mic: true,
  monitor_enabled: false,
  monitor_output_device: null,
  monitor_volume: 0.7,
  monitor_mode: "processed",
  sample_rate: 48000,
  block_size: 512,
  pitch_semitones: -7,
  depth: 1,
  reverb: 0.42,
  echo: 0.2,
  lowpass_hz: 2400,
  noise_gate: 0.004,
  gain: 1.1
};

export function App() {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [audioDevices, setAudioDevices] = useState<AudioDeviceInfo[]>([]);
  const [capabilities, setCapabilities] = useState<CapabilityStatus | null>(null);
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus | null>(null);
  const [voiceRoute, setVoiceRoute] = useState<VoiceRouteStatus | null>(null);
  const [avatarProfiles, setAvatarProfiles] = useState<AvatarProfile[]>([]);
  const [avatarJobs, setAvatarJobs] = useState<AvatarRenderJob[]>([]);
  const [effect, setEffect] = useState<EffectConfig>(defaultEffect);
  const [voice, setVoice] = useState<VoiceConfig>(defaultVoice);
  const [sourceIndex, setSourceIndex] = useState<number | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [effectLoaded, setEffectLoaded] = useState(false);
  const [busy, setBusy] = useState<string[]>([]);
  const [backgroundBusy, setBackgroundBusy] = useState(false);
  const [thumbnailVersion, setThumbnailVersion] = useState(0);
  const [thumbnailFailed, setThumbnailFailed] = useState(false);
  const activeCommands = useRef(new Set<string>());
  const commandEpoch = useRef(0);
  const hydratedControls = useRef(false);
  const confirmedBackground = useRef<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [voiceNotice, setVoiceNotice] = useState<string | null>(null);
  const [avatarNotice, setAvatarNotice] = useState<string | null>(null);
  const [toneBusy, setToneBusy] = useState(false);
  const [avatarBusy, setAvatarBusy] = useState(false);
  const [avatarVoiceRecording, setAvatarVoiceRecording] = useState(false);
  const [avatarCameraActive, setAvatarCameraActive] = useState(false);
  const [selectedAvatarProfileId, setSelectedAvatarProfileId] = useState("");
  const [avatarName, setAvatarName] = useState("Monk avatar");
  const [avatarTargetFile, setAvatarTargetFile] = useState<File | null>(null);
  const [avatarVoiceFile, setAvatarVoiceFile] = useState<File | null>(null);
  const [avatarCameraFile, setAvatarCameraFile] = useState<File | null>(null);
  const [avatarConsent, setAvatarConsent] = useState(false);
  const [avatarText, setAvatarText] = useState(
    "Welcome. This avatar was generated from a reference voice and a target image."
  );
  const [previewNonce, setPreviewNonce] = useState(Date.now());
  const [diagnostics, setDiagnostics] = useState<string[]>([]);
  const targetUploadInputRef = useRef<HTMLInputElement | null>(null);
  const backgroundUploadInputRef = useRef<HTMLInputElement | null>(null);
  const avatarVoiceRecorderRef = useRef<MediaRecorder | null>(null);
  const avatarVoiceStreamRef = useRef<MediaStream | null>(null);
  const avatarVoiceChunksRef = useRef<Blob[]>([]);
  const avatarCameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const avatarCameraStreamRef = useRef<MediaStream | null>(null);

  const effectRef = useRef(effect);
  const effectRevision = useRef(0);
  const pendingEffect = useRef<EffectConfig | null>(null);
  const effectTimer = useRef<number | null>(null);
  const sendingEffect = useRef(false);
  const lastEffectSend = useRef(-Infinity);
  const finishEffectPending = useRef(false);

  function scheduleEffect() {
    if (sendingEffect.current || effectTimer.current !== null || !pendingEffect.current) return;
    const delay = finishEffectPending.current ? 0 : Math.max(0, 150 - (performance.now() - lastEffectSend.current));
    effectTimer.current = window.setTimeout(() => {
      effectTimer.current = null;
      void flushEffect();
    }, delay);
  }

  function finishEffect() {
    if (!pendingEffect.current) return;
    finishEffectPending.current = true;
    if (effectTimer.current !== null) window.clearTimeout(effectTimer.current);
    effectTimer.current = null;
    void flushEffect();
  }

  async function flushEffect() {
    if (sendingEffect.current || !pendingEffect.current) return;
    if (effectTimer.current !== null) window.clearTimeout(effectTimer.current);
    effectTimer.current = null;
    finishEffectPending.current = false;
    lastEffectSend.current = performance.now();
    const next = pendingEffect.current;
    pendingEffect.current = null;
    sendingEffect.current = true;
    try {
      const result = await api.updateEffect({ effect: next, expected_revision: effectRevision.current });
      effectRevision.current = result.effect_revision;
      setStatus(result);
      if (confirmedBackground.current !== next.background_path) {
        confirmedBackground.current = next.background_path ?? null;
        setThumbnailVersion(v => v + 1);
        setThumbnailFailed(false);
      }
    } catch (err) {
      pendingEffect.current = null;
      setError(err instanceof Error ? err.message : String(err));
      try {
        const current = await api.effect();
        effectRevision.current = current.revision;
        effectRef.current = current.effect;
        setEffect(current.effect);
      } catch { /* Keep the network error visible until refresh succeeds. */ }
    } finally {
      sendingEffect.current = false;
      scheduleEffect();
    }
  }

  const running = status?.phase === "running";
  const cameraActive = status?.running || busy.includes('camera-start');
  const cameraPhase = !connected ? 'Connection lost — status unknown' : busy.includes('camera-stop') ? 'Stopping' : busy.includes('camera-start') ? 'Starting' : status?.phase ?? 'idle';
  const activeCapture = status?.active_capture;
  const selectedCamera = devices.find(device =>
    sourceId ? device.device_id === sourceId : device.index === sourceIndex);
  const virtualCamera = selectedCamera?.kind === "virtual";
  const cameraPending = Boolean(status?.running && activeCapture && (
    (sourceId ? sourceId !== activeCapture.source_id : sourceIndex !== activeCapture.source_index) ||
    virtualCamera !== activeCapture.virtual_camera));
  const voicePending = Boolean(voiceStatus?.running && voiceStatus.active_config &&
    Object.entries(voiceStatus.active_config).some(([key, value]) =>
      (voice[key as keyof VoiceConfig] ?? null) !== (value ?? null)));
  async function command(key: string, work: () => Promise<void>) {
    if (activeCommands.current.has(key) || activeCommands.current.has('all')) return;
    activeCommands.current.add(key);
    commandEpoch.current += 1;
    setBusy([...activeCommands.current]);
    try { await work(); } finally {
      activeCommands.current.delete(key);
      commandEpoch.current += 1;
      setBusy([...activeCommands.current]);
    }
  }
  const start = () => command('camera-start', performStart);
  const stop = () => command('camera-stop', performStop);
  const startVoice = () => command('voice-start', performStartVoice);
  const stopVoice = () => command('voice-stop', performStopVoice);
  const stopAll = () => command('all', async () => {
    try {
      const results = await api.stopAll();
      setStatus(results.video.status as SessionStatus);
      setVoiceStatus(results.voice.status as VoiceStatus);
      const failures = Object.entries(results).filter(([,r]) => r.error).map(([name,r]) => `${name}: ${r.error}`);
      setError(failures.length ? failures.join('; ') : null);
    } catch (err) { setError(String(err)); }
  });
  const voiceRunning = voiceStatus?.running ?? false;
  const audioInputs = audioDevices.filter((device) => device.max_input_channels > 0);
  const audioOutputs = audioDevices.filter((device) => device.max_output_channels > 0);
  const virtualMicReady =
    voiceRoute?.source_present ?? voiceStatus?.virtual_mic_ready ?? capabilities?.voice_virtual_source_present ?? false;
  const selectedTargetPath = targetOptions.some((option) => option.path === effect.target_image_path)
    ? effect.target_image_path ?? "__custom__"
    : "__custom__";
  const latestAvatarJob = avatarJobs[0] ?? null;
  const previewSrc = useMemo(() => `/preview.mjpeg?t=${previewNonce}`, [previewNonce]);

  function neuralEffect(nextEffect: EffectConfig): EffectConfig {
    // These settings are intentionally fixed rather than exposed as controls.
    return {
      ...nextEffect,
      mode: "onnx_faceswap",
      precision: "fp32",
      color_match: 0.5,
      smoothing: defaultEffect.smoothing,
      y_offset: defaultEffect.y_offset,
      temporal_smoothing: defaultEffect.temporal_smoothing
    };
  }

  function commitEffect(nextEffect: EffectConfig) {
    const normalizedEffect = neuralEffect(nextEffect);
    effectRef.current = normalizedEffect;
    setEffect(normalizedEffect);
    pendingEffect.current = normalizedEffect;
    scheduleEffect();
  }

  function patchEffect(patch: Partial<EffectConfig>) {
    commitEffect({ ...effectRef.current, ...patch });
  }

  function selectTarget(path: string) {
    const target = targetOptions.find((option) => option.path === path);
    commitEffect({
      ...effect,
      ...target?.effectOverrides,
      target_image_path: path
    });
  }

  async function refresh() {
    setError(null);
    try {
      const [
        nextStatus,
        nextCapabilities,
        nextDevices,
        nextVoiceStatus,
        nextAudioDevices,
        nextVoiceRoute,
        nextAvatarProfiles,
        nextAvatarJobs
      ] = await Promise.all([
        api.status(),
        api.capabilities(),
        api.devices(),
        api.voiceStatus(),
        api.audioDevices(),
        api.voiceRoute(),
        AVATAR_STUDIO_ENABLED ? api.avatarProfiles() : Promise.resolve([]),
        AVATAR_STUDIO_ENABLED ? api.avatarJobs() : Promise.resolve([])
      ]);
      setStatus(nextStatus);
      setVoiceStatus(nextVoiceStatus);
      setVoiceRoute(nextVoiceRoute);
      setCapabilities(nextCapabilities);
      setConnected(true);
      if (!hydratedControls.current) {
        const active = nextStatus.active_capture;
        if (active) {
          setSourceIndex(active.source_index);
          setSourceId(active.source_id ?? null);
        }
        if (nextVoiceStatus.active_config) setVoice(nextVoiceStatus.active_config);
        hydratedControls.current = true;
      }
      setDevices(nextDevices);
      setAudioDevices(nextAudioDevices);
      setAvatarProfiles(nextAvatarProfiles);
      setAvatarJobs(nextAvatarJobs);
      setSelectedAvatarProfileId((current) =>
        current && nextAvatarProfiles.some((profile) => profile.id === current) ? current : nextAvatarProfiles[0]?.id ?? ""
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function performStart() {
    setError(null);
    try {
      if (effectTimer.current) window.clearTimeout(effectTimer.current);
      await flushEffect();
      const deadline = Date.now() + 5000;
      while (sendingEffect.current || pendingEffect.current) {
        if (Date.now() > deadline) throw new Error("Settings are still saving; retry Start shortly.");
        await new Promise(resolve => window.setTimeout(resolve, 20));
      }
      const next = await api.start({
        source_index: sourceIndex,
        source_id: sourceId,
        width: CAMERA_WIDTH,
        height: CAMERA_HEIGHT,
        virtual_camera: virtualCamera,
        effect: neuralEffect(effectRef.current)
      });
      effectRevision.current = next.effect_revision;
      setStatus(next);
      setPreviewNonce(Date.now());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function performStop() {
    setError(null);
    try {
      setStatus(await api.stop());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function performStartVoice() {
    setError(null);
    setVoiceNotice(null);
    try {
      setVoiceStatus(await api.startVoice({ voice }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function performStopVoice() {
    setError(null);
    setVoiceNotice(null);
    try {
      setVoiceStatus(await api.stopVoice());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function testMonitorTone() {
    setError(null);
    setVoiceNotice(null);
    setToneBusy(true);
    try {
      const result = await api.testMonitorTone({
        output_device: voice.monitor_output_device,
        volume: voice.monitor_volume,
        duration_ms: 650,
        frequency_hz: 880
      });
      setVoiceNotice(`Test tone played on ${result.output_name}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setToneBusy(false);
    }
  }

  async function uploadTarget(file: File | undefined) {
    if (!file) return;
    setError(null);
    try {
      const asset = await api.uploadTarget(file);
      commitEffect({ ...effectRef.current, target_image_path: asset.path });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function uploadBackground(file: File | undefined) {
    if (!file || activeCommands.current.has('background')) return;
    activeCommands.current.add('background');
    setBackgroundBusy(true);
    setError(null);
    try {
      const asset = await api.uploadBackground(file);
      commitEffect({ ...effectRef.current, background_enabled: true, background_path: asset.path });
      finishEffect();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      activeCommands.current.delete('background');
      setBackgroundBusy(false);
    }
  }

  async function createAvatarProfile() {
    setError(null);
    setAvatarNotice(null);
    if (!avatarTargetFile || !avatarVoiceFile) {
      setError("Avatar Studio needs a target image and a WAV voice reference.");
      return;
    }
    if (!avatarConsent) {
      setError("Confirm consent before creating an avatar profile.");
      return;
    }
    setAvatarBusy(true);
    try {
      const profile = await api.createAvatarProfile({
        displayName: avatarName,
        targetImage: avatarTargetFile,
        voiceReference: avatarVoiceFile,
        cameraReference: avatarCameraFile,
        consentConfirmed: avatarConsent
      });
      setAvatarProfiles((current) => [profile, ...current.filter((item) => item.id !== profile.id)]);
      setSelectedAvatarProfileId(profile.id);
      setAvatarNotice(`Avatar profile created: ${profile.display_name}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAvatarBusy(false);
    }
  }

  async function renderAvatarVideo() {
    setError(null);
    setAvatarNotice(null);
    if (!selectedAvatarProfileId) {
      setError("Create or select an avatar profile first.");
      return;
    }
    if (!avatarText.trim()) {
      setError("Avatar script cannot be empty.");
      return;
    }
    setAvatarBusy(true);
    try {
      const job = await api.createAvatarJob({
        profile_id: selectedAvatarProfileId,
        text: avatarText.trim(),
        voice: "af_heart",
        speed: 1,
        synthesis_enabled: true,
        synthesis_quality: "balanced",
        fps: 24
      });
      setAvatarJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setAvatarNotice("Avatar render queued");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAvatarBusy(false);
    }
  }

  async function startAvatarVoiceRecording() {
    setError(null);
    setAvatarNotice(null);
    if (typeof MediaRecorder === "undefined") {
      setError("This browser does not support MediaRecorder voice capture.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      avatarVoiceChunksRef.current = [];
      avatarVoiceStreamRef.current = stream;
      avatarVoiceRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          avatarVoiceChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(avatarVoiceChunksRef.current, { type });
        const extension = type.includes("ogg") ? "ogg" : type.includes("wav") ? "wav" : "webm";
        setAvatarVoiceFile(new File([blob], `avatar-voice-reference.${extension}`, { type }));
        setAvatarVoiceRecording(false);
        setAvatarNotice("Voice reference recorded");
        stopAvatarVoiceStream();
      };
      recorder.start();
      setAvatarVoiceRecording(true);
    } catch (err) {
      setAvatarVoiceRecording(false);
      stopAvatarVoiceStream();
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function stopAvatarVoiceRecording() {
    const recorder = avatarVoiceRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  function stopAvatarVoiceStream() {
    avatarVoiceStreamRef.current?.getTracks().forEach((track) => track.stop());
    avatarVoiceStreamRef.current = null;
    avatarVoiceRecorderRef.current = null;
  }

  async function startAvatarCameraReference() {
    setError(null);
    setAvatarNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      avatarCameraStreamRef.current = stream;
      if (avatarCameraVideoRef.current) {
        avatarCameraVideoRef.current.srcObject = stream;
      }
      setAvatarCameraActive(true);
    } catch (err) {
      setAvatarCameraActive(false);
      stopAvatarCameraReference();
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function captureAvatarCameraReference() {
    const video = avatarCameraVideoRef.current;
    if (!video) {
      setError("Camera preview is not ready.");
      return;
    }
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      setError("Could not capture camera reference.");
      return;
    }
    context.drawImage(video, 0, 0, width, height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Could not encode camera reference.");
        return;
      }
      setAvatarCameraFile(new File([blob], "avatar-camera-reference.png", { type: "image/png" }));
      setAvatarNotice("Camera reference captured");
      stopAvatarCameraReference();
    }, "image/png");
  }

  function stopAvatarCameraReference() {
    avatarCameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    avatarCameraStreamRef.current = null;
    if (avatarCameraVideoRef.current) {
      avatarCameraVideoRef.current.srcObject = null;
    }
    setAvatarCameraActive(false);
  }

  useEffect(() => {
    void refresh();
    void api.effect().then(current => {
      setEffectLoaded(true);
      if (!pendingEffect.current && !sendingEffect.current) {
        effectRevision.current = current.revision;
        const normalized = neuralEffect(current.effect);
        effectRef.current = normalized;
        setEffect(normalized);
        if (current.effect.mode !== normalized.mode ||
            current.effect.precision !== normalized.precision ||
            current.effect.color_match !== normalized.color_match) {
          pendingEffect.current = normalized;
          scheduleEffect();
        }
      }
    }).catch(err => setError(String(err)));
    const id = window.setInterval(() => {
      if (activeCommands.current.size) return;
      const epoch = commandEpoch.current;
      Promise.all([api.status(), api.voiceStatus(), AVATAR_STUDIO_ENABLED ? api.avatarJobs() : Promise.resolve([]), api.diagnostics()])
        .then(([nextStatus, nextVoiceStatus, nextAvatarJobs, nextDiagnostics]) => {
          if (epoch !== commandEpoch.current) return;
          setConnected(true);
          setStatus(nextStatus);
          setVoiceStatus(nextVoiceStatus);
          setAvatarJobs(nextAvatarJobs);
          setDiagnostics(nextDiagnostics.entries);
        })
        .catch((err) => { setConnected(false); setError(err instanceof Error ? err.message : String(err)); });
    }, 1000);
    return () => {
      window.clearInterval(id);
      if (effectTimer.current) window.clearTimeout(effectTimer.current);
      stopAvatarVoiceStream();
      stopAvatarCameraReference();
    };
  }, []);

  return (
    <main className="app-shell">
      <section className="preview-stage" aria-label="Live preview">
        {running && status?.preview_ready ? (
          <img className="preview-frame" src={previewSrc} alt="Live processed camera preview" />
        ) : (
          <div className="idle-preview">
            <Video size={58} strokeWidth={1.5} />
            <span>{cameraPhase}</span>
          </div>
        )}
          {(error || status?.last_error || voiceStatus?.last_error) && (
            <div className="error-banner">
              <CircleAlert size={18} />
              <span>{error ?? status?.last_error ?? voiceStatus?.last_error}</span>
            </div>
          )}
          {(voiceNotice || avatarNotice) && !error && !status?.last_error && !voiceStatus?.last_error && (
            <div className="notice-banner">
              <CheckCircle2 size={18} />
              <span>{voiceNotice ?? avatarNotice}</span>
            </div>
          )}
      </section>

      <aside className="control-rail">
        <header className="rail-header">
          <div>
            <h1>FaceSwap</h1>
            <p>Local live filter</p>
          </div>
          <button className="icon-button" onClick={() => void refresh()} title="Refresh devices">
            <RefreshCcw size={18} />
          </button>
        </header>

        <div className="persistent-actions">
          <div className="action-row action-row-top">
            {cameraActive ? (
              <button className="primary-button stop" onClick={() => void stop()} disabled={busy.includes('camera-stop') || status?.phase === 'stopping'}>
                <Pause size={18} /> Stop camera
              </button>
            ) : (
              <button className="primary-button" onClick={() => void start()} disabled={!connected || !effectLoaded || !hydratedControls.current || busy.includes('all')}>
                <Play size={18} /> Start camera
              </button>
            )}
            <button className="secondary-button stop" onClick={() => void stopAll()} disabled={!connected || busy.includes('all')}>Stop all</button>
          </div>
          <div className="rail-status" role="status">
            <StatusPill active={running && connected} text={cameraPhase} />
            <StatusPill active={voiceRunning && connected} text={`Voice: ${connected ? voiceStatus?.phase ?? 'idle' : 'unknown'}`} />
          </div>
          {(error || status?.last_error || voiceStatus?.last_error) && <p className="control-error" role="alert">{error ?? status?.last_error ?? voiceStatus?.last_error}</p>}
        </div>
        <div className="control-tree">
        <ControlGroup title="Camera" icon={<Camera size={18} />}>
          {cameraPending && <div className="pending-settings"><p>Changes apply on restart</p><button className="secondary-button" onClick={() => void start()} disabled={busy.includes('camera-start') || status?.phase === 'stopping'}>Apply &amp; restart camera</button></div>}
          {status?.virtual_camera_state === 'waiting_for_receiver' && <p className="section-help">Waiting for receiving application</p>}
          <label className="field-label" htmlFor="camera">
            <Camera size={16} />
            Camera (needs restart)
          </label>
          <select
            id="camera"
            value={sourceIndex ?? "auto"}
            onChange={(event) => {
              const index = event.target.value === 'auto' ? null : Number(event.target.value);
              setSourceIndex(index);
              setSourceId(devices.find(d => d.index === index)?.device_id ?? null);
            }}
          >
            <option value="auto">Automatic (standard webcam)</option>
            {sourceIndex !== null && !devices.some((device) => device.index === sourceIndex) && (
              <option value={sourceIndex} disabled>
                Selected camera unavailable ({sourceIndex})
              </option>
            )}
            {devices.map((device) => (
              <option key={device.index} value={device.index} title={device.label}>
                {device.label}
                {device.is_default ? " — default" : ""}
              </option>
            ))}
          </select>
          {devices.length === 0 && <p className="section-help">No camera detected. Connect a webcam and refresh devices.</p>}

        </ControlGroup>

        <ControlGroup title="Face swap" icon={<SlidersHorizontal size={18} />} defaultOpen>
          <label className="model-path-field" htmlFor="target-image">
            <span>Target</span>
            <select
              id="target-image"
              value={selectedTargetPath}
              onChange={(event) => {
                if (event.target.value === "__custom__") return;
                if (event.target.value === "__upload__") {
                  targetUploadInputRef.current?.click();
                  return;
                }
                selectTarget(event.target.value);
              }}
            >
              {selectedTargetPath === "__custom__" && <option value="__custom__">Uploaded/custom target</option>}
              {targetOptions.filter(option => !capabilities?.target_presets || capabilities.target_presets.includes(option.path.split("/").pop()!)).map((option) => (
                <option key={option.path} value={option.path}>
                  {option.label}
                </option>
              ))}
              <option value="__upload__">Upload target</option>
            </select>
          </label>
          <input
            ref={targetUploadInputRef}
            className="hidden-file-input"
            type="file"
            accept="image/*"
            onChange={(event) => {
              void uploadTarget(event.target.files?.[0]);
              event.currentTarget.value = "";
            }}
          />
          <div className="transformation-control">
            <Slider
              label="Transformation"
              min={0}
              max={100}
              step={1}
              value={Math.round(effect.strength * 100)}
              onChange={(value) => patchEffect({ strength: value / 100 })}
              onCommit={finishEffect}
              formatValue={(value) => `${value}%`}
            />
            <div className="transformation-endpoints">
              <span>0% Original face</span>
              <span>100% Full swap</span>
            </div>
          </div>

          <details className="settings-tree">
            <summary>Blending & detail</summary>
            <div className="settings-tree-body">
              <Slider
                label="Edge blend"
                min={0}
                max={1}
                step={0.01}
                value={effect.edge_feather}
                onChange={(edge_feather) => patchEffect({ edge_feather })}
              />
              <Slider
                label="Detail"
                min={0}
                max={1}
                step={0.01}
                value={effect.sharpen}
                onChange={(sharpen) => patchEffect({ sharpen })}
              />
            </div>
          </details>
          <details className="settings-tree">
            <summary>Placement</summary>
            <div className="settings-tree-body">
              <Slider
                label="Scale"
                min={0.6}
                max={2}
                step={0.01}
                value={effect.scale}
                onChange={(scale) => patchEffect({ scale })}
              />
            </div>
          </details>
        </ControlGroup>

        <ControlGroup title="Background" icon={<ImagePlus size={18} />}>
          <div className="background-options">
            {effect.background_path && !thumbnailFailed && <img className="background-thumbnail" alt="Selected background" src={`/api/assets/background/thumbnail?v=${thumbnailVersion}`} onError={() => setThumbnailFailed(true)} />}
            <p className="section-help">{effect.background_path?.split(/[\\/]/).pop() ?? 'No background selected'}</p>
            <label className="toggle-inline"><input type="checkbox" checked={effect.background_enabled} disabled={!effect.background_path || backgroundBusy} onChange={e => patchEffect({background_enabled: e.target.checked})} /> Enabled</label>
            <button className="secondary-button" disabled={backgroundBusy} onClick={() => backgroundUploadInputRef.current?.click()}>{backgroundBusy ? 'Uploading…' : effect.background_path ? 'Change image' : 'Choose image'}</button>
            {effect.background_path && <button className="secondary-button" disabled={backgroundBusy} onClick={() => patchEffect({background_enabled: false, background_path: null})}>Remove image</button>}
            <input
              ref={backgroundUploadInputRef}
              className="hidden-file-input"
              type="file"
              accept="image/*"
              onChange={(event) => {
                void uploadBackground(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </div>
        </ControlGroup>

        <ControlGroup title="Voice" icon={<Mic size={18} />}>
          {voicePending && <div className="pending-settings"><p>Changes apply on restart</p><button className="secondary-button" onClick={() => void startVoice()} disabled={busy.includes('voice-start') || voiceStatus?.phase === 'stopping'}>Apply &amp; restart voice</button></div>}
          <div className="field-grid">
            {voiceStatus?.running || busy.includes("voice-start") ? (
              <button className="secondary-button stop" type="button" onClick={() => void stopVoice()} disabled={busy.includes("voice-stop") || voiceStatus?.phase === "stopping"}>
                <Pause size={16} />
                Stop voice
              </button>
            ) : (
              <button className="secondary-button" type="button" onClick={() => void startVoice()} disabled={!voice.enabled || !connected || busy.includes("voice-start") || busy.includes("all")}>
                <Play size={16} />
                Start voice
              </button>
            )}
          </div>

          <details className="settings-tree">
            <summary>Input &amp; voice effect (needs restart)</summary>
            <div className="settings-tree-body">
              <label className="toggle-inline">
                <input
                  type="checkbox"
                  checked={voice.enabled}
                  onChange={(event) => setVoice((current) => ({ ...current, enabled: event.target.checked }))}
                />
                Voice enabled
              </label>
              <div className="field-grid">
                <label>
                  <span>Mode</span>
                  <select
                    id="voice-preset"
                    value={voice.synthesis_mode}
                    onChange={(event) =>
                      setVoice((current) => ({
                        ...current,
                        synthesis_mode: event.target.value as VoiceConfig["synthesis_mode"],
                      }))
                    }
                  >
                    <option value="dsp">DSP</option>
                    <option disabled={capabilities?.platform === "Windows"} value="speech_to_speech">Speech synthesis</option>
                  </select>
                </label>
                <label>
                  <span>Input</span>
                  <select
                    value={voice.input_device ?? ""}
                    onChange={(event) =>
                      setVoice((current) => ({
                        ...current,
                        input_device: event.target.value === "" ? null : Number(event.target.value),
                      }))
                    }
                  >
                    <option value="">Default</option>
                    {audioInputs.map((device) => (
                      <option key={device.index} value={device.index}>
                        {device.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {voice.synthesis_mode === "dsp" ? (
                <label className="model-path-field">
                  <span>Preset</span>
                  <select
                    value={voice.preset}
                    onChange={(event) =>
                      setVoice((current) => ({ ...current, preset: event.target.value as VoiceConfig["preset"] }))
                    }
                  >
                    <option value="monk">Monk</option>
                    <option value="custom">Custom</option>
                  </select>
                </label>
              ) : (
                <div className="field-grid">
                  <label>
                    <span>Target</span>
                    <select
                      value={voice.synthesis_target}
                      onChange={(event) => setVoice((current) => ({ ...current, synthesis_target: event.target.value }))}
                    >
                      <option value="monk">Monk synth</option>
                      <option value="shadow">Shadow synth</option>
                      <option value="bright">Bright synth</option>
                    </select>
                  </label>
                  <label>
                    <span>Latency</span>
                    <select
                      value={voice.synthesis_quality}
                      onChange={(event) =>
                        setVoice((current) => ({
                          ...current,
                          synthesis_quality: event.target.value as VoiceConfig["synthesis_quality"],
                          block_size: event.target.value === "low_latency" ? 512 : 1024,
                        }))
                      }
                    >
                      <option value="low_latency">Low</option>
                      <option value="balanced">Balanced</option>
                      <option value="quality">Quality</option>
                    </select>
                  </label>
                </div>
              )}

              <LevelMeter
                level={voiceStatus?.audio_meter_level ?? voiceStatus?.audio_level ?? 0}
                peak={voiceStatus?.audio_peak_level ?? 0}
                active={voiceRunning}
              />
            </div>
          </details>
          <details className="settings-tree">
            <summary>Virtual microphone (needs restart)</summary>
            <div className="settings-tree-body">
              {capabilities?.platform === 'Windows' && <>
                <p className="section-help">Install VB-CABLE separately. Send to CABLE Input here; select CABLE Output in your call app.</p>
                <label className="model-path-field"><span>Virtual voice output</span><select aria-label="Virtual voice output" value={voice.virtual_output_device ?? ''} onChange={e => setVoice(current => ({...current, virtual_output_device: e.target.value === '' ? null : Number(e.target.value)}))}>
                  <option value="">Select CABLE Input</option>
                  {audioOutputs.filter(d => /cable input/i.test(d.name)).map(d => <option key={d.index} value={d.index}>{d.name} ({d.hostapi})</option>)}
                </select></label>
              </>}
              <label className="toggle-inline">
                <input
                  type="checkbox"
                  checked={voice.virtual_mic}
                  onChange={(event) => setVoice((current) => ({ ...current, virtual_mic: event.target.checked }))}
                />
                <Radio size={16} />
                Virtual mic
              </label>
              <div className={virtualMicReady ? "route-status ready" : "route-status"}>
                {virtualMicReady ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
                <span>
                  {virtualMicReady ? (voiceRoute?.source_name ?? voiceStatus?.virtual_source_name) : "Mic route missing"}
                </span>
              </div>
            </div>
          </details>
          <details className="settings-tree">
            <summary>Headphone monitor (needs restart)</summary>
            <div className="settings-tree-body">
              <label className="toggle-inline">
                <input
                  type="checkbox"
                  checked={voice.monitor_enabled}
                  onChange={(event) =>
                    setVoice((current) => ({
                      ...current,
                      monitor_enabled: event.target.checked,
                    }))
                  }
                />
                <Headphones size={16} />
                Headphone monitor
              </label>

              {voice.monitor_enabled && (
                <>
                  <label className="model-path-field">
                    <span>Output</span>
                    <select
                      value={voice.monitor_output_device ?? ""}
                      onChange={(event) =>
                        setVoice((current) => ({
                          ...current,
                          monitor_output_device: event.target.value === "" ? null : Number(event.target.value),
                        }))
                      }
                    >
                      <option value="">Default output</option>
                      {audioOutputs.map((device) => (
                        <option key={device.index} value={device.index}>
                          {device.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Slider
                    label="Monitor"
                    min={0}
                    max={1}
                    step={0.01}
                    value={voice.monitor_volume}
                    onChange={(monitor_volume) => setVoice((current) => ({ ...current, monitor_volume }))}
                  />
                  <div className="segmented-control" aria-label="Headphone monitor source">
                    <button
                      type="button"
                      className={voice.monitor_mode === "processed" ? "active" : ""}
                      onClick={() => setVoice((current) => ({ ...current, monitor_mode: "processed" }))}
                    >
                      Modified
                    </button>
                    <button
                      type="button"
                      className={voice.monitor_mode === "input" ? "active" : ""}
                      onClick={() => setVoice((current) => ({ ...current, monitor_mode: "input" }))}
                    >
                      Raw mic
                    </button>
                  </div>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void testMonitorTone()}
                    disabled={toneBusy}
                  >
                    <Headphones size={16} />
                    {toneBusy ? "Playing tone" : "Test tone"}
                  </button>
                </>
              )}
            </div>
          </details>
          {voice.synthesis_mode === "dsp" && (
            <details className="settings-tree">
              <summary>
                <SlidersHorizontal size={16} />
                <span>DSP modifiers (needs restart)</span>
              </summary>
              <div className="settings-tree-body">
                <Slider
                  label="Pitch"
                  min={-12}
                  max={4}
                  step={0.25}
                  value={voice.pitch_semitones}
                  onChange={(pitch_semitones) => setVoice((current) => ({ ...current, pitch_semitones }))}
                />
                <Slider
                  label="Depth"
                  min={0}
                  max={1}
                  step={0.01}
                  value={voice.depth}
                  onChange={(depth) => setVoice((current) => ({ ...current, depth }))}
                />
                <Slider
                  label="Reverb"
                  min={0}
                  max={1}
                  step={0.01}
                  value={voice.reverb}
                  onChange={(reverb) => setVoice((current) => ({ ...current, reverb }))}
                />
                <Slider
                  label="Echo"
                  min={0}
                  max={1}
                  step={0.01}
                  value={voice.echo}
                  onChange={(echo) => setVoice((current) => ({ ...current, echo }))}
                />
              </div>
            </details>
          )}
        </ControlGroup>

        <ControlGroup title="Performance" icon={<Cpu size={18} />}>
          <div className="neural-options">
            <label className="field-label" htmlFor="swap-backend">Face-swap engine</label>
            <select
              id="swap-backend"
              value={effect.swap_backend ?? "inswapper"}
              onChange={(event) => patchEffect({ swap_backend: event.target.value, model_path: null })}
            >
              {(capabilities?.swap_backends ?? [{ id: "inswapper", label: "InSwapper" }]).map((backend) => (
                <option key={backend.id} value={backend.id}>{backend.label}</option>
              ))}
            </select>
            <label className="field-label" htmlFor="provider">
              <Cpu size={16} />
              Provider
            </label>
            <select
              id="provider"
              value={effect.provider}
              onChange={(event) => patchEffect({ provider: event.target.value as EffectConfig["provider"] })}
            >
              <option value="auto">Auto</option>
              {capabilities?.platform !== "Windows" && <option value="tensorrt">TensorRT</option>}
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
        </ControlGroup>

        {AVATAR_STUDIO_ENABLED && <div className="control-section avatar-studio">
          <label className="field-label" htmlFor="avatar-profile">
            <Video size={16} />
            Avatar Studio
          </label>
          <p className="section-help">
            Local v1: enroll a consented target image and WAV voice reference, then render a scripted talking-head MP4.
          </p>

          <label className="model-path-field" htmlFor="avatar-profile">
            <span>Active profile</span>
            <select
              id="avatar-profile"
              value={selectedAvatarProfileId}
              onChange={(event) => setSelectedAvatarProfileId(event.target.value)}
            >
              <option value="">No profile</option>
              {avatarProfiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.display_name}
                </option>
              ))}
            </select>
          </label>

          <label className="model-path-field" htmlFor="avatar-name">
            <span>Profile name</span>
            <input id="avatar-name" type="text" value={avatarName} onChange={(event) => setAvatarName(event.target.value)} />
          </label>

          <div className="field-grid">
            <label className="file-field">
              <span>Target image</span>
              <input
                type="file"
                accept="image/*"
                onChange={(event) => setAvatarTargetFile(event.currentTarget.files?.[0] ?? null)}
              />
            </label>
            <label className="file-field">
              <span>Voice reference</span>
              <input
                type="file"
                accept="audio/*,.wav,.webm,.ogg,.m4a,.mp3"
                onChange={(event) => setAvatarVoiceFile(event.currentTarget.files?.[0] ?? null)}
              />
            </label>
          </div>

          <div className="field-grid">
            <button
              className={avatarVoiceRecording ? "secondary-button stop" : "secondary-button"}
              type="button"
              onClick={() => (avatarVoiceRecording ? stopAvatarVoiceRecording() : void startAvatarVoiceRecording())}
            >
              {avatarVoiceRecording ? "Stop voice recording" : "Record voice"}
            </button>
            <div className="avatar-job-detail">{avatarVoiceFile ? avatarVoiceFile.name : "No voice reference selected"}</div>
          </div>

          <label className="file-field">
            <span>Camera reference optional</span>
            <input
              type="file"
              accept="image/*,video/*"
              onChange={(event) => setAvatarCameraFile(event.currentTarget.files?.[0] ?? null)}
            />
          </label>

          <div className="field-grid">
            <button
              className="secondary-button"
              type="button"
              onClick={() => (avatarCameraActive ? captureAvatarCameraReference() : void startAvatarCameraReference())}
            >
              {avatarCameraActive ? "Capture photo" : "Open camera"}
            </button>
            <button className="secondary-button" type="button" onClick={stopAvatarCameraReference} disabled={!avatarCameraActive}>
              Close camera
            </button>
          </div>
          <video
            ref={avatarCameraVideoRef}
            className={avatarCameraActive ? "avatar-camera-preview" : "avatar-camera-preview hidden"}
            autoPlay
            playsInline
            muted
          />
          <div className="avatar-job-detail">{avatarCameraFile ? avatarCameraFile.name : "No camera reference captured"}</div>

          <label className="toggle-inline avatar-consent">
            <input type="checkbox" checked={avatarConsent} onChange={(event) => setAvatarConsent(event.target.checked)} />
            I have permission to use this person&apos;s likeness and voice.
          </label>

          <button
            className="secondary-button"
            type="button"
            onClick={() => void createAvatarProfile()}
            disabled={avatarBusy || !avatarTargetFile || !avatarVoiceFile || !avatarConsent}
          >
            Create profile
          </button>

          <label className="model-path-field" htmlFor="avatar-script">
            <span>Script</span>
            <textarea id="avatar-script" value={avatarText} onChange={(event) => setAvatarText(event.target.value)} />
          </label>

          <button
            className="primary-button"
            type="button"
            onClick={() => void renderAvatarVideo()}
            disabled={avatarBusy || !selectedAvatarProfileId || !avatarText.trim()}
          >
            Render avatar MP4
          </button>

          {avatarJobs.filter(job => ["queued", "running", "cancelling"].includes(job.status)).map(job => (
            <div key={job.id} className="avatar-job-card">
              <span>{job.status}: {job.text_preview}</span>
              <button disabled={job.status === "cancelling"} onClick={() => void api.cancelAvatar(job.id)
                .then(() => api.avatarJobs()).then(setAvatarJobs).catch(err => setError(String(err)))}>Cancel</button>
            </div>
          ))}
          {latestAvatarJob && (
            <div className={latestAvatarJob.status === "succeeded" ? "avatar-job-card ready" : "avatar-job-card"}>
              <div className={latestAvatarJob.status === "succeeded" ? "tts-status active" : "tts-status"}>
                <span>{latestAvatarJob.status}</span>
                <strong>{Math.round((latestAvatarJob.progress ?? 0) * 100)}%</strong>
              </div>
              {latestAvatarJob.detail && <div className="avatar-job-detail">{latestAvatarJob.detail}</div>}
              {latestAvatarJob.error && <div className="tts-detail">{latestAvatarJob.error}</div>}
              {latestAvatarJob.status === "succeeded" && latestAvatarJob.video_url && (
                <video className="avatar-video" controls src={`${latestAvatarJob.video_url}?t=${latestAvatarJob.completed_at ?? latestAvatarJob.id}`} />
              )}
            </div>
          )}
        </div>}

        <footer className="runtime-panel" aria-label="Runtime metadata">
          <details className="settings-tree control-group">
            <summary><Cpu size={16} /><span>Metadata</span></summary>
            <div className="settings-tree-body">
              <details className="settings-tree">
                <summary>Video</summary>
                <div className="settings-tree-body">
                  <Metric icon={<Gauge size={16} />} label="FPS" value={(status?.fps_actual ?? 0).toFixed(1)} />
                  <Metric label="Frames" value={String(status?.frames_processed ?? 0)} />
                  <Metric label="Frame age" value={`${(status?.frame_age_ms ?? 0).toFixed(0)} ms`} />
                  <Metric label="Dropped frames" value={String(status?.dropped_frames ?? 0)} />
                  <Metric label="Provider" value={status?.model_provider ?? "-"} />
                  <Metric label="Precision" value="FP32" />
                  <Metric label="Camera" value={selectedCamera?.label ?? "Automatic standard webcam"} />
                  <Metric label="Camera state" value={running && connected ? "live" : cameraPhase} />
                  <Metric label="Virtual camera" value={status?.virtual_camera_ready ? "ready" : virtualCamera ? "selected" : "off"} />
                  <Metric label="Target" value={(status?.target_ready ?? Boolean(effect.target_image_path)) ? "ready" : "missing"} />
                  <Metric label="Background" value={(status?.background_ready ?? effect.background_enabled) ? "ready" : "off"} />
                  <Metric label="Neural model" value={status?.model_ready ? "ready" : "not ready"} />
                  <Metric label="Face lock" value={status?.face_locked ? "locked" : "searching"} />
                </div>
              </details>
              <details className="settings-tree">
                <summary>Processing</summary>
                <div className="settings-tree-body">
                  <Metric label="Neural ms" value={(status?.model_latency_ms ?? 0).toFixed(1)} />
                  <Metric label="Detect ms" value={(status?.model_detect_ms ?? 0).toFixed(1)} />
                  <Metric label="Swap ms" value={(status?.model_swap_ms ?? 0).toFixed(1)} />
                  <Metric label="Inference ms" value={(status?.model_inference_ms ?? 0).toFixed(1)} />
                  <Metric label="Controls ms" value={(status?.model_controls_ms ?? 0).toFixed(1)} />
                  <Metric label="Bg ms" value={(status?.background_latency_ms ?? 0).toFixed(1)} />
                  <Metric label="Mask ms" value={(status?.background_segment_ms ?? 0).toFixed(1)} />
                </div>
              </details>
              <details className="settings-tree">
                <summary>Audio</summary>
                <div className="settings-tree-body">
                  <Metric label="Voice" value={voiceRunning ? "live" : "idle"} />
                  <Metric label="Voice mic" value={virtualMicReady ? voiceRoute?.source_name ?? voiceStatus?.virtual_source_name ?? "ready" : "not routed"} />
                  <Metric label="Input" value={voiceStatus?.input_backend ? `${voiceStatus.input_backend}: ${voiceStatus.input_name ?? "-"}` : "-"} />
                  <Metric label="Monitor" value={voiceStatus?.monitor_ready ? voiceStatus.monitor_output_name ?? "ready" : voice.monitor_enabled ? "armed" : "off"} />
                  <Metric label="Monitor mode" value={voiceStatus?.monitor_mode === "input" ? "raw mic" : "modified"} />
                  <Metric label="Voice level" value={(voiceStatus?.audio_level ?? 0).toFixed(3)} />
                  <Metric label="Voice ms" value={(voiceStatus?.callback_ms ?? 0).toFixed(1)} />
                  <Metric label="Synth" value={voiceStatus?.synthesis_mode === "speech_to_speech" ? voiceStatus.synthesis_fallback ? "fallback" : "ready" : "dsp"} />
                  <Metric label="Synth ms" value={(voiceStatus?.synthesis_latency_ms ?? 0).toFixed(1)} />
                  <Metric label="Synth RT" value={(voiceStatus?.synthesis_rt_factor ?? 0).toFixed(2)} />
                </div>
              </details>
              <details className="settings-tree">
                <summary>Runtime &amp; devices</summary>
                <div className="settings-tree-body">
                  <Metric label="OpenCV" value={capabilities?.opencv ?? "-"} />
                  <Metric label="CUDA" value={String(capabilities?.cuda_devices ?? 0)} />
                  <Metric label="Loopback" value={capabilities?.v4l2loopback_devices.length ? "ready" : "missing"} />
                  <Metric label="ONNX" value={capabilities?.onnxruntime ? "ready" : "not installed"} />
                  <Metric label="TensorRT" value={capabilities?.tensorrt ? "ready" : "missing"} />
                  <Metric label="MediaPipe" value={capabilities?.mediapipe ? "ready" : "missing"} />
                  <Metric label="PortAudio" value={capabilities?.portaudio ? "ready" : "missing"} />
                  <Metric label="Pulse" value={capabilities?.pulse_server ?? "-"} />
                  <Metric label="Kokoro" value={capabilities?.kokoro ? "ready" : "missing"} />
                  <Metric label="Avatar render" value={capabilities?.avatar_renderer ?? "local"} />
                  <Metric label="MuseTalk" value={capabilities?.musetalk_configured ? "ready" : capabilities?.musetalk_status ?? "not installed"} />
                </div>
              </details>
              <details className="settings-tree diagnostics-panel">
                <summary>Camera diagnostics</summary>
                <div className="settings-tree-body">
                  <div className="diagnostics-header">
                    <span>{status?.capture_backend ?? "waiting"}{status?.capture_format ? ` · ${status.capture_format}` : ""}</span>
                    <button className="secondary-button" onClick={() => void navigator.clipboard.writeText(diagnostics.join('\n'))}>Copy</button>
                    <button className="secondary-button" onClick={() => void api.clearDiagnostics().then(result => setDiagnostics(result.entries))}>Clear</button>
                  </div>
                  <pre>{diagnostics.length ? diagnostics.join('\n') : "Start the camera to collect diagnostics."}</pre>
                </div>
              </details>
            </div>
          </details>
        </footer>
        </div>
      </aside>
    </main>
  );
}

function ControlGroup({ title, icon, defaultOpen = false, children }: {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return <details className="settings-tree control-group" open={defaultOpen}>
    <summary>{icon}<span>{title}</span></summary>
    <div className="control-group-body">{children}</div>
  </details>;
}

function StatusPill({ active, text }: { active: boolean; text: string }) {
  return (
    <span className={active ? "status-pill active" : "status-pill"}>
      {active ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
      {text}
    </span>
  );
}

function LevelMeter({ level, peak, active }: { level: number; peak: number; active: boolean }) {
  const db = levelToDb(level);
  const clampedDb = Math.max(-60, Math.min(0, db));
  const percent = ((clampedDb + 60) / 60) * 100;
  const peakDb = levelToDb(peak);
  const peakPercent = ((Math.max(-60, Math.min(0, peakDb)) + 60) / 60) * 100;
  return (
    <div className="level-meter">
      <div className="level-meter-head">
        <span>Input level</span>
        <strong>{active ? `${clampedDb.toFixed(1)} dB` : "idle"}</strong>
      </div>
      <div className="level-meter-track" aria-label="Input level">
        <div className="level-meter-fill" style={{ width: `${percent}%` }} />
        <div className="level-meter-peak" style={{ left: `${peakPercent}%` }} />
      </div>
    </div>
  );
}

function levelToDb(level: number) {
  return level > 0 ? 20 * Math.log10(Math.max(level, 0.00001)) : -60;
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  onCommit,
  formatValue = (value: number) => value.toFixed(2)
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  onCommit?: () => void;
  formatValue?: (value: number) => string;
}) {
  return (
    <label className="slider-row">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        aria-valuetext={formatValue(value)}
        onChange={(event) => onChange(Number(event.target.value))}
        onPointerUp={onCommit} onPointerCancel={onCommit} onKeyUp={onCommit} onBlur={onCommit} />
      <output>{formatValue(value)}</output>
    </label>
  );
}

function Metric({ icon, label, value }: { icon?: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      <span>
        {icon}
        {label}
      </span>
      <strong>{value}</strong>
    </div>
  );
}
