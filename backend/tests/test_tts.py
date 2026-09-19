import time

import numpy as np
from fastapi.testclient import TestClient

from faceswap.app import app
from faceswap.detectors import FaceBox
from faceswap.lipsync import LipSyncController, MouthAnchor, apply_mouth_animation
from faceswap.schemas import TTSSpeakRequest
from faceswap.speech import SpeechSynthesisStatus
from faceswap.tts import (
    FallbackSpeechEngine,
    TTSSession,
    _declick_audio_chunk,
    _resample_audio,
    _should_duck_live_voice,
    _voice_config_for_tts,
)


def test_fallback_speech_engine_generates_audio_chunks():
    engine = FallbackSpeechEngine()

    chunks = list(engine.generate("hello world", voice="af_heart", speed=1.0))

    assert chunks
    sample_rate, audio = chunks[0]
    assert sample_rate == 24_000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) > 0


def test_resample_audio_changes_sample_count():
    audio = np.ones(240, dtype=np.float32)

    resampled = _resample_audio(audio, 24_000, 48_000)

    assert 470 <= resampled.size <= 490
    assert resampled.dtype == np.float32


def test_declick_audio_chunk_removes_dc_and_fades_edges():
    audio = np.full(480, 0.2, dtype=np.float32)
    audio[120:360] += 0.1

    cleaned = _declick_audio_chunk(audio, 48_000)

    assert abs(float(np.mean(cleaned))) < 0.03
    assert abs(float(cleaned[0])) < 1e-6
    assert abs(float(cleaned[-1])) < 1e-6


def test_lip_sync_controller_tracks_audio_envelope():
    controller = LipSyncController(window_ms=20)
    controller.begin(sample_rate=48_000, delay_ms=0, amount=1.0, smoothing=0.0, enabled=True)
    controller.append_audio(np.full(960, 0.12, dtype=np.float32))

    controller.start_playback()
    value = controller.value()

    assert value > 0.5


def test_mouth_animation_changes_face_region_only():
    frame = np.full((120, 120, 3), 160, dtype=np.uint8)
    output = apply_mouth_animation(frame, FaceBox(x=30, y=20, w=60, h=70), 1.0)

    assert output.shape == frame.shape
    assert not np.array_equal(output, frame)
    assert np.array_equal(output[0, 0], frame[0, 0])
    assert int(output.min()) >= 120


def test_mouth_animation_uses_supplied_anchor():
    frame = np.full((140, 140, 3), 170, dtype=np.uint8)
    frame[:, :, 1] = np.tile(np.arange(140, dtype=np.uint8), (140, 1))
    anchor = MouthAnchor(cx=92, cy=44, width=36, height=14)

    output = apply_mouth_animation(frame, FaceBox(x=20, y=20, w=90, h=90), 1.0, anchor)

    anchored_delta = np.mean(np.abs(output[36:54, 74:110].astype(np.int16) - frame[36:54, 74:110].astype(np.int16)))
    fallback_delta = np.mean(np.abs(output[78:98, 38:92].astype(np.int16) - frame[78:98, 38:92].astype(np.int16)))
    assert anchored_delta > 0.5
    assert fallback_delta < anchored_delta


def test_tts_session_plays_fake_engine_without_audio_devices():
    session = TTSSession(engine=_FakeEngine())

    status = session.speak(
        TTSSpeakRequest(
            text="hello",
            virtual_mic=False,
            monitor_enabled=False,
            mute_live_mic=False,
            synthesis_enabled=False,
            sample_rate=48_000,
            lip_delay_ms=0,
        )
    )
    assert status.running is True

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and session.status().running:
        time.sleep(0.02)

    status = session.status()
    assert status.running is False
    assert status.generated_seconds > 0
    assert status.last_error is None


def test_tts_session_ducks_live_voice_when_output_is_enabled():
    assert _should_duck_live_voice(
        TTSSpeakRequest(
            text="hello",
            virtual_mic=True,
            monitor_enabled=False,
            mute_live_mic=False,
            sample_rate=48_000,
        )
    )


def test_tts_voice_config_uses_speech_synthesis_target():
    config = _voice_config_for_tts(
        TTSSpeakRequest(
            text="hello",
            synthesis_target="shadow",
            synthesis_quality="balanced",
            sample_rate=48_000,
            monitor_volume=0.4,
        )
    )

    assert config.synthesis_mode == "speech_to_speech"
    assert config.synthesis_target == "shadow"
    assert config.synthesis_quality == "balanced"
    assert config.virtual_mic is False
    assert config.monitor_enabled is False


def test_tts_session_reports_speech_synth_status(monkeypatch):
    monkeypatch.setattr("faceswap.tts.SpeechToSpeechProcessor", _FakeSynth)
    session = TTSSession(engine=_FakeEngine())

    session.speak(
        TTSSpeakRequest(
            text="hello",
            virtual_mic=False,
            monitor_enabled=False,
            mute_live_mic=False,
            sample_rate=48_000,
            synthesis_target="bright",
            lip_delay_ms=0,
        )
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and session.status().running:
        time.sleep(0.02)

    status = session.status()
    assert status.synthesis_target == "bright"
    assert status.synthesis_ready is True
    assert status.synthesis_latency_ms == 12.5
    assert status.synthesis_detail == "fake synth ready"


def test_tts_endpoints_return_idle_status_and_voices():
    client = TestClient(app)

    status_response = client.get("/api/tts/status")
    voices_response = client.get("/api/tts/voices")

    assert status_response.status_code == 200
    assert voices_response.status_code == 200
    assert status_response.json()["running"] is False
    assert any(voice["name"] == "af_heart" for voice in voices_response.json())


class _FakeEngine:
    name = "fake"

    def generate(self, _text: str, *, voice: str, speed: float):
        del voice, speed
        sample_rate = 48_000
        t = np.arange(2400, dtype=np.float32) / sample_rate
        yield sample_rate, 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)


class _FakeSynth:
    def __init__(self, sample_rate, config):
        self.sample_rate = sample_rate
        self.config = config

    def process(self, block):
        return np.asarray(block, dtype=np.float32) * 0.5, 0.1

    def status(self):
        return SpeechSynthesisStatus(
            ready=True,
            latency_ms=12.5,
            queue_ms=0.0,
            rt_factor=0.1,
            fallback=False,
            detail="fake synth ready",
        )

    def close(self):
        pass


def test_lipsync_waits_for_playback_and_uses_consumed_position():
    controller = LipSyncController()
    controller.begin(sample_rate=48000, delay_ms=0, amount=1, smoothing=0, enabled=True)
    controller.append_audio(np.full(4800, .12, dtype=np.float32))
    assert controller.value(now=time.monotonic() + 10) == 0
    position = [0.0]
    controller.start_playback(lambda: position[0])
    assert controller.value(now=time.monotonic() + 10) > .5
    position[0] = .2
    assert controller.value() == 0
    controller.end()
    assert controller.value() == 0


def test_playback_clock_clamps_during_generation_gap():
    from faceswap.playback import PlaybackWriter
    clock = [5.0]
    writer = PlaybackWriter(sample_rate=48000, clock=lambda: clock[0])
    writer._submitted = .1
    writer._played = -.03
    writer._observed = 5.0
    assert writer.position_seconds() == 0
    clock[0] = 5.08
    assert abs(writer.position_seconds() - .05) < .001
    clock[0] = 15
    assert writer.position_seconds() == .1
