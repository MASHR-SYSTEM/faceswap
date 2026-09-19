from faceswap.schemas import EffectConfig
from faceswap.session import VideoSession


def test_update_effect_changes_live_effect_without_running_camera():
    session = VideoSession()
    session._state.last_error = "old error"

    status = session.update_effect(
        EffectConfig(
            mode="onnx_faceswap",
            target_image_path="assets/insightface-sea-monster-yellow-eyes-target.png",
            provider="cuda",
            strength=0.9,
        )
    )

    current = session._current_effect()
    assert status.running is False
    assert status.active_effect == "onnx_faceswap"
    assert status.last_error is None
    assert current.mode == "onnx_faceswap"
    assert current.target_image_path == "assets/insightface-sea-monster-yellow-eyes-target.png"
    assert current.provider == "cuda"
    assert current.strength == 0.9

import threading
import pytest
from faceswap.lifecycle import SessionBusy
from faceswap.voice import VoiceSession
from faceswap.tts import TTSSession
from faceswap.schemas import SessionStartRequest, VoiceConfig, TTSSpeakRequest


@pytest.mark.parametrize('factory,argument,method', [
    (VideoSession, SessionStartRequest(), 'start'),
    (VoiceSession, VoiceConfig(virtual_mic=False), 'start'),
    (TTSSession, TTSSpeakRequest(text='test', virtual_mic=False), 'speak'),
])
def test_stuck_worker_retains_ownership_and_cancellation(factory, argument, method, monkeypatch):
    worker = factory()
    worker.stop_timeout = .01
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr('faceswap.session._resolve_source_index', lambda index, source_id=None: index)
    def blocked(_argument):
        entered.set()
        release.wait(2)
    monkeypatch.setattr(worker, '_run', blocked)
    getattr(worker, method)(argument)
    assert entered.wait(1)
    thread, token = worker._thread, worker._stop_event
    try:
        stopped = worker.stop()
        assert stopped.running and stopped.phase == 'stopping'
        assert worker._thread is thread and token.is_set()
        with pytest.raises(SessionBusy):
            getattr(worker, method)(argument)
        assert worker._thread is thread and worker._stop_event is token
    finally:
        release.set()
        thread.join(2)
        worker.stop()
    assert not worker.status().running
    getattr(worker, method)(argument)
    worker.stop()
    assert worker._stop_event is not token


def test_mjpeg_ends_on_stopped_session_even_with_cached_frame():
    session = VideoSession()
    session._latest_jpeg = b'old'
    session._stop_event.set()
    assert list(session.mjpeg_stream()) == []


def test_effect_persistence_and_revision_prevent_stale_edits(tmp_path):
    from faceswap.schemas import EffectConfig
    path = tmp_path/'effect.json'
    session = VideoSession(config_path=path)
    updated = session.update_effect(EffectConfig(strength=.4), expected_revision=0)
    assert updated.effect_revision == 1
    with pytest.raises(SessionBusy):
        session.update_effect(EffectConfig(strength=.9), expected_revision=0)
    restored = VideoSession(config_path=path)
    assert restored._current_effect().strength == .4
    assert session._current_effect().strength == .4
