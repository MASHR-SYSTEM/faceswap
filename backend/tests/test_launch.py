import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

from faceswap import app as application
from faceswap.schemas import EffectConfig, VoiceConfig, DeviceInfo
from faceswap.session import _resolve_source_index
from faceswap.output import PacedOutput


def png():
    data = io.BytesIO()
    Image.new('RGB', (12, 8), 'red').save(data, format='PNG')
    return data.getvalue()


def upload(data):
    return UploadFile(file=io.BytesIO(data), filename='background.png', headers=Headers({'content-type': 'image/png'}))


def test_invalid_background_cannot_destroy_current_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(application.settings, 'assets_dir', tmp_path)
    old = tmp_path / 'background.png'
    old.write_bytes(png())
    with pytest.raises(HTTPException):
        asyncio.run(application._save_uploaded_image(upload(b'broken'), 'background', 'Background'))
    assert old.read_bytes() == png()
    result = asyncio.run(application._save_uploaded_image(upload(png()), 'background', 'Background'))
    assert result.path != str(old)
    assert old.read_bytes() == png()
    assert (result.width, result.height) == (12, 8)


def test_background_thumbnail_rejects_outside_assets(tmp_path, monkeypatch):
    folder = tmp_path / 'assets'
    folder.mkdir()
    outside = tmp_path / 'private.png'
    outside.write_bytes(png())
    monkeypatch.setattr(application.settings, 'assets_dir', folder)
    monkeypatch.setattr(application.session, '_current_effect', lambda: EffectConfig(background_path=str(outside)))
    with pytest.raises(HTTPException) as error:
        application.current_background_thumbnail()
    assert error.value.status_code == 404


def test_stop_all_attempts_every_worker_and_preserves_stopping(monkeypatch):
    called = []
    def worker(name, fail=False):
        def stop():
            called.append(name)
            if fail: raise RuntimeError('blocked')
            return SimpleNamespace(model_dump=lambda: {'phase': 'idle'})
        return SimpleNamespace(stop=stop, status=lambda: SimpleNamespace(model_dump=lambda: {'phase': 'stopping'}))
    monkeypatch.setattr(application, 'session', worker('video', True))
    monkeypatch.setattr(application, 'voice_session', worker('voice'))
    monkeypatch.setattr(application, 'tts_session', worker('tts'))
    result = asyncio.run(application.stop_all())
    assert set(called) == {'video', 'voice', 'tts'}
    assert result['video']['status']['phase'] == 'stopping'
    assert result['video']['error'] == 'blocked'
    assert result['voice']['error'] is None


def test_stable_camera_id_survives_index_changes(monkeypatch):
    monkeypatch.setattr('faceswap.session.list_camera_devices', lambda: [DeviceInfo(index=6, device_id='rgb', label='RGB')])
    assert _resolve_source_index(0, 'rgb') == 6
    with pytest.raises(ValueError, match='unavailable'):
        _resolve_source_index(6, 'missing')


def test_windows_voice_requires_explicit_cable_not_speakers(monkeypatch):
    from faceswap import voice
    monkeypatch.setattr(voice, 'list_audio_devices', lambda: [SimpleNamespace(index=1, name='Speakers', max_output_channels=2)])
    with pytest.raises(RuntimeError, match='CABLE Input'):
        voice.WindowsCableWriter(VoiceConfig(virtual_output_device=1))


def test_output_rejects_wrong_frame_dimensions_and_type():
    output = PacedOutput(10, 8, 30)
    output.validate(np.zeros((8, 10, 3), np.uint8))
    with pytest.raises(ValueError): output.validate(np.zeros((8, 9, 3), np.uint8))
    with pytest.raises(ValueError): output.validate(np.zeros((8, 10, 3), np.float32))
