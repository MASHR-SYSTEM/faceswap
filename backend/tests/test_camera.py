import errno
import struct

import pytest
import numpy as np

from faceswap import camera
from faceswap.schemas import DeviceInfo, SessionStartRequest
from faceswap.session import _resolve_source_index
from faceswap.session import _looks_like_scanline_corruption, _open_capture


def device_tree(tmp_path, monkeypatch, nodes):
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")
    for index, name, interface in nodes:
        node = tmp_path / f'video{index}'
        node.mkdir()
        (node / 'name').write_text(name)
        (node / 'index').write_text(str(interface))
    monkeypatch.setattr(camera, 'SYS_VIDEO', tmp_path)


def test_discovery_lists_standard_and_ir_without_streaming(tmp_path, monkeypatch):
    device_tree(tmp_path, monkeypatch, [(0, 'ASUS IR camera', 0),
        (1, 'ASUS IR camera', 1), (4, 'ASUS FHD webcam', 0), (5, 'ASUS FHD webcam', 1)])
    monkeypatch.setattr(camera, '_query_capabilities', lambda path:
        ('uvcvideo', 1 if path.name in {'video0', 'video4'} else 0x00800000))
    devices = camera.list_camera_devices()
    assert [d.index for d in devices] == [4, 0]
    assert [d.kind for d in devices] == ['standard', 'infrared']
    assert devices[0].is_default and not devices[1].is_default
    assert 'Infrared' in devices[1].label


@pytest.mark.parametrize('error', [errno.EBUSY, errno.EACCES])
def test_busy_or_inaccessible_camera_stays_in_list(tmp_path, monkeypatch, error):
    device_tree(tmp_path, monkeypatch, [(0, 'FHD webcam', 0), (1, 'FHD webcam', 1), (2, 'Infrared camera', 0)])
    def query(path):
        if path.name in {'video0', 'video1'}:
            raise OSError(error, 'device unavailable')
        return 'uvcvideo', 1
    monkeypatch.setattr(camera, '_query_capabilities', query)
    assert [d.index for d in camera.list_camera_devices()] == [0, 2]


def test_virtual_camera_never_becomes_standard_default(tmp_path, monkeypatch):
    device_tree(tmp_path, monkeypatch, [(0, 'FaceSwap output', 0), (2, 'IR camera', 0)])
    monkeypatch.setattr(camera, '_query_capabilities', lambda path:
        ('v4l2loopback' if path.name == 'video0' else 'uvcvideo', 1))
    devices = camera.list_camera_devices()
    assert {d.kind for d in devices} == {'virtual', 'infrared'}
    assert not any(d.is_default for d in devices)


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="Linux ioctl ABI test")
def test_querycap_uses_per_node_caps_and_closes_fd(monkeypatch):
    closed = []
    monkeypatch.setattr(camera.os, 'open', lambda path, flags: 123)
    monkeypatch.setattr(camera.os, 'close', closed.append)
    def ioctl(fd, command, buf):
        assert command == camera.VIDIOC_QUERYCAP
        buf[:] = struct.pack('=16s32s32s6I', b'uvcvideo', b'webcam', b'usb', 0,
                             camera.V4L2_CAP_DEVICE_CAPS | 1, 0x00800000, 0, 0, 0)
    monkeypatch.setattr(camera.fcntl, 'ioctl', ioctl)
    assert camera._query_capabilities(camera.DEV_VIDEO / 'video1') == ('uvcvideo', 0x00800000)
    assert closed == [123]


def test_default_and_explicit_selection_never_silently_switch_to_ir(monkeypatch):
    devices = [DeviceInfo(index=0, label='IR', kind='infrared'), DeviceInfo(index=4, label='FHD')]
    monkeypatch.setattr('faceswap.session.list_camera_devices', lambda: devices)
    assert SessionStartRequest().source_index is None
    assert _resolve_source_index(None) == 4
    assert _resolve_source_index(0) == 0  # IR remains an explicit choice.
    with pytest.raises(ValueError, match='unavailable'):
        _resolve_source_index(8)
    devices.pop()
    with pytest.raises(ValueError, match='No standard webcam'):
        _resolve_source_index(None)


def test_start_missing_camera_returns_actionable_error(monkeypatch):
    from fastapi.testclient import TestClient
    from faceswap.app import app
    monkeypatch.setattr('faceswap.session.list_camera_devices', lambda: [])
    response = TestClient(app).post('/api/session/start', json={})
    assert response.status_code == 400
    assert 'No standard webcam' in response.json()['detail']


def test_scanline_corruption_is_rejected_but_black_privacy_frame_is_valid():
    black = np.zeros((480, 640, 3), np.uint8)
    scanline = black.copy()
    scanline[0, :, 0] = np.arange(640, dtype=np.uint16).astype(np.uint8)
    scanline[0, :, 1] = 255
    assert not _looks_like_scanline_corruption(black)
    assert _looks_like_scanline_corruption(scanline)


def test_windows_capture_falls_back_after_corrupt_stream(monkeypatch):
    good = np.full((480, 640, 3), 80, np.uint8)
    class Capture:
        def __init__(self, frame, opened=True): self.frame, self.opened, self.released = frame, opened, False
        def set(self, *_): return True
        def isOpened(self): return self.opened
        def read(self): return True, self.frame
        def release(self): self.released = True
    first, second = Capture(good, opened=False), Capture(good)
    captures = iter((first, second))
    monkeypatch.setattr('faceswap.session.capture_backends', lambda: [(1, 'Media Foundation'), (2, 'DirectShow')])
    monkeypatch.setattr('faceswap.session.cv2.VideoCapture', lambda *_: next(captures))
    request = SessionStartRequest(source_index=0)
    capture, name, frame = _open_capture(request)
    assert first.released
    assert capture is second and name == 'DirectShow'
    assert np.array_equal(frame, good)
