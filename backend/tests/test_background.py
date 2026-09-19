import cv2
import numpy as np

from faceswap.background import VirtualBackgroundEffect, _resize_cover
from faceswap.schemas import EffectConfig


def test_resize_cover_matches_destination_shape():
    image = np.zeros((100, 300, 3), dtype=np.uint8)

    resized = _resize_cover(image, width=160, height=120)

    assert resized.shape == (120, 160, 3)
    assert resized.flags["C_CONTIGUOUS"]


def test_background_missing_model_returns_frame(tmp_path):
    background = tmp_path / "background.png"
    cv2.imwrite(str(background), np.zeros((20, 30, 3), dtype=np.uint8))
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    effect = VirtualBackgroundEffect(tmp_path / "missing.tflite")

    output = effect.apply(
        frame.copy(),
        EffectConfig(background_enabled=True, background_path=str(background)),
        background_path=background,
    )

    assert np.array_equal(output, frame)
    assert effect.ready is False
    assert "segmentation model" in (effect.last_error or "")


def test_background_resize_is_cached_and_resolution_invalidates(tmp_path, monkeypatch):
    effect = VirtualBackgroundEffect(tmp_path / 'model')
    effect._ready = True
    effect._segmenter = object()
    effect._mp = object()
    effect._background_bgr = np.full((20, 30, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(effect, 'configure', lambda _: None)
    monkeypatch.setattr(effect, '_foreground_alpha', lambda frame, _: np.zeros(frame.shape[:2], np.float32))
    calls = []
    def resize(image, width, height):
        calls.append((width, height))
        return _resize_cover(image, width, height)
    monkeypatch.setattr('faceswap.background._resize_cover', resize)
    config = EffectConfig(background_enabled=True)
    for size in [(24, 32), (24, 32), (48, 64)]:
        output = effect.apply(np.zeros((*size, 3), np.uint8), config, background_path=None)
        assert np.all(output == 200)
    assert calls == [(32, 24), (64, 48)]
    effect.close()
    assert effect._resized_background is None
