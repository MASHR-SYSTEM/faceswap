import numpy as np

from faceswap.detectors import FaceBox
from faceswap.effects import apply_privacy_blur


def test_face_box_expanded_stays_in_bounds():
    box = FaceBox(x=20, y=20, w=50, h=60)
    expanded = box.expanded(scale=2.0, max_width=100, max_height=100)
    assert expanded.x >= 0
    assert expanded.y >= 0
    assert expanded.x + expanded.w <= 100
    assert expanded.y + expanded.h <= 100


def test_face_box_smoothing_moves_toward_new_box():
    previous = FaceBox(x=0, y=0, w=100, h=100)
    current = FaceBox(x=100, y=100, w=200, h=200)
    smoothed = current.smoothed(previous, smoothing=0.5)
    assert smoothed.x == 50
    assert smoothed.y == 50
    assert smoothed.w == 150
    assert smoothed.h == 150


def test_privacy_blur_modifies_face_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[30:70, 30:70] = 255
    output = apply_privacy_blur(frame.copy(), FaceBox(x=30, y=30, w=40, h=40), strength=1.0)
    assert output.shape == frame.shape
    assert not np.array_equal(output, frame)

