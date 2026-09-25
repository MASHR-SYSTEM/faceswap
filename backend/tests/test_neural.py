import cv2
import numpy as np

from faceswap.detectors import FaceBox
from faceswap.neural import NeuralFaceSwapEffect, _apply_live_controls, _mouth_anchor_from_insightface, build_provider_specs
from faceswap.processor import FrameProcessor
from faceswap.schemas import EffectConfig


def test_provider_specs_prefer_tensorrt_then_cuda_then_cpu(tmp_path):
    specs = build_provider_specs(
        "auto",
        ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"),
        tmp_path,
    )

    names = [provider[0] if isinstance(provider, tuple) else provider for provider in specs]
    assert names == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def test_provider_specs_cuda_skips_tensorrt(tmp_path):
    specs = build_provider_specs(
        "cuda",
        ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"),
        tmp_path,
    )

    names = [provider[0] if isinstance(provider, tuple) else provider for provider in specs]
    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    cuda_options = specs[0][1]
    assert cuda_options["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    assert cuda_options["cudnn_conv_use_max_workspace"] == "1"


def test_neural_configure_reuses_loaded_runtime(tmp_path, monkeypatch):
    model = tmp_path / "inswapper_128.onnx"
    model.write_bytes(b"placeholder")
    target = tmp_path / "target.png"
    cv2.imwrite(str(target), np.zeros((16, 16, 3), dtype=np.uint8))

    calls = 0

    def fake_load(self, _config):
        nonlocal calls
        calls += 1
        self._info.ready = True

    monkeypatch.setattr(NeuralFaceSwapEffect, "_load", fake_load)
    effect = NeuralFaceSwapEffect(tmp_path, tmp_path / "cache")

    effect.configure(model_path=model, target_image_path=target, provider="auto")
    effect.configure(model_path=model, target_image_path=target, provider="auto")

    assert calls == 1
    assert effect.ready is True


def test_neural_configure_reloads_when_target_file_changes_at_same_path(tmp_path, monkeypatch):
    model = tmp_path / "inswapper_128.onnx"
    model.write_bytes(b"placeholder")
    target = tmp_path / "target.png"
    target.write_bytes(b"old")

    calls = 0

    def fake_load(self, _config):
        nonlocal calls
        calls += 1
        self._info.ready = True

    monkeypatch.setattr(NeuralFaceSwapEffect, "_load", fake_load)
    effect = NeuralFaceSwapEffect(tmp_path, tmp_path / "cache")

    effect.configure(model_path=model, target_image_path=target, provider="auto")
    target.write_bytes(b"new-target")
    effect.configure(model_path=model, target_image_path=target, provider="auto")

    assert calls == 2
    assert effect.ready is True


def test_neural_live_controls_strength_zero_returns_original_frame():
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    swapped = np.full((64, 64, 3), 255, dtype=np.uint8)
    config = EffectConfig(
        strength=0.0,
        edge_feather=0.0,
        color_match=0.0,
        sharpen=0.0,
        temporal_smoothing=0.0,
    )

    output, _ = _apply_live_controls(original, swapped, FaceBox(16, 16, 32, 32), config, None)

    assert np.array_equal(output, original)


def test_neural_live_controls_blends_swapped_face_region():
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    swapped = np.full((64, 64, 3), 255, dtype=np.uint8)
    config = EffectConfig(
        strength=1.0,
        edge_feather=0.0,
        color_match=0.0,
        sharpen=0.0,
        temporal_smoothing=0.0,
    )

    output, _ = _apply_live_controls(original, swapped, FaceBox(16, 16, 32, 32), config, None)

    assert output[32, 32, 0] == 255
    assert output[0, 0, 0] == 0


def test_neural_live_controls_temporal_smoothing_uses_previous_output():
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    swapped = np.full((64, 64, 3), 255, dtype=np.uint8)
    previous = np.zeros((64, 64, 3), dtype=np.uint8)
    config = EffectConfig(
        strength=1.0,
        edge_feather=0.0,
        color_match=0.0,
        sharpen=0.0,
        temporal_smoothing=0.5,
    )

    output, _ = _apply_live_controls(original, swapped, FaceBox(16, 16, 32, 32), config, previous)

    assert 120 <= int(output[32, 32, 0]) <= 135


def test_mouth_anchor_uses_insightface_mouth_keypoints():
    face = type(
        "Face",
        (),
        {
            "kps": np.asarray(
                [
                    [20.0, 20.0],
                    [44.0, 20.0],
                    [32.0, 34.0],
                    [24.0, 48.0],
                    [42.0, 50.0],
                ],
                dtype=np.float32,
            )
        },
    )()

    anchor = _mouth_anchor_from_insightface(face, FaceBox(10, 10, 60, 80))

    assert 32 <= anchor.cx <= 34
    assert 49 <= anchor.cy <= 53
    assert anchor.width > 30


def test_neural_mode_missing_assets_returns_frame_without_optional_imports():
    processor = FrameProcessor()
    processor._default_target_path = processor._default_target_path.parent / "missing-target.png"
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    config = EffectConfig(mode="onnx_faceswap", target_image_path=None)

    output = processor.process(frame.copy(), config)

    assert output.shape == frame.shape
    assert processor.model_ready is False
    assert "target image" in (processor.model_error or "")


def test_neural_error_does_not_leak_into_non_neural_mode():
    processor = FrameProcessor()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    processor.process(frame.copy(), EffectConfig(mode="onnx_faceswap", target_image_path=None))
    processor.process(frame.copy(), EffectConfig(mode="passthrough"))

    assert processor.model_error is None
    assert processor.model_ready is False

import pytest
from faceswap.neural import _face_alpha_mask, _apply_controls_region


@pytest.mark.parametrize('offset', [-.5, .5])
@pytest.mark.parametrize('feather', [0, 1])
def test_vertical_extremes_preserve_forehead_and_chin_coverage(offset, feather):
    box = FaceBox(80, 80, 80, 100)
    centred = _face_alpha_mask((300, 300), box, 2, 0, feather)
    shifted = _face_alpha_mask((300, 300), box, 2, offset, feather)
    assert np.all(shifted >= centred)
    assert shifted[70, 120] > .9  # Forehead above the downward-shifted mask.
    assert shifted[185, 120] > .9  # Chin below the upward-shifted mask.


def test_max_vertical_has_no_horizontal_forehead_cutoff():
    original = np.zeros((300, 300, 3), np.uint8)
    swapped = np.full_like(original, 200)
    config = EffectConfig(scale=2, y_offset=.5, strength=1, edge_feather=1,
                          color_match=0, sharpen=0, temporal_smoothing=0)
    output, _ = _apply_live_controls(original, swapped, FaceBox(80, 80, 80, 100), config, None)
    assert np.all(output[70:90, 120, 0] > 180)
    assert np.max(np.abs(np.diff(output[70:90, 120, 0].astype(int)))) <= 2


@pytest.mark.parametrize('box', [FaceBox(30, 20, 35, 48), FaceBox(0, 0, 30, 30), FaceBox(75, 65, 25, 35)])
@pytest.mark.parametrize('feather,scale,offset,smoothing', [
    (0, 1, 0, 0), (1, 1.7, -.3, .5), (.35, .65, .4, 0),
    (1, 2, .5, .4), (1, 2, -.5, .4),
])
def test_region_controls_match_full_frame_reference(box, feather, scale, offset, smoothing):
    rng = np.random.default_rng(24)
    original = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    swapped = rng.integers(0, 256, original.shape, dtype=np.uint8)
    previous = rng.integers(0, 256, original.shape, dtype=np.uint8)
    config = EffectConfig(edge_feather=feather, scale=scale, y_offset=offset, temporal_smoothing=smoothing)
    mask = _face_alpha_mask(original.shape[:2], box, scale, offset, feather)
    reference = _apply_controls_region(original, swapped, mask, config, previous)
    actual, _ = _apply_live_controls(original, swapped, box, config, previous)
    assert np.max(np.abs(reference.astype(int) - actual.astype(int))) <= 1
    assert np.array_equal(actual[mask == 0], original[mask == 0])


def test_failed_initialisation_is_not_retried_every_frame(tmp_path, monkeypatch):
    model, target = tmp_path/'model.onnx', tmp_path/'target.png'
    model.write_bytes(b'model'); target.write_bytes(b'target')
    effect = NeuralFaceSwapEffect(tmp_path, tmp_path/'cache')
    attempts = []
    def fail(config):
        attempts.append(config)
        raise RuntimeError('unsupported runtime')
    monkeypatch.setattr(effect, '_load', fail)
    for _ in range(10):
        effect.configure(model_path=model, target_image_path=target, provider='cuda')
    assert len(attempts) == 1 and not effect.ready
    target.write_bytes(b'changed target')
    effect.configure(model_path=model, target_image_path=target, provider='cuda')
    assert len(attempts) == 2


def test_source_change_reuses_runtime(tmp_path, monkeypatch):
    model, target = tmp_path/'model.onnx', tmp_path/'target.png'
    model.write_bytes(b'model'); target.write_bytes(b'target')
    effect = NeuralFaceSwapEffect(tmp_path, tmp_path/'cache')
    loads, identities = [], []
    def load(config):
        loads.append(config)
        effect._face_app = object(); effect._swapper = object(); effect._info.ready = True
    monkeypatch.setattr(effect, '_load', load)
    monkeypatch.setattr(effect, '_set_source', lambda config: identities.append(config))
    effect.configure(model_path=model, target_image_path=target, provider='cuda')
    target.write_bytes(b'new source')
    effect.configure(model_path=model, target_image_path=target, provider='cuda')
    assert len(loads) == 1 and len(identities) == 1


def test_face_selection_keeps_overlapping_face_when_larger_person_enters():
    from types import SimpleNamespace
    from faceswap.neural import _select_live_face
    tracked = SimpleNamespace(bbox=[10,10,50,50])
    stranger = SimpleNamespace(bbox=[60,0,160,100])
    assert _select_live_face([stranger, tracked], FaceBox(11,11,40,40)) is tracked


@pytest.mark.parametrize('smoothing', [0, .5, .95])
def test_transformation_endpoints_and_midpoint_with_history(smoothing):
    original = np.full((64, 64, 3), 30, np.uint8)
    swapped = np.full_like(original, 210)
    previous = np.full_like(original, 150)
    box = FaceBox(16, 16, 32, 32)
    config = EffectConfig(strength=1, temporal_smoothing=smoothing,
                          color_match=0, sharpen=0, edge_feather=.35)
    full, full_history = _apply_live_controls(original, swapped, box, config, previous)
    zero, zero_history = _apply_live_controls(original, swapped, box,
        config.model_copy(update={'strength': 0}), previous)
    half, _ = _apply_live_controls(original, swapped, box,
        config.model_copy(update={'strength': .5}), previous)
    assert np.array_equal(zero, original)
    expected_half = (original.astype(float) + full.astype(float)) / 2
    assert np.max(np.abs(half.astype(float) - expected_half)) <= 1
    mask = _face_alpha_mask(original.shape[:2], box, config.scale, config.y_offset, config.edge_feather)
    assert np.array_equal(full[mask == 0], original[mask == 0])
    if smoothing:
        assert np.array_equal(full_history, zero_history)
        assert full[32, 32, 0] == int(210 * (1-smoothing) + 150 * smoothing)
    else:
        assert full_history is None and zero_history is None
        assert full[32, 32, 0] == 210


def test_rapid_strength_changes_do_not_contaminate_temporal_history():
    original = np.full((64, 64, 3), 30, np.uint8)
    swapped = np.full_like(original, 210)
    box = FaceBox(16, 16, 32, 32)
    config = EffectConfig(strength=1, temporal_smoothing=.7, color_match=0, sharpen=0)
    history = reference_history = None
    for strength in [1, 0, .25, 0, 1]:
        reference, reference_history = _apply_live_controls(original, swapped, box, config, reference_history)
        result, history = _apply_live_controls(original, swapped, box,
            config.model_copy(update={'strength': strength}), history)
        assert np.array_equal(history, reference_history)
        expected = original.astype(float) + strength * (reference.astype(float) - original)
        assert np.max(np.abs(result.astype(float) - expected)) <= 1
    _, history = _apply_live_controls(original, swapped, box,
        config.model_copy(update={'temporal_smoothing': 0}), history)
    assert history is None


@pytest.mark.parametrize('strength', [0, .25, .82, 1])
def test_strength_without_temporal_smoothing_matches_legacy_blend(strength):
    rng = np.random.default_rng(83)
    original = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    swapped = rng.integers(0, 256, original.shape, dtype=np.uint8)
    box = FaceBox(16, 16, 32, 32)
    config = EffectConfig(strength=strength, color_match=0, sharpen=0, temporal_smoothing=0)
    mask = _face_alpha_mask(original.shape[:2], box, config.scale, config.y_offset, config.edge_feather)
    alpha = (mask * strength)[:, :, None]
    legacy = np.clip(original.astype(np.float32) * (1-alpha) + swapped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    output, _ = _apply_live_controls(original, swapped, box, config, None)
    assert np.max(np.abs(output.astype(int) - legacy.astype(int))) <= 1


def test_strength_changes_keep_models_and_inference_warm(tmp_path, monkeypatch):
    from types import SimpleNamespace
    model, target = tmp_path/'model.onnx', tmp_path/'target.png'
    model.write_bytes(b'model'); target.write_bytes(b'target')
    loads, swaps = [], []
    detected = True
    def detect(*args, **kwargs):
        if not detected:
            return np.empty((0, 5)), np.empty((0, 5, 2))
        return np.array([[16, 16, 48, 48, .99]]), np.array([[[24, 25], [40, 25], [32, 33], [26, 40], [38, 40]]])
    def load(self, config):
        loads.append(config)
        self._face_app = SimpleNamespace(det_model=SimpleNamespace(detect=detect))
        self._swapper = object()
        self._source_face = object()
        self._info.ready = True
    def swap(*args):
        swaps.append(1)
        return np.full((64, 64, 3), 200, np.uint8)
    monkeypatch.setattr(NeuralFaceSwapEffect, '_load', load)
    monkeypatch.setattr('faceswap.neural.swap_frame', swap)
    effect = NeuralFaceSwapEffect(tmp_path, tmp_path/'cache')
    original = np.zeros((64, 64, 3), np.uint8)
    for strength in [1, 0, .5, 1]:
        effect.configure(model_path=model, target_image_path=target, provider='cuda')
        output = effect.apply(original, EffectConfig(strength=strength, temporal_smoothing=.5, color_match=0, sharpen=0))
        assert effect.last_error is None
        assert effect._previous_candidate is not None
        if strength == 0:
            assert np.array_equal(output, original)
    assert len(loads) == 1 and len(swaps) == 4
    detected = False
    effect.apply(original, EffectConfig())
    assert effect._previous_candidate is None
