from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from faceswap import swap_backends
from faceswap.processor import FrameProcessor
from faceswap.schemas import EffectConfig


class ExampleBackend:
    ready = True
    target_ready = True
    active_provider = "test"
    last_error = None
    last_face = None
    last_mouth_anchor = None
    timings = SimpleNamespace(frame_ms=1, detect_ms=0, swap_ms=1)

    def __init__(self, models_dir, cache_dir):
        self.cache_dir = cache_dir
        self.closed = False
        self.configurations = []

    def configure(self, **kwargs):
        self.configurations.append(kwargs)

    def apply(self, frame, config):
        return np.full_like(frame, 73)

    def close(self):
        self.closed = True


@pytest.fixture
def example(monkeypatch):
    monkeypatch.setattr(swap_backends, "_BACKENDS", dict(swap_backends._BACKENDS))
    instances = []

    def factory(*args):
        backend = ExampleBackend(*args)
        instances.append(backend)
        return backend

    swap_backends.register_backend(swap_backends.BackendSpec(
        "example", "Example", "example.onnx", factory))
    return instances


def test_legacy_config_defaults_and_unknown_backend_rejected():
    assert EffectConfig.model_validate({"mode": "onnx_faceswap"}).swap_backend == "inswapper"
    with pytest.raises(ValidationError, match="Unknown face-swap backend"):
        EffectConfig(swap_backend="missing")


def test_backend_switch_routes_frames_status_and_releases_resources(example):
    processor = FrameProcessor()
    original = processor._neural_faceswap
    config = EffectConfig(swap_backend="example", mirror=False)
    frame = np.zeros((32, 32, 3), np.uint8)
    try:
        assert np.all(processor.process(frame, config) == 73)
        backend = example[0]
        assert backend.configurations[-1]["model_path"].name == "example.onnx"
        assert backend.cache_dir.name == "example"
        assert processor.model_provider == "test"
        assert processor.model_ready
        assert processor.model_latency_ms == 1
        processor.process(frame, config)
        assert len(example) == 1
        processor.process(frame, EffectConfig(mode="passthrough"))
        assert processor.model_provider is None
        assert not processor.model_ready
        processor.process(frame, EffectConfig())
        assert backend.closed
        assert processor._neural_faceswap is not original
        processor.process(frame, config)
        assert len(example) == 2
    finally:
        processor.close()
    assert example[-1].closed


def test_catalog_and_api_validation(example):
    from faceswap.app import app
    client = TestClient(app)
    catalog = client.get("/api/capabilities").json()["swap_backends"]
    assert {"id": "example", "label": "Example", "default_model": "example.onnx"} in catalog
    response = client.post("/api/session/start", json={"effect": {"swap_backend": "missing"}})
    assert response.status_code == 422
    config = EffectConfig(swap_backend="example")
    assert EffectConfig.model_validate_json(config.model_dump_json()).swap_backend == "example"


def test_duplicate_registration_rejected(example):
    with pytest.raises(ValueError, match="already registered"):
        swap_backends.register_backend(swap_backends.get_backend("example"))
