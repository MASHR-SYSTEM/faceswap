from fastapi.testclient import TestClient

from faceswap.app import app


def test_status_endpoint_returns_idle_state():
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is False
    assert data["active_effect"] == "passthrough"


def test_capabilities_endpoint_returns_paths():
    client = TestClient(app)
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "models_dir" in data
    assert "assets_dir" in data



def test_production_ui_and_effect_configuration():
    from faceswap.app import settings
    client = TestClient(app)
    config = client.get('/api/session/config')
    assert config.status_code == 200
    assert 'effect' in config.json() and 'revision' in config.json()
    if settings.frontend_dir.is_dir():
        response = client.get('/')
        assert response.status_code == 200 and '<html' in response.text
    assert client.get('/api/missing').status_code == 404
