from pathlib import Path

import pytest

from faceswap.model_setup import CATALOG, SetupManager


def test_setup_starts_missing_and_requires_explicit_terms(tmp_path: Path):
    manager = SetupManager(tmp_path / "models", tmp_path / "state" / "setup.json")
    state = manager.state()
    assert state["ready"] is False
    assert {item["id"] for item in state["components"]} == {"inswapper", "buffalo_l", "background"}
    assert state["download_bytes"] == sum(item.size for item in CATALOG)
    with pytest.raises(ValueError, match="Accept"):
        manager.start(accept_terms=False)


def test_setup_completion_is_separate_from_model_readiness(tmp_path: Path):
    manager = SetupManager(tmp_path / "models", tmp_path / "state" / "setup.json")
    assert manager.complete()["completed"] is True
    assert manager.state()["ready"] is False
