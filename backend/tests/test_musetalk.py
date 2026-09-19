from pathlib import Path

import numpy as np
import pytest
import yaml

from faceswap.avatar import AvatarStudio, MuseTalkConfig, _musetalk_command_parts
from faceswap.schemas import AvatarProfile, AvatarRenderJob


def test_musetalk_status_reports_not_installed_when_root_missing(tmp_path):
    studio = AvatarStudio(tmp_path / "assets", renderer="musetalk", musetalk_config=MuseTalkConfig(root=tmp_path / "missing"))

    status, detail = studio.musetalk_status()

    assert status == "not_installed"
    assert "root" in detail.lower()


def test_musetalk_command_parts_reports_missing_weights(tmp_path):
    root = tmp_path / "MuseTalk"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "inference.py").write_text("print('fake')\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="UNet model"):
        _musetalk_command_parts(MuseTalkConfig(root=root, python=python, ffmpeg_path=str(ffmpeg)))


def test_musetalk_renderer_builds_yaml_and_copies_output(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    root = tmp_path / "MuseTalk"
    python = tmp_path / "python"
    ffmpeg = tmp_path / "ffmpeg"
    unet_model = root / "models" / "musetalkV15" / "unet.pth"
    unet_config = root / "models" / "musetalkV15" / "musetalk.json"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "inference.py").write_text("print('fake')\n", encoding="utf-8")
    unet_model.parent.mkdir(parents=True)
    unet_model.write_bytes(b"model")
    unet_config.write_text("{}", encoding="utf-8")
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")

    target = tmp_path / "target.png"
    target.write_bytes(b"target")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio")

    studio = AvatarStudio(
        assets,
        renderer="musetalk",
        musetalk_config=MuseTalkConfig(root=root, python=python, ffmpeg_path=str(ffmpeg)),
    )
    job_id = "a" * 32
    (assets / "avatar_outputs" / job_id).mkdir(parents=True)
    video_path = assets / "avatar_outputs" / job_id / "avatar.mp4"
    profile = AvatarProfile(
        id="avatar-123",
        display_name="Avatar",
        target_image_path=str(target),
        voice_reference_path=str(audio),
        created_at="2026-01-01T00:00:00+00:00",
        consent_confirmed=True,
    )

    studio._store_job(AvatarRenderJob(id=job_id, profile_id=profile.id, status="running",
        created_at="2026-01-01T00:00:00+00:00", text_preview="test"))

    def fake_create_still_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"source")

    def fake_run(command, **kwargs):
        result_dir = Path(command[command.index("--result_dir") + 1])
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.mp4").write_bytes(b"musetalk-video")
        return _Completed(stdout="ok", stderr="")

    monkeypatch.setattr("faceswap.avatar._create_still_video", fake_create_still_video)
    monkeypatch.setattr(studio, "_run_render_command", fake_run)

    detail = studio._render_video_with_musetalk(
        profile=profile,
        target_path=target,
        audio_samples=np.ones(48_000, dtype=np.float32),
        sample_rate=48_000,
        audio_path=audio,
        video_path=video_path,
        job_id=job_id,
    )

    assert detail == "MuseTalk renderer"
    assert video_path.read_bytes() == b"musetalk-video"
    yaml_text = (assets / "avatar_outputs" / job_id / "musetalk" / "inference.yaml").read_text(encoding="utf-8")
    assert "source-25fps.mp4" in yaml_text
    assert yaml.safe_load(yaml_text)["avatar_0"]["audio_path"] == str(audio.resolve())


class _Completed:
    def __init__(self, stdout: str, stderr: str):
        self.stdout = stdout
        self.stderr = stderr
