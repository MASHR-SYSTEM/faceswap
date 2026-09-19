from functools import lru_cache
from pathlib import Path
import os
import sys
import platform

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
PACKAGED = bool(getattr(sys, 'frozen', False))

def application_data_dir():
    override = os.environ.get('FACESWAP_DATA_DIR')
    if override:
        return Path(override).expanduser()
    if not PACKAGED:
        return ROOT
    if platform.system() == 'Windows':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'MASHr/FaceSwap'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'mashr/faceswap'

DATA = application_data_dir()


class Settings(BaseSettings):
    project_root: Path = ROOT
    config_path: Path = DATA / ".cache/faceswap/effect.json"
    frontend_dir: Path = ROOT / "frontend/dist"
    host: str = "127.0.0.1"
    port: int = 7865
    default_width: int = 1280
    default_height: int = 720
    default_fps: int = 30
    assets_dir: Path = DATA / "assets"
    models_dir: Path = DATA / "models"
    neural_cache_dir: Path = DATA / ".cache/faceswap/tensorrt"
    background_model_path: Path = DATA / "models/mediapipe/selfie_segmenter_landscape.tflite"
    virtual_camera_device: str | None = None
    avatar_renderer: str = "local"
    musetalk_root: Path | None = None
    musetalk_python: Path | None = None
    musetalk_ffmpeg_path: str | None = None
    musetalk_version: str = "v15"
    musetalk_unet_model_path: Path | None = None
    musetalk_unet_config_path: Path | None = None
    musetalk_bbox_shift: int = 0

    model_config = SettingsConfigDict(env_prefix="FACESWAP_", env_file=None if PACKAGED else ROOT / ".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
