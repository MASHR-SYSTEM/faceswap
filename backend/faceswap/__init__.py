from pathlib import Path
import sys
import tomllib

__all__ = ["__version__"]

_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
try:
    __version__ = tomllib.loads((_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
except (OSError, KeyError, tomllib.TOMLDecodeError):
    __version__ = "0+unknown"
