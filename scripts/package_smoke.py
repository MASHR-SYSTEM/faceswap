"""Run package readiness in isolated data/port without opening cameras or GUI."""
from pathlib import Path
import os
import socket
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
binary = root / 'dist/FaceSwap' / ('FaceSwap.exe' if sys.platform == 'win32' else 'FaceSwap')
with tempfile.TemporaryDirectory(prefix='faceswap-smoke-') as data:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, FACESWAP_PORT=str(port), FACESWAP_DATA_DIR=data)
    subprocess.run([str(binary), '--smoke'], env=env, check=True, timeout=90)
