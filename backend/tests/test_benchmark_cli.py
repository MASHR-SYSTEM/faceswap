import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_rejects_zero_samples():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_neural.py'), '--frames', '0'], capture_output=True, text=True)
    assert result.returncode == 2 and 'must be positive' in result.stderr


def test_benchmark_fails_for_missing_input(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_neural.py'), '--frame', str(tmp_path/'missing.png')], capture_output=True, text=True)
    assert result.returncode == 1 and 'Cannot read image' in result.stderr
