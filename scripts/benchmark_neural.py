#!/usr/bin/env python3
"""Recorded-input benchmark; never opens a camera or changes a running session."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from faceswap.neural import NeuralFaceSwapEffect
from faceswap.schemas import EffectConfig
from faceswap.settings import get_settings


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def summary(values):
    return {name: round(float(fn(values)), 3) for name, fn in (
        ('mean', np.mean), ('p50', np.median), ('p95', lambda x: np.percentile(x, 95)))}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_metadata():
    try:
        return subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'],
            text=True, timeout=5).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='assets/aging-cyber-monk-target.png')
    parser.add_argument('--frame', help='Still image; defaults to target')
    parser.add_argument('--video', help='Recorded clip; loops at EOF')
    parser.add_argument('--model', default='models/inswapper_128.onnx')
    parser.add_argument('--provider', choices=['auto', 'cuda', 'tensorrt', 'cpu'], default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--width', type=positive, default=640)
    parser.add_argument('--height', type=positive, default=480)
    parser.add_argument('--warmup', type=positive, default=5)
    parser.add_argument('--frames', type=positive, default=60)
    parser.add_argument('--config', type=Path, help='JSON EffectConfig overrides')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', type=Path, help='Write JSON result')
    args = parser.parse_args()
    capture = None
    try:
        config = EffectConfig(**(json.loads(args.config.read_text()) if args.config else {}))
        config.provider = args.provider
        config.precision = args.precision
        model, target = resolve_path(args.model), resolve_path(args.target)
        input_path = resolve_path(args.video or args.frame or args.target)
        if args.video:
            capture = cv2.VideoCapture(str(input_path))
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open clip: {input_path}')
            def next_frame():
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Clip has no decodable frames')
                return frame
        else:
            still = cv2.imread(str(input_path))
            if still is None:
                raise RuntimeError(f'Cannot read image: {input_path}')
            def next_frame():
                return still.copy()
        settings = get_settings()
        effect = NeuralFaceSwapEffect(settings.models_dir, settings.neural_cache_dir)
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            effect.configure(model_path=model, target_image_path=target, provider=args.provider, precision=args.precision)
        load_ms = (time.perf_counter() - start) * 1000
        if not effect.ready:
            raise RuntimeError(effect.last_error or 'Runtime not ready')
        samples = {}
        face_frames = 0
        for index in range(args.frames + args.warmup):
            frame = cv2.resize(next_frame(), (args.width, args.height), interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            effect.apply(frame, config)
            elapsed = (time.perf_counter() - start) * 1000
            if effect.last_error:
                raise RuntimeError(effect.last_error)
            if index >= args.warmup:
                face_frames += effect.last_face is not None
                for key, value in {'wall_ms': elapsed, **vars(effect.timings)}.items():
                    samples.setdefault(key, []).append(value)
        if not face_frames:
            raise RuntimeError('No successful face swaps in measured frames')
        versions = {}
        for package in ['onnxruntime-gpu', 'insightface', 'numpy', 'opencv-python']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        result = {'provider': effect.active_provider, 'requested_provider': args.provider,
                  'resolution': [args.width, args.height], 'frames': args.frames,
                  'face_frames': face_frames, 'warmup': args.warmup, 'load_ms': round(load_ms, 2),
                  'timings_ms': {key: summary(values) for key, values in samples.items()},
                  'config': config.model_dump(), 'model_sha256': sha256(model),
                  'target_sha256': sha256(target), 'input_sha256': sha256(input_path),
                  'python': platform.python_version(), 'platform': platform.platform(),
                  'versions': versions, 'gpu': gpu_metadata(),
                  'peak_rss_kib': (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None),
                  'scope': 'neural processing only; excludes capture/background/audio/output'}
        encoded = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + '\n')
        print(encoded if args.json else f"{result['provider']} {args.width}x{args.height}: "
              f"{1000 / np.mean(samples['wall_ms']):.2f} processing FPS\n" + encoded)
        return 0
    except Exception as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()


if __name__ == '__main__':
    raise SystemExit(main())
