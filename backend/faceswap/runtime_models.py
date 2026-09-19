"""Versioned derived models. Never modify the supplied weights."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile


def artifact_directory(model: Path, cache: Path, precision: str, ort_version: str) -> Path:
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
            '--format=csv,noheader'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = 'cpu'
    try:
        trt = importlib.metadata.version('tensorrt_cu13')
    except importlib.metadata.PackageNotFoundError:
        trt = 'unavailable'
    metadata = {'model_sha256': digest.hexdigest(), 'precision': precision,
                'onnxruntime': ort_version, 'tensorrt': trt, 'gpu': gpu, 'format': 3}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    return directory


def fp16_model(model: Path, directory: Path) -> Path:
    target = directory / 'swapper-fp16.onnx'
    if target.exists():
        return target
    import onnx
    from onnxconverter_common.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
    graph = onnx.load(str(model))
    # InSwapper's variance/normalisation arithmetic can overflow FP16. Keep it FP32.
    half_ops = {'Conv', 'Gemm', 'Relu', 'LeakyRelu', 'Pad', 'Resize', 'Tanh'}
    blocked = set(DEFAULT_OP_BLOCK_LIST) | {node.op_type for node in graph.graph.node if node.op_type not in half_ops}
    converted = convert_float_to_float16(graph, keep_io_types=True, op_block_list=list(blocked))
    onnx.checker.check_model(converted)
    handle, temporary = tempfile.mkstemp(suffix='.onnx', dir=directory)
    os.close(handle)
    try:
        onnx.save(converted, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
