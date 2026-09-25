#!/usr/bin/env python3
"""Build a native candidate archive; this does not publish or certify a release."""
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']


def module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def windows_version_file() -> Path:
    numeric = [int(part) for part in VERSION.split('-', 1)[0].split('.')]
    numeric = (numeric + [0, 0, 0, 0])[:4]
    if '-' in VERSION:
        numeric[3] = max(numeric[3], 1)
    template = (ROOT / 'installer/windows-version.txt.in').read_text(encoding='utf-8')
    destination = ROOT / 'build/windows-version.txt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('@NUMERIC_VERSION@', ','.join(map(str, numeric)))
                           .replace('@VERSION@', VERSION), encoding='utf-8')
    return destination


def main():
    if not (ROOT / 'frontend/dist/index.html').is_file():
        raise SystemExit('Build the production frontend first: npm run build')
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise SystemExit('Candidate packaging supports Windows/Linux x86-64 only')
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
               '--name', 'FaceSwap', '--paths', str(ROOT / 'backend'),
               '--add-data', f'{ROOT / "frontend/dist"}:frontend/dist',
               '--add-data', f'{ROOT / "pyproject.toml"}:.',
               '--collect-data', 'cv2', '--collect-all', 'sounddevice',
               '--hidden-import', 'faceswap.app', '--hidden-import', 'tkinter',
               '--hidden-import', 'cv2_enumerate_cameras' if platform.system() == 'Windows' else 'fcntl']
    if platform.system() == 'Windows':
        command.extend(['--windowed', '--version-file', str(windows_version_file())])
    for module in ('onnxruntime', 'insightface', 'mediapipe', 'onnx', 'onnxconverter_common'):
        if module_exists(module):
            command.extend(['--collect-all', module])
    # ONNX Runtime preload_dlls(directory="") discovers these redistributable
    # packages at runtime. PyInstaller must preserve their DLL directories.
    for module in ('nvidia.cuda_nvrtc', 'nvidia.cuda_runtime', 'nvidia.cublas',
                   'nvidia.cufft', 'nvidia.curand', 'nvidia.cudnn', 'nvidia.nvjitlink'):
        if module_exists(module):
            command.extend(['--collect-all', module])
    # Assets are chosen explicitly by the public export; never collect user files.
    for path in sorted((ROOT / 'assets').glob('*.png')):
        if path.name in {'aging-cyber-monk-target.png', 'default-target.png',
                         'optimus.png',
                         'mashr-female.png', 'ordo-basement-cyber-warehouse-photoreal.png'}:
            command.extend(['--add-data', f'{path}:assets'])
    command.append(str(ROOT / 'backend/launcher.py'))
    subprocess.run(command, cwd=ROOT, check=True)
    bundle = ROOT / 'dist/FaceSwap'
    notices = bundle / 'third-party-notices'
    notices.mkdir(exist_ok=True)
    for document in (ROOT / 'docs/public').glob('*.md'):
        shutil.copy2(document, bundle / document.name)
    inventory = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get('Name', '')):
        name = distribution.metadata.get('Name', 'unknown')
        inventory.append({'name': name, 'version': distribution.version,
                          'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License')})
        for entry in distribution.files or []:
            if any(term in entry.name.lower() for term in ('license', 'copying', 'notice')):
                source = distribution.locate_file(entry)
                if source.is_file():
                    destination = notices / name / str(entry).replace('..', '_')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
    # Minified frontend bundles still require their dependency notices.
    frontend_inventory = []
    lock = json.loads((ROOT / 'frontend/package-lock.json').read_text(encoding="utf-8"))
    for relative, metadata in sorted(lock.get('packages', {}).items()):
        if not relative or metadata.get('dev'):
            continue
        package = ROOT / 'frontend' / relative
        manifest = package / 'package.json'
        if not manifest.is_file() and metadata.get('optional'):
            continue  # Platform-specific optional dependency not included in this build.
        if not manifest.is_file():
            raise RuntimeError(f'Missing frontend dependency: {relative}; run npm ci')
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        name = identity.get('name', relative)
        frontend_inventory.append({'name': name, 'version': identity.get('version'),
                                   'license': identity.get('license')})
        for source in package.iterdir():
            if source.is_file() and any(term in source.name.lower() for term in ('license', 'copying', 'notice')):
                destination = notices / 'frontend' / name / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    (bundle / 'frontend-dependency-inventory.json').write_text(json.dumps(frontend_inventory, indent=2))
    for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(ROOT / name, bundle / name)
    (bundle / 'dependency-inventory.json').write_text(json.dumps(inventory, indent=2))
    profile = 'cuda' if module_exists('onnxruntime') and any(d['name'] == 'onnxruntime-gpu' for d in inventory) else 'cpu'
    stem = f'FaceSwap-{VERSION}-{platform.system().lower()}-x64-{profile}'
    archive = shutil.make_archive(str(ROOT / 'dist' / stem), 'zip' if platform.system() == 'Windows' else 'gztar',
                                  root_dir=bundle.parent, base_dir=bundle.name)
    digest = hashlib.file_digest(open(archive, 'rb'), 'sha256').hexdigest()
    Path(archive + '.sha256').write_text(f'{digest}  {Path(archive).name}\n')
    print(archive)


if __name__ == '__main__':
    main()
