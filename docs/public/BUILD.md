# Build from source

Use Python 3.12 and Node.js 22. On Ubuntu:

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-base.lock
npm ci --prefix frontend
npm run build
.venv/bin/python backend/launcher.py
```

On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\python.exe` for the
Python commands. Source checkouts use the same UI/API as packages.

Run `python -m pytest -q`, `npm run build` and the frontend Playwright suite.
Build a native directory package with `python scripts/build_package.py` in a
clean build environment with `pyinstaller==6.22.0` installed. Build on the target
operating system; this is not cross compilation. GPU/model support is optional:
install `requirements-release-cpu.lock` or `requirements-release-cuda.lock` into
that environment before packaging, then run
`python -m pip install --no-deps -r requirements-model-adapters.lock`.
The runtime lock supplies their dependencies explicitly. Both adapters use the
single OpenCV contrib distribution: InsightFace's metadata names opencv-python,
while MediaPipe names opencv-contrib-python. Installing both would overwrite the
same cv2 module. Consequently pip check reports the unsatisfied distribution
name for InsightFace; validate imports and inference separately. Model weights
are never build inputs.

The build collects dependency notices/inventory. Review those notices and native
library redistribution terms before publishing any package. Public builds use
only this repository, without private source access. Packages and checksums are
candidate artifacts until their hardware acceptance checklist passes.
