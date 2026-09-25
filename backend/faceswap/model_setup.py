"""Download and verify the optional, separately licensed model assets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
import zipfile


INSIGHTFACE_TERMS_URL = "https://github.com/deepinsight/insightface#license"


@dataclass(frozen=True)
class ModelAsset:
    id: str
    label: str
    url: str
    size: int
    sha256: str
    destination: str
    archive_members: tuple[tuple[str, str], ...] = ()


CATALOG = (
    ModelAsset(
        "inswapper", "InSwapper face transformation model",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
        554_253_681, "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        "inswapper_128.onnx",
    ),
    ModelAsset(
        "buffalo_l", "InsightFace detection and recognition models",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        288_621_354, "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        "insightface/models/buffalo_l",
        (("det_10g.onnx", "det_10g.onnx"), ("w600k_r50.onnx", "w600k_r50.onnx")),
    ),
    ModelAsset(
        "background", "MediaPipe background segmentation model",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
        250_177, "490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e",
        "mediapipe/selfie_segmenter_landscape.tflite",
    ),
)


class SetupManager:
    def __init__(self, models_dir: Path, state_path: Path):
        self.models_dir = models_dir
        self.state_path = state_path
        self.download_dir = state_path.parent / "downloads"
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._cancellations: dict[str, threading.Event] = {}

    def state(self) -> dict:
        saved = self._read_state()
        components = [self._component(asset) for asset in CATALOG]
        return {
            "terms_url": INSIGHTFACE_TERMS_URL,
            "terms_accepted": bool(saved.get("terms_accepted")),
            "completed": bool(saved.get("completed")),
            "download_bytes": sum(item.size for item in CATALOG),
            "required_free_bytes": 1_200_000_000,
            "components": components,
            "ready": all(item["ready"] for item in components),
        }

    def verify(self) -> dict:
        return self.state()

    def complete(self) -> dict:
        saved = self._read_state()
        saved["completed"] = True
        self._write_state(saved)
        return self.state()

    def start(self, *, accept_terms: bool, component_ids: list[str] | None = None) -> dict:
        if not accept_terms:
            raise ValueError("Accept the InsightFace non-commercial research terms before downloading.")
        selected = [asset for asset in CATALOG if component_ids is None or asset.id in component_ids]
        if not selected:
            raise ValueError("Select at least one model component.")
        unknown = set(component_ids or ()) - {asset.id for asset in CATALOG}
        if unknown:
            raise ValueError(f"Unknown model component: {sorted(unknown)[0]}")
        free = shutil.disk_usage(self.models_dir.parent).free
        needed = sum(asset.size for asset in selected if not self._asset_ready(asset))
        if free < max(needed + 300_000_000, 500_000_000):
            raise ValueError("Not enough free disk space. Free at least 1.2 GB and try again.")
        saved = self._read_state()
        saved.update({"terms_accepted": True, "terms_accepted_at": int(time.time())})
        self._write_state(saved)
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "state": "downloading", "progress": 0.0,
               "downloaded_bytes": 0, "total_bytes": sum(a.size for a in selected),
               "component": None, "error": None}
        cancellation = threading.Event()
        with self._lock:
            self._jobs[job_id] = job
            self._cancellations[job_id] = cancellation
        threading.Thread(target=self._run, args=(job_id, selected, cancellation), daemon=True).start()
        return dict(job)

    def job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            event = self._cancellations.get(job_id)
            job = self._jobs.get(job_id)
            if not event or not job:
                return None
            event.set()
            if job["state"] not in {"ready", "error", "cancelled"}:
                job["state"] = "cancelled"
            return dict(job)

    def _run(self, job_id: str, assets: list[ModelAsset], cancellation: threading.Event):
        downloaded = 0
        try:
            for asset in assets:
                if cancellation.is_set():
                    raise InterruptedError
                self._update(job_id, component=asset.id)
                if self._asset_ready(asset):
                    downloaded += asset.size
                    self._update(job_id, downloaded_bytes=downloaded,
                                 progress=downloaded / self._jobs[job_id]["total_bytes"])
                    continue
                archive = self._download(asset, cancellation, lambda value: self._update(
                    job_id, downloaded_bytes=downloaded + value,
                    progress=min(0.99, (downloaded + value) / self._jobs[job_id]["total_bytes"])))
                self._update(job_id, state="verifying")
                self._install(asset, archive)
                downloaded += asset.size
                self._update(job_id, state="downloading", downloaded_bytes=downloaded,
                             progress=downloaded / self._jobs[job_id]["total_bytes"])
            self._update(job_id, state="ready", component=None, progress=1.0)
        except InterruptedError:
            self._update(job_id, state="cancelled")
        except Exception as exc:
            self._update(job_id, state="error", error=str(exc))

    def _download(self, asset: ModelAsset, cancellation: threading.Event, progress) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        partial = self.download_dir / f"{asset.id}.part"
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == asset.size:
            digest = _sha256(partial)
            if digest == asset.sha256:
                progress(existing)
                return partial
            partial.unlink()
            existing = 0
        elif existing > asset.size:
            partial.unlink()
            existing = 0
        headers = {"User-Agent": "MASHr-FaceSwap/0.2", "Accept": "application/octet-stream"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(asset.url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = response.geturl()
            parsed = urllib.parse.urlparse(final_url)
            allowed_hosts = {"github.com", "release-assets.githubusercontent.com", "storage.googleapis.com"}
            if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
                raise RuntimeError("Model host redirected outside the approved secure hosts.")
            if existing and getattr(response, "status", None) != 206:
                existing = 0
            mode = "ab" if existing else "wb"
            with partial.open(mode) as output:
                count = existing
                while True:
                    if cancellation.is_set():
                        raise InterruptedError
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > asset.size:
                        raise RuntimeError(f"{asset.label} exceeded its expected size.")
                    output.write(chunk)
                    progress(count)
        if partial.stat().st_size != asset.size:
            raise RuntimeError(f"{asset.label} download was incomplete; retry will resume it.")
        digest = _sha256(partial)
        if digest != asset.sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{asset.label} failed its security checksum.")
        return partial

    def _install(self, asset: ModelAsset, downloaded: Path):
        destination = self.models_dir / asset.destination
        if not asset.archive_members:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".new")
            shutil.copyfile(downloaded, temporary)
            temporary.replace(destination)
            return
        staging = destination.with_name(destination.name + ".new")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(downloaded) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            for suffix, output_name in asset.archive_members:
                matches = [item for item in files if Path(item.filename).name == suffix]
                if len(matches) != 1 or matches[0].file_size > 500_000_000:
                    raise RuntimeError(f"The {asset.label} archive has an unexpected layout.")
                with archive.open(matches[0]) as source, (staging / output_name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)

    def _component(self, asset: ModelAsset) -> dict:
        return {"id": asset.id, "label": asset.label, "size": asset.size,
                "ready": self._asset_ready(asset), "state": "ready" if self._asset_ready(asset) else "missing"}

    def _asset_ready(self, asset: ModelAsset) -> bool:
        destination = self.models_dir / asset.destination
        if asset.archive_members:
            return all((destination / output).is_file() and (destination / output).stat().st_size > 1_000_000
                       for _, output in asset.archive_members)
        return destination.is_file() and destination.stat().st_size == asset.size

    def _update(self, job_id: str, **values):
        with self._lock:
            self._jobs[job_id].update(values)

    def _read_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_state(self, value: dict):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".new")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
