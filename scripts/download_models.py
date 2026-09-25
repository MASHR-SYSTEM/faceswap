#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "backend"))

from faceswap.model_setup import INSIGHTFACE_TERMS_URL, SetupManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify FaceSwap's optional models")
    parser.add_argument("--accept-research-terms", action="store_true",
        help="confirm that your use is eligible under the InsightFace model terms")
    args = parser.parse_args()
    if not args.accept_research_terms:
        parser.error(f"review {INSIGHTFACE_TERMS_URL}, then pass --accept-research-terms if eligible")
    manager = SetupManager(MODELS, ROOT / ".cache/faceswap/setup.json")
    job = manager.start(accept_terms=True)
    while job["state"] not in {"ready", "error", "cancelled"}:
        print(f"\r{job['state']}: {job['progress']:.0%}", end="", flush=True)
        time.sleep(.5)
        job = manager.job(job["id"])
    print(f"\r{job['state']}: {job['progress']:.0%}")
    if job["state"] != "ready":
        print(job.get("error") or "Setup cancelled", file=sys.stderr)
        return 1
    manager.complete()
    print(f"Models are ready in {MODELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
