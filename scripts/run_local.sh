#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Run scripts/setup_local.sh first (Python 3.12 and Node 22 required)." >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  echo "UI build missing. Run npm ci --prefix frontend && npm run build." >&2
  exit 1
fi
source scripts/neural_env.sh
exec .venv/bin/python -X faulthandler backend/run_faceswap_app.py
