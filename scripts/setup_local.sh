#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v node >/dev/null || { echo "Install Node.js 22 first." >&2; exit 1; }
node -e 'if (Number(process.versions.node.split(".")[0]) < 22) process.exit(1)'
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-base.lock
npm ci --prefix frontend
npm run build
printf 'Base app ready. Run scripts/run_local.sh, then open http://127.0.0.1:7865\nInstall the optional neural runtime and models before neural swapping. See docs/LOCAL_ALPHA.md.\n'
