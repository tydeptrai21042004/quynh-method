#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
"$PY" -m pip install -q --no-build-isolation -e .
"$PY" -m safegrip.cli prepare --dataset synthetic
"$PY" -m safegrip.cli benchmark --dataset synthetic --preset quick
"$PY" -m pytest -q
