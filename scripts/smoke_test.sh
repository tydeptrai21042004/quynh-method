#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
"$PY" -m pip install -q --no-build-isolation -e .
"$PY" -m safegrip.cli datasets >/dev/null
"$PY" -m pytest -q
printf '\nPASS: real-data-only CLI and unit suite are healthy.\n'
