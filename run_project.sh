#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  printf '%s\n' 'Environnement Python absent. Dans le dossier du projet, exécuter :' 'python3 -m venv .venv' '.venv/bin/python -m pip install -r requirements.txt' >&2
  exit 1
fi

exec "$PYTHON" "$PROJECT_DIR/scripts/run_local.py" "$@"
