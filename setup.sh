#!/usr/bin/env bash
set -euo pipefail

# Wind-Up Language Recognition: first-time setup.
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python3}"
WINDUP_VENV="${VENV_DIR:-.venv}"
printf 'Wind-Up Language Recognition setup\n'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { printf 'Error: Python 3.10+ is required. Set PYTHON to its executable.\n' >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.10+ is required.") if sys.version_info < (3, 10) else None'
printf 'Creating virtual environment: %s\n' "$WINDUP_VENV"
"$PYTHON_BIN" -m venv "$WINDUP_VENV"
WINDUP_PYTHON="$WINDUP_VENV/bin/python"
if [ ! -x "$WINDUP_PYTHON" ]; then
  printf 'Error: this script requires a POSIX virtual environment. On Windows, use the manual setup guide.\n' >&2
  exit 1
fi
printf 'Installing runtime dependencies...\n'
"$WINDUP_PYTHON" -m pip install --upgrade pip
"$WINDUP_PYTHON" -m pip install -r requirements.txt
"$WINDUP_PYTHON" -c 'import torch, ftfy; print("Dependencies ready; PyTorch", torch.__version__)'
if [ ! -f model.pt ]; then
  printf 'Error: model.pt is missing. Download the complete repository, including its bundled checkpoint.\n' >&2
  exit 1
fi
printf '\nSetup complete. Try:\n  "%s" language_detector.py predict "Hello, I need help signing in."\n' "$WINDUP_PYTHON"
printf 'Using Codex? AGENTS.md provides project context.\n'
