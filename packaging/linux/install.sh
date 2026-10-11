#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
.venv/bin/python -m pip install -e '.[ui,reports]'
echo "Install complete. Run ./packaging/linux/start.sh"
