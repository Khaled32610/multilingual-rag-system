#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ ! -x .venv/bin/streamlit ]]; then
  echo "Missing .venv. Run ./packaging/linux/install.sh first." >&2
  exit 1
fi
exec .venv/bin/streamlit run src/rag_system/ui/app.py --server.address 127.0.0.1 --server.port 8501
