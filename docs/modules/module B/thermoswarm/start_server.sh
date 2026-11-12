#!/usr/bin/env bash
# One-command start of the ground server (Linux / macOS).
# First run creates .venv and installs requirements (Ubuntu 24.04 forbids system-wide pip).
#   ./start_server.sh            real drones
#   ./start_server.sh --sim      simulation
#   ./start_server.sh --http-host 0.0.0.0   UI reachable from other devices
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo ">>> creating .venv"
  if ! python3 -m venv .venv; then
    echo "!!! python3-venv is missing: sudo apt install -y python3-venv   and run again"; exit 1
  fi
  .venv/bin/pip install -q --upgrade pip
fi
# (re)install when requirements.txt changed since the last run (new versions add packages)
req_hash=$(sha256sum requirements.txt | cut -c1-16)
if [ "$(cat .venv/.req_hash 2>/dev/null || true)" != "$req_hash" ]; then
  echo ">>> installing requirements"
  .venv/bin/pip install -q -r requirements.txt
  echo "$req_hash" > .venv/.req_hash
fi
exec .venv/bin/python run_server.py "$@"
