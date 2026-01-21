#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/v3_preflight}"

python -m nhsea.leak_gate_v3 --out "${OUT_DIR}/leak"
python -m nhsea.v3_preflight --out "${OUT_DIR}"
