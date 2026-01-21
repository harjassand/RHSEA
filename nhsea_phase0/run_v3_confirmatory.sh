#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/v3_confirmatory}"

python -m nhsea.v3_confirmatory --out "${OUT_DIR}" --resume_if_exists
