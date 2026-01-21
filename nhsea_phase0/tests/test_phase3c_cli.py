from __future__ import annotations

import subprocess
import sys


def test_phase3c_adapt_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase3c_fewshot_adapt.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_phase3c_aggregate_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase3c_fewshot_aggregate.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
