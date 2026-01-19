import subprocess
import sys


def test_leak_gate_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/leak_gate.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
