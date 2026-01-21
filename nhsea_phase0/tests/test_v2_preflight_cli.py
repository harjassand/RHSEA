import json
import subprocess
import sys


def test_v2_preflight_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "nhsea.phase_v2_preflight", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_v2_preflight_cli_smoke(tmp_path):
    out_path = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nhsea.phase_v2_preflight",
            "--n",
            "16",
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["n"] == 16
