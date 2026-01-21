import subprocess
import sys


def test_v3_preflight_smoke(tmp_path):
    out_dir = tmp_path / "v3_preflight"
    result = subprocess.run(
        [sys.executable, "-m", "nhsea.v3_preflight", "--out", str(out_dir), "--smoke"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (out_dir / "v3_preflight_master.csv").exists()
    assert (out_dir / "v3_preflight_report.md").exists()
    assert (out_dir / "v3_preflight_report.json").exists()
