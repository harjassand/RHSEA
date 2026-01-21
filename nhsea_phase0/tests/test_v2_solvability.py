from pathlib import Path
import sys

from nhsea.generators_v2 import V2MappingConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_v2_solvability import audit_task


def test_v2_solvability_rate_low():
    cfg = V2MappingConfig(n_symbols=16, n_facts=8, T=64, vocab_size=200)
    stats = audit_task("forward", cfg, seeds=[0], eval_size=256)
    assert stats["ambiguous_rate"] <= 0.01
