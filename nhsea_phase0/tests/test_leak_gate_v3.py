from nhsea.generators_v3 import V3Config, generate_v3_instance
from nhsea.leak_gate_v3 import candidate_features


def test_v3_leak_gate_invariant_features():
    cfg = V3Config()
    inst = generate_v3_instance("run", "i0", "pair0", "OBC", "conclusion", cfg)
    span0 = (cfg.T - 2, cfg.T - 1)
    span1 = (cfg.T - 1, cfg.T)
    feats_01 = candidate_features(inst.tokens, span0, span1)
    feats_10 = candidate_features(inst.tokens, span1, span0)
    assert feats_01 == feats_10
