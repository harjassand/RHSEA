from nhsea.generators_v2 import V2MappingConfig, generate_v2_mapping
from nhsea.leak_gate_v2 import candidate_features


def test_v2_bijection_forward_and_backward():
    cfg = V2MappingConfig(n_symbols=16, n_facts=8, T=64, vocab_size=200)
    inst_fwd = generate_v2_mapping("run", "i0", cfg, "forward")
    true_token_fwd = inst_fwd.candidates[inst_fwd.true_index]
    assert true_token_fwd == f"B{inst_fwd.mapping[inst_fwd.query_index]:02d}"

    inst_bwd = generate_v2_mapping("run", "i1", cfg, "backward")
    true_token_bwd = inst_bwd.candidates[inst_bwd.true_index]
    assert true_token_bwd == f"A{inst_bwd.query_index:02d}"


def test_v2_candidate_symmetry_and_positions():
    cfg = V2MappingConfig(n_symbols=16, n_facts=8, T=64, vocab_size=200)
    inst = generate_v2_mapping("run", "i2", cfg, "forward")
    span0, span1 = inst.candidate_spans
    len0 = span0[1] - span0[0] + 1
    len1 = span1[1] - span1[0] + 1
    assert len0 == len1 == 1
    assert span0[0] == cfg.T - 2
    assert span1[0] == cfg.T - 1
    assert len(inst.tokens) == cfg.T


def test_v2_determinism_same_seed_same_instance():
    cfg = V2MappingConfig(n_symbols=16, n_facts=8, T=64, vocab_size=200)
    inst1 = generate_v2_mapping("run", "i3", cfg, "forward")
    inst2 = generate_v2_mapping("run", "i3", cfg, "forward")
    assert inst1.tokens == inst2.tokens
    assert inst1.candidates == inst2.candidates
    assert inst1.true_index == inst2.true_index


def test_v2_leak_gate_invariant_features():
    cfg = V2MappingConfig(n_symbols=16, n_facts=8, T=64, vocab_size=200)
    inst = generate_v2_mapping("run", "i4", cfg, "forward")
    span0, span1 = inst.candidate_spans
    feats_01 = candidate_features(inst.tokens, span0, span1)
    feats_10 = candidate_features(inst.tokens, span1, span0)
    assert feats_01 == feats_10
