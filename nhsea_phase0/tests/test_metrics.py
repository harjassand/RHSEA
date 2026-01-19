import numpy as np

from nhsea.metrics import (
    SelLocResult,
    cyc_stats,
    delta_cyc_stats,
    participation_ratio,
    permrow_null,
    prop_pipeline_a,
    prop_pipeline_b,
    token_sel_loc_gap,
)
from nhsea.operator import OperatorSpec
from nhsea.seeding import instance_seed_u32


def test_permrow_null_preserves_support_and_values():
    W = np.array(
        [
            [0.0, 0.2, 0.3],
            [0.4, 0.0, 0.0],
            [0.1, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    Wn = permrow_null(W, seed=123)
    from nhsea.topk import diagzero, rownorm_plus
    W_ref = rownorm_plus(diagzero(W))
    # Same support pattern (off-diagonal nonzeros)
    assert np.all((Wn > 0) == (W > 0))
    # Row-wise multiset of values preserved
    for a in range(W.shape[0]):
        assert np.allclose(np.sort(Wn[a]), np.sort(W_ref[a]))


def test_cyc_stats_delta_zero_when_uniform():
    # Equal weights off-diagonal; permutation should not change matrix.
    W = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    delta = delta_cyc_stats(W, "run", "inst", n_perm=5, ells=(2, 3, 4))
    for v in delta.values():
        assert np.isclose(v, 0.0)


def test_participation_ratio_two_node_swap():
    W = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    pr = participation_ratio(W, prem_prop_ids=[0])
    assert np.isclose(pr, 1.0)


def test_prop_pipelines_match_shapes():
    W_tok = np.zeros((4, 4), dtype=np.float64)
    W_tok[0, 1] = 1.0
    W_tok[1, 2] = 1.0
    prop_spans = [(0, 1), (2, 3)]
    Wa = prop_pipeline_a(W_tok, prop_spans, k_tok=1, k_prop=1)
    Wb = prop_pipeline_b(W_tok, prop_spans, k_prop=1)
    assert Wa.shape == (2, 2)
    assert Wb.shape == (2, 2)


def test_token_sel_loc_gap_zero_when_variants_equal():
    L = np.zeros((3, 3), dtype=np.float64)
    U = np.zeros((3, 3), dtype=np.float64)
    spec_mech = OperatorSpec(alpha=0.0, beta=0.0, variant="mechanism")
    spec_sym = OperatorSpec(alpha=0.0, beta=0.0, variant="symmetric_control")
    res = token_sel_loc_gap(
        L=L,
        U=U,
        spec_mech=spec_mech,
        spec_sym=spec_sym,
        k_tok=1,
        prem_tokens=[0],
        cand_true_tokens=[1],
        cand_false_tokens=[2],
        run_id="run",
        instance_id="inst",
    )
    assert isinstance(res, SelLocResult)
    assert np.isclose(res.sel_loc_gap, 0.0)


def test_permrow_determinism_across_runs_and_r():
    W = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.0, 0.5, 0.6],
            [0.7, 0.8, 0.0, 0.9],
            [1.0, 1.1, 1.2, 0.0],
        ],
        dtype=np.float64,
    )
    run_id = "run"
    instance_id = "inst"
    seed0 = instance_seed_u32(run_id, instance_id, salt="PERMROW_0")
    seed1 = instance_seed_u32(run_id, instance_id, salt="PERMROW_1")
    W0a = permrow_null(W, seed=seed0)
    W0b = permrow_null(W, seed=seed0)
    W1 = permrow_null(W, seed=seed1)
    assert np.allclose(W0a, W0b)
    assert not np.allclose(W0a, W1)
