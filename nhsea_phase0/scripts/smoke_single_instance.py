#!/usr/bin/env python
"""Phase 0.4: single-instance pipeline smoke test (no training)."""

from __future__ import annotations

import argparse

import numpy as np

from nhsea.generators import ForwardChainConfig, generate_forward_chain, candidate_token_spans, premise_token_set
from nhsea.metrics import (
    delta_cyc_stats,
    participation_ratio,
    prop_pipeline_a,
    prop_pipeline_b,
    token_sel_loc_gap,
)
from nhsea.operator import OperatorSpec, build_run_operator, token_weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="phase0")
    ap.add_argument("--instance_id", type=str, default="smoke0")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--k_tok", type=int, default=8)
    ap.add_argument("--k_prop", type=int, default=2)
    args = ap.parse_args()

    # Keep premises small so the matched-random pool can cover |A_prem| in a tiny smoke instance.
    cfg = ForwardChainConfig(n_prem=2)
    inst = generate_forward_chain(args.run_id, args.instance_id, cfg)
    T = len(inst.tokens)

    rng = np.random.default_rng(args.seed)
    L = rng.normal(size=(T, T))
    U = rng.normal(size=(T, T))

    cand1, cand2 = candidate_token_spans(inst)
    prem = list(premise_token_set(inst))
    true_cand = cand1 if inst.true_index == 0 else cand2
    false_cand = cand2 if inst.true_index == 0 else cand1

    spec_mech = OperatorSpec(alpha=args.alpha, beta=args.beta, variant="mechanism")
    spec_sym = OperatorSpec(alpha=args.alpha, beta=args.beta, variant="symmetric_control")

    sel = token_sel_loc_gap(
        L=L,
        U=U,
        spec_mech=spec_mech,
        spec_sym=spec_sym,
        k_tok=args.k_tok,
        prem_tokens=prem,
        cand_true_tokens=true_cand,
        cand_false_tokens=false_cand,
        run_id=args.run_id,
        instance_id=args.instance_id,
    )

    O_mech = build_run_operator(L, U, spec_mech)
    W_tok = token_weights(O_mech)
    W_prop_a = prop_pipeline_a(W_tok, inst.prop_spans, k_tok=args.k_tok, k_prop=args.k_prop)
    W_prop_b = prop_pipeline_b(W_tok, inst.prop_spans, k_prop=args.k_prop)

    delta_a = delta_cyc_stats(W_prop_a, args.run_id, args.instance_id, n_perm=100, ells=(2, 3, 4), salt_prefix="PERMROW_A")
    delta_b = delta_cyc_stats(W_prop_b, args.run_id, args.instance_id, n_perm=100, ells=(2, 3, 4), salt_prefix="PERMROW_B")

    pr_a = participation_ratio(W_prop_a, inst.premises)
    pr_b = participation_ratio(W_prop_b, inst.premises)

    row_sums_a = W_prop_a.sum(axis=1)
    row_sums_b = W_prop_b.sum(axis=1)
    print("Smoke report:")
    print(f"- tokens: {T}")
    print(f"- W_tok shape: {W_tok.shape}")
    print(f"- W_prop_A shape: {W_prop_a.shape}")
    print(f"- W_prop_B shape: {W_prop_b.shape}")
    print(f"- row sum A min/max: {row_sums_a.min():.4f} / {row_sums_a.max():.4f}")
    print(f"- row sum B min/max: {row_sums_b.min():.4f} / {row_sums_b.max():.4f}")
    print(f"- nonzeros W_prop_A: {int(np.count_nonzero(W_prop_a))}")
    print(f"- nonzeros W_prop_B: {int(np.count_nonzero(W_prop_b))}")
    print(f"- true candidate index: {inst.true_index}")
    print(f"- SelLocGap: {sel.sel_loc_gap:.6f}")
    print(f"- DeltaCyc2_A: {delta_a[2]:.6f}")
    print(f"- DeltaCyc3_A: {delta_a[3]:.6f}")
    print(f"- DeltaCyc4_A: {delta_a[4]:.6f}")
    print(f"- DeltaCyc2_B: {delta_b[2]:.6f}")
    print(f"- DeltaCyc3_B: {delta_b[3]:.6f}")
    print(f"- DeltaCyc4_B: {delta_b[4]:.6f}")
    print(f"- PR_A: {pr_a:.6f}")
    print(f"- PR_B: {pr_b:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
