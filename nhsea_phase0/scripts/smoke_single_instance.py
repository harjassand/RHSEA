#!/usr/bin/env python
"""Phase 0.4: single-instance pipeline smoke test (no training).

Creates one synthetic instance, runs operator -> clamp -> sigmoid -> TopK,
computes a few basic metrics, and prints a compact sanity report.
"""

from __future__ import annotations

import argparse

import numpy as np

from nhsea.generators import ForwardChainConfig, generate_forward_chain, candidate_token_spans, premise_token_set
from nhsea.operator import OperatorSpec, build_run_operator, token_weights
from nhsea.topk import topk_rownorm


def sel_loc_gap(W: np.ndarray, prem: list[int], cand_true: list[int], cand_false: list[int]) -> float:
    """Selectivity localization gap: mean prem->true minus prem->false mass."""
    prem = np.asarray(prem, dtype=np.int64)
    true = np.asarray(cand_true, dtype=np.int64)
    false = np.asarray(cand_false, dtype=np.int64)
    if prem.size == 0 or true.size == 0 or false.size == 0:
        return 0.0
    mass_true = W[np.ix_(prem, true)].mean()
    mass_false = W[np.ix_(prem, false)].mean()
    return float(mass_true - mass_false)


def cycle_weight(W: np.ndarray, k: int) -> float:
    """Average cycle weight of length k via trace(W^k)/T."""
    if k < 2:
        raise ValueError("k must be >= 2")
    T = W.shape[0]
    wk = np.linalg.matrix_power(W, k)
    return float(np.trace(wk) / T)


def preference_ratio(W: np.ndarray, prem: list[int], cand_true: list[int], cand_false: list[int]) -> float:
    """Ratio of prem->true mass to total prem->candidate mass."""
    prem = np.asarray(prem, dtype=np.int64)
    true = np.asarray(cand_true, dtype=np.int64)
    false = np.asarray(cand_false, dtype=np.int64)
    if prem.size == 0 or true.size == 0 or false.size == 0:
        return 0.0
    mass_true = W[np.ix_(prem, true)].sum()
    mass_false = W[np.ix_(prem, false)].sum()
    denom = mass_true + mass_false
    if denom == 0:
        return 0.0
    return float(mass_true / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="phase0")
    ap.add_argument("--instance_id", type=str, default="smoke0")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--variant", type=str, default="mechanism")
    args = ap.parse_args()

    cfg = ForwardChainConfig()
    inst = generate_forward_chain(args.run_id, args.instance_id, cfg)
    T = len(inst.tokens)

    rng = np.random.default_rng(args.seed)
    L = rng.normal(size=(T, T))
    U = rng.normal(size=(T, T))

    spec = OperatorSpec(alpha=args.alpha, beta=args.beta, variant=args.variant)
    O = build_run_operator(L, U, spec)
    W = token_weights(O)
    W_hat = topk_rownorm(W, k=args.k)

    cand1, cand2 = candidate_token_spans(inst)
    prem = list(premise_token_set(inst))
    true_cand = cand1 if inst.true_index == 0 else cand2
    false_cand = cand2 if inst.true_index == 0 else cand1

    metrics = {
        "SelLocGap": sel_loc_gap(W_hat, prem, true_cand, false_cand),
        "DeltaCyc2": cycle_weight(W_hat, 2),
        "DeltaCyc3": cycle_weight(W_hat, 3),
        "DeltaCyc4": cycle_weight(W_hat, 4),
        "PR": preference_ratio(W_hat, prem, true_cand, false_cand),
    }

    row_sums = W_hat.sum(axis=1)
    print("Smoke report:")
    print(f"- tokens: {T}")
    print(f"- W_hat shape: {W_hat.shape}")
    print(f"- row sum min/max: {row_sums.min():.4f} / {row_sums.max():.4f}")
    print(f"- nonzeros: {int(np.count_nonzero(W_hat))}")
    print(f"- true candidate index: {inst.true_index}")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
