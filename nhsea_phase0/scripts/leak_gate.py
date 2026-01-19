#!/usr/bin/env python
"""Phase 0.1: generator-only leak gate.

Runs a simple leak check using only permutation-invariant aggregates of
candidate-specific features, as required by the spec.

This is a minimal harness intended to catch obvious generator bugs or biases.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from nhsea.generators import ForwardChainConfig, generate_forward_chain


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via rank statistic; y_true in {0,1}."""
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # average ranks for ties
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie handling
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.1, l2: float = 1e-2) -> np.ndarray:
    """Tiny deterministic logistic regression (batch GD)."""
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="phase0")
    ap.add_argument("--auroc_max", type=float, default=0.55)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()


    cfg = ForwardChainConfig()

    feats = []
    labels = []
    for i in range(args.n):
        inst = generate_forward_chain(f"{args.run_id}_seed{args.seed}", f"i{i}", cfg)
        c1 = inst.candidates[0]
        c2 = inst.candidates[1]
        span1 = inst.prop_spans[c1]
        span2 = inst.prop_spans[c2]
        tok1 = inst.tokens[span1[0]:span1[1]+1]
        tok2 = inst.tokens[span2[0]:span2[1]+1]
        start1 = span1[0]
        start2 = span2[0]
        # candidate-specific scalar features (allowed once symmetrized):
        f_len = (len(tok1), len(tok2))
        f_pos = (start1 / len(inst.tokens), start2 / len(inst.tokens))
        f_char = (sum(len(t) for t in tok1), sum(len(t) for t in tok2))

        for (a, b) in (f_len, f_pos, f_char):
            feats.append([min(a, b), max(a, b), abs(a - b)])
            labels.append(inst.true_index)

    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    # Standardize features deterministically.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    w = fit_logreg(Xz, y)
    scores = Xz @ w
    auroc = auc_rank(y, scores)

    if args.verbose:
        print("ForwardChainConfig:", asdict(cfg))
        print("AUROC:", auroc)

    if auroc > args.auroc_max:
        print(f"FAIL leak gate: AUROC={auroc:.4f} > {args.auroc_max}")
        return 2
    print(f"PASS leak gate: AUROC={auroc:.4f} <= {args.auroc_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
