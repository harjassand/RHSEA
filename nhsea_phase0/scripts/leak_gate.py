#!/usr/bin/env python
"""Phase 0.1: generator-only leak gate.

Runs a simple leak check using only permutation-invariant aggregates of
candidate-specific features, as required by the spec.

This is a minimal harness intended to catch obvious generator bugs or biases.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Bootstrap CI for AUROC using paired resamples."""
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=np.float64)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aucs[i] = auc_rank(y[idx], scores[idx])
    lo = float(np.quantile(aucs, alpha / 2.0))
    hi = float(np.quantile(aucs, 1.0 - alpha / 2.0))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", type=str, default="phase0")
    ap.add_argument("--auroc_max", type=float, default=0.55)
    ap.add_argument("--report", type=str, default="leak_gate_report.json")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.config:
        cfg_data = json.loads(Path(args.config).read_text())
        fwd_cfg = cfg_data.get("forward_gen", {})
        cfg = ForwardChainConfig(**fwd_cfg)
    else:
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

        row = []
        for (a, b) in (f_len, f_pos, f_char):
            row.extend([min(a, b), max(a, b), abs(a - b)])
        feats.append(row)
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
    auroc_ci = None
    if args.bootstrap > 0:
        lo, hi = bootstrap_ci(y, scores, n_boot=args.bootstrap, seed=args.seed, alpha=args.ci_alpha)
        auroc_ci = (lo, hi)

    if args.verbose:
        print("ForwardChainConfig:", asdict(cfg))
        print("AUROC:", auroc)

    features_used = [
        "len_min", "len_max", "len_absdiff",
        "pos_min", "pos_max", "pos_absdiff",
        "char_min", "char_max", "char_absdiff",
    ]
    report = {
        "run_id": args.run_id,
        "seed": args.seed,
        "n": args.n,
        "features": features_used,
        "auroc": float(auroc),
        "auroc_ci": auroc_ci,
        "auroc_max": args.auroc_max,
        "bootstrap": args.bootstrap,
        "passed": bool(auroc <= args.auroc_max),
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    if auroc > args.auroc_max:
        print(f"FAIL leak gate: AUROC={auroc:.4f} > {args.auroc_max}")
        return 2
    print(f"PASS leak gate: AUROC={auroc:.4f} <= {args.auroc_max}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
