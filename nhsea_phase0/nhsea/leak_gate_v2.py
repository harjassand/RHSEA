"""Leak-gate utilities for NHSEA v2 generator."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .generators_v2 import V2MappingConfig, generate_v2_mapping


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via rank statistic; y_true in {0,1}."""
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
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
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=np.float64)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aucs[i] = auc_rank(y[idx], scores[idx])
    lo = float(np.quantile(aucs, alpha / 2.0))
    hi = float(np.quantile(aucs, 1.0 - alpha / 2.0))
    return lo, hi


def candidate_features(tokens: List[str], span_a: Tuple[int, int], span_b: Tuple[int, int]) -> List[float]:
    tok1 = tokens[span_a[0] : span_a[1] + 1]
    tok2 = tokens[span_b[0] : span_b[1] + 1]
    start1 = span_a[0]
    start2 = span_b[0]
    f_len = (len(tok1), len(tok2))
    f_pos = (start1 / len(tokens), start2 / len(tokens))
    f_char = (sum(len(t) for t in tok1), sum(len(t) for t in tok2))

    row: List[float] = []
    for (a, b) in (f_len, f_pos, f_char):
        row.extend([min(a, b), max(a, b), abs(a - b)])
    return row


def build_leak_features(
    task: str,
    n: int,
    seed: int,
    run_id: str,
    cfg: V2MappingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[List[float]] = []
    labels: List[int] = []
    run_key = f"{run_id}_{task}_seed{seed}"
    for i in range(n):
        instance_id = f"i{i}"
        inst = generate_v2_mapping(run_key, instance_id, cfg, task)
        span1 = inst.candidate_spans[0]
        span2 = inst.candidate_spans[1]
        feats.append(candidate_features(inst.tokens, span1, span2))
        labels.append(inst.true_index)
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def run_leak_gate(
    task: str,
    n: int,
    seed: int,
    run_id: str,
    auroc_max: float,
    bootstrap: int,
    ci_alpha: float,
    cfg: V2MappingConfig,
) -> Tuple[dict, List[str]]:
    X, y = build_leak_features(task, n=n, seed=seed, run_id=run_id, cfg=cfg)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    w = fit_logreg(Xz, y)
    scores = Xz @ w
    auroc = auc_rank(y, scores)
    auroc_ci = None
    if bootstrap > 0:
        auroc_ci = bootstrap_ci(y, scores, n_boot=bootstrap, seed=seed, alpha=ci_alpha)

    features_used = [
        "len_min",
        "len_max",
        "len_absdiff",
        "pos_min",
        "pos_max",
        "pos_absdiff",
        "char_min",
        "char_max",
        "char_absdiff",
    ]

    report = {
        "task": task,
        "run_id": f"{run_id}_{task}_seed{seed}",
        "seed": seed,
        "n": n,
        "features": features_used,
        "auroc": float(auroc),
        "auroc_ci": auroc_ci,
        "auroc_max": auroc_max,
        "bootstrap": bootstrap,
        "passed": bool(auroc <= auroc_max),
        "v2_gen": asdict(cfg),
    }
    return report, features_used


def write_leak_gate_report(report: dict, report_path: Path, features: List[str], features_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    features_path.write_text(json.dumps({"features": features, "task": report.get("task")}, indent=2, sort_keys=True))
