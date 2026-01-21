"""Leak gate for NHSEA v3 generator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .data_v3 import V3DatasetConfig, V3Dataset
from .generators_v3 import V3Config


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
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


def candidate_features(tokens: List[str], span_a: Tuple[int, int], span_b: Tuple[int, int]) -> List[float]:
    tok1 = tokens[span_a[0] : span_a[1]]
    tok2 = tokens[span_b[0] : span_b[1]]
    start1 = span_a[0]
    start2 = span_b[0]
    f_len = (len(tok1), len(tok2))
    f_pos = (start1 / len(tokens), start2 / len(tokens))
    f_char = (sum(len(t) for t in tok1), sum(len(t) for t in tok2))

    row: List[float] = []
    for (a, b) in (f_len, f_pos, f_char):
        row.extend([min(a, b), max(a, b), abs(a - b)])
    return row


def _candidate_spans(tokens: List[str]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if len(tokens) < 2:
        raise ValueError("tokens too short for candidate spans")
    # Candidates are fixed to the last two tokens.
    a = (len(tokens) - 2, len(tokens) - 1)
    b = (len(tokens) - 1, len(tokens))
    return a, b


def run_leak_gate(
    task: str,
    n: int,
    seed: int,
    cfg: V3Config,
    auroc_max: float,
) -> Tuple[dict, List[str]]:
    data_cfg = V3DatasetConfig(
        task=task,
        split="leak",
        size=n if n % 2 == 0 else n + 1,
        seed=seed,
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    dataset = V3Dataset(data_cfg, cfg)

    feats = []
    labels = []
    for idx in range(len(dataset)):
        inst = dataset[idx]["meta"]
        span1, span2 = _candidate_spans(inst.tokens)
        feats.append(candidate_features(inst.tokens, span1, span2))
        if task == "conclusion":
            labels.append(inst.true_index)
        else:
            labels.append(0 if inst.topology == "OBC" else 1)

    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    w = fit_logreg(Xz, y)
    scores = Xz @ w
    auroc = auc_rank(y, scores)

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
        "seed": seed,
        "n": len(dataset),
        "features": features_used,
        "auroc": float(auroc),
        "auroc_max": auroc_max,
        "passed": bool(auroc <= auroc_max),
        "v3_gen": asdict(cfg),
    }
    return report, features_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auroc_max", type=float, default=0.55)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--L_min", type=int, default=3)
    ap.add_argument("--L_max", type=int, default=6)
    ap.add_argument("--vocab_size", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = V3Config(T=args.T, M=args.M, K=args.K, L_min=args.L_min, L_max=args.L_max, vocab_size=args.vocab_size)

    task_map = {
        "forward": "conclusion",
        "backward": "topology",
    }

    for label, task in task_map.items():
        report, features = run_leak_gate(task, args.n, args.seed, cfg, args.auroc_max)
        report_path = out_dir / f"leak_gate_v3_{label}.json"
        feat_path = out_dir / f"leak_gate_v3_{label}_features.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        feat_path.write_text(json.dumps({"features": features, "task": task}, indent=2, sort_keys=True) + "\n")
        status = "PASS" if report["passed"] else "FAIL"
        print(f"{status} leak gate ({task}) AUROC={report['auroc']:.4f} -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
