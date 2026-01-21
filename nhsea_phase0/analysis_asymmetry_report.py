#!/usr/bin/env python
"""Compute asymmetry metrics for L/U across trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from nhsea.generators import (
    ForwardChainConfig,
    BackwardChainConfig,
    CycleRegimeConfig,
    generate_forward_chain,
    generate_backward_chain,
    generate_cycle_regime,
)
from nhsea.generators_phase3 import Phase3ChainConfig, generate_phase3_forward, generate_phase3_backward
from nhsea.model import ModelConfig, TinyTransformer

PROBE_LAYER = 2

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


def _asym_ratio(mat: np.ndarray) -> float:
    num = np.linalg.norm(mat - mat.T)
    den = np.linalg.norm(mat + mat.T) + 1e-12
    return float(num / den)


def _encode(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[t] for t in tokens]


def _eval_run_id(task: str, seed: int, phase3: bool) -> str:
    if phase3:
        return f"phase3_{task}_eval_seed{seed}"
    return f"{task}_eval_seed{seed}"


def _task_token(vocab: Dict[str, int], task: str) -> int | None:
    key = "TASK_FWD" if task == "forward" else "TASK_BWD"
    return vocab.get(key)


def _iter_instances(
    task: str,
    seed: int,
    phase3: bool,
    eval_size: int,
) -> List[object]:
    if task == "forward":
        if phase3:
            cfg = Phase3ChainConfig()
            run_id = _eval_run_id("forward", seed, phase3=True)
            return [generate_phase3_forward(run_id, f"eval_{i}", cfg) for i in range(eval_size)]
        cfg = ForwardChainConfig()
        run_id = _eval_run_id("forward", seed, phase3=False)
        return [generate_forward_chain(run_id, f"eval_{i}", cfg) for i in range(eval_size)]
    if task == "backward":
        if phase3:
            cfg = Phase3ChainConfig()
            run_id = _eval_run_id("backward", seed, phase3=True)
            return [generate_phase3_backward(run_id, f"eval_{i}", cfg) for i in range(eval_size)]
        cfg = BackwardChainConfig()
        run_id = _eval_run_id("backward", seed, phase3=False)
        return [generate_backward_chain(run_id, f"eval_{i}", cfg) for i in range(eval_size)]
    if task == "cycle":
        cfg = CycleRegimeConfig()
        run_id = _eval_run_id("cycle", seed, phase3=False)
        return [generate_cycle_regime(run_id, f"eval_{i}", cfg) for i in range(eval_size)]
    raise ValueError(f"Unsupported task: {task}")


def _batch_iter(items: List[object], batch_size: int) -> List[List[object]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _compute_stats(
    model: TinyTransformer,
    instances: List[object],
    vocab: Dict[str, int],
    task: str,
    variant: str,
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
) -> Dict[str, float]:
    asym_L_vals: List[float] = []
    asym_U_vals: List[float] = []
    A_vals: List[float] = []
    B0_vals: List[float] = []

    task_tok = _task_token(vocab, task)
    for batch in _batch_iter(instances, 64):
        input_ids = []
        attn_mask = []
        for inst in batch:
            ids = _encode(inst.tokens, vocab)
            if task_tok is not None:
                ids[0] = task_tok
            mask = [1 if tok != 0 else 0 for tok in ids]
            input_ids.append(ids)
            attn_mask.append(mask)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn_mask_t = torch.tensor(attn_mask, dtype=torch.long, device=device)

        with torch.no_grad():
            _logits, probe_logits, probe_U = model(
                input_ids_t,
                attn_mask=attn_mask_t,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                return_probe=True,
            )
        L_np = probe_logits.cpu().numpy()
        U_np = probe_U.cpu().numpy()
        for i in range(L_np.shape[0]):
            L = L_np[i].mean(axis=0)
            U = U_np[i]
            asym_L_vals.append(_asym_ratio(L))
            asym_U_vals.append(_asym_ratio(U))
            A = np.linalg.norm(beta * (U - U.T))
            B0 = np.linalg.norm(beta * (U + U.T))
            A_vals.append(float(A))
            B0_vals.append(float(B0))

    A_median = float(np.median(np.asarray(A_vals))) if A_vals else 0.0
    B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
    if B0_median == 0.0:
        gamma_est = 1.0
    else:
        gamma_est = A_median / B0_median
    if variant == "symmetric_control_v2_normmatched":
        B_median = gamma_est * B0_median
    else:
        B_median = B0_median
    if variant == "mechanism":
        used_term = A_median
    elif variant in ("symmetric_control", "symmetric_control_v2_normmatched"):
        used_term = B_median
    elif variant == "no_drift":
        used_term = A_median
    else:
        used_term = 0.0

    return {
        "asym_L_median": float(np.median(np.asarray(asym_L_vals))) if asym_L_vals else 0.0,
        "asym_U_median": float(np.median(np.asarray(asym_U_vals))) if asym_U_vals else 0.0,
        "A_median": A_median,
        "B0_median": B0_median,
        "gamma": float(gamma_est),
        "B_median": float(B_median),
        "used_term_norm_median": float(used_term),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs")
    ap.add_argument("--out_csv", type=str, default="asymmetry_master.csv")
    ap.add_argument("--out_report", type=str, default="asymmetry_report.md")
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    root = Path(args.root)
    ckpts = list(root.rglob("checkpoint_final.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {root}")

    rows: List[Dict[str, object]] = []
    device = _resolve_device(args.device)

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        task = str(ckpt.get("task", ""))
        variant = str(ckpt.get("variant", ""))
        seed = int(ckpt.get("seed", 0))
        phase3 = bool(ckpt.get("phase3", False))

        eval_tasks: List[str]
        if task == "multitask":
            eval_tasks = ["forward", "backward"]
            phase3 = True
        elif task in ("forward", "backward", "cycle"):
            eval_tasks = [task]
        else:
            continue

        model_cfg = ModelConfig(**ckpt["model_cfg"])
        model = TinyTransformer(model_cfg, probe_layer=PROBE_LAYER)
        if task == "cycle":
            model.set_num_classes(4)
        else:
            model.set_num_classes(2)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        vocab = ckpt.get("vocab")
        if vocab is None:
            # fallback to minimal vocab
            vocab = {}

        alpha = float(ckpt.get("alpha", 0.0))
        beta = float(ckpt.get("beta", 0.0))
        gamma = float(ckpt.get("gamma", 1.0))

        for eval_task in eval_tasks:
            instances = _iter_instances(eval_task, seed, phase3, args.eval_size)
            stats = _compute_stats(
                model=model,
                instances=instances,
                vocab=vocab,
                task=eval_task,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                device=device,
            )
            rows.append(
                {
                    "checkpoint": str(ckpt_path),
                    "train_task": task,
                    "eval_task": eval_task,
                    "variant": variant,
                    "seed": seed,
                    "phase3": phase3,
                    "asym_L_median": stats["asym_L_median"],
                    "asym_U_median": stats["asym_U_median"],
                    "A_median": stats["A_median"],
                    "B0_median": stats["B0_median"],
                    "gamma": stats["gamma"],
                    "B_median": stats["B_median"],
                    "used_term_norm_median": stats["used_term_norm_median"],
                    "n_eval": len(instances),
                }
            )

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "train_task",
                "eval_task",
                "variant",
                "seed",
                "phase3",
                "asym_L_median",
                "asym_U_median",
                "A_median",
                "B0_median",
                "gamma",
                "B_median",
                "used_term_norm_median",
                "n_eval",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Report summary focusing on baseline asymmetry.
    def _subset(variant: str) -> List[float]:
        return [float(r["asym_L_median"]) for r in rows if r["variant"] == variant]

    baseline_vals = _subset("no_injection")
    mech_vals = _subset("mechanism")
    sym_vals = _subset("symmetric_control_v2_normmatched")

    def _stats(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.min(arr)), float(np.median(arr)), float(np.max(arr))

    b_min, b_med, b_max = _stats(baseline_vals)
    m_min, m_med, m_max = _stats(mech_vals)
    s_min, s_med, s_max = _stats(sym_vals)

    seeds = sorted({int(r["seed"]) for r in rows})
    seed_text = ", ".join(str(s) for s in seeds) if seeds else "none"
    report_lines = [
        "# Asymmetry Report",
        "",
        f"baseline(no_injection) asym_L median range: min={b_min:.4f}, median={b_med:.4f}, max={b_max:.4f}",
        f"mechanism asym_L median range: min={m_min:.4f}, median={m_med:.4f}, max={m_max:.4f}",
        f"symmetric_control_v2 asym_L median range: min={s_min:.4f}, median={s_med:.4f}, max={s_max:.4f}",
        "",
        "Interpretation:",
        "If baseline asym_L medians are materially above 0, L already carries directionality prior to injection.",
        "",
        "Measured object definition:",
        f"- L: raw attention logits (QK^T/sqrt(d_head)) at probe layer {PROBE_LAYER}, pre-bias, pre-mask, pre-clamp; heads averaged.",
        f"- U: raw u_q/u_k pathway at probe layer {PROBE_LAYER}, pre-clamp.",
        "- Mask: attn_mask is applied inside attention, but asymmetry uses logits_raw before masking; padded positions remain in L/U.",
        f"- Eval: eval_size={args.eval_size} per checkpoint; generator run_id seeded from ckpt seed(s): {seed_text}.",
    ]

    out_report = Path(args.out_report)
    out_report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_report}")
    print(
        "Asymmetry measured on raw pre-mask logits at probe layer "
        f"{PROBE_LAYER} (heads averaged), eval_size={args.eval_size}, seeds={seed_text}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
