#!/usr/bin/env python
"""NHSEA v2 evaluation for forward/backward tasks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from nhsea.data_v2 import V2DatasetConfig, V2MappingDataset, collate_batch
from nhsea.generators_v2 import V2MappingConfig
from nhsea.model import ModelConfig, TinyTransformer


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--eval_task", choices=["forward", "backward"], required=True)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    gen_cfg = V2MappingConfig(**ckpt["gen_cfg"])
    data_cfg = V2DatasetConfig(
        task=args.eval_task,
        split="eval",
        size=args.eval_size,
        seed=int(ckpt.get("seed", 0)),
        T=gen_cfg.T,
        n_symbols=gen_cfg.n_symbols,
        n_facts=gen_cfg.n_facts,
        vocab_size=gen_cfg.vocab_size,
    )
    dataset = V2MappingDataset(data_cfg, gen_cfg)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))
    gamma = float(ckpt.get("gamma", 1.0))
    variant = str(ckpt.get("variant", "no_injection"))

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
            input_ids, attn_mask, labels, _ = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            preds = torch.argmax(logits, dim=-1).cpu()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    acc = correct / total if total else 0.0
    ci_low, ci_high = wilson_ci(correct, total)

    train_task = str(ckpt.get("task", ""))
    mode = "in_task" if train_task == args.eval_task else "zero_shot"

    summary = {
        "train_task": train_task,
        "eval_task": args.eval_task,
        "mode": mode,
        "variant": variant,
        "seed": int(ckpt.get("seed", 0)),
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "acc": float(acc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "correct": int(correct),
        "total": int(total),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "model_cfg": asdict(model_cfg),
        "gen_cfg": asdict(gen_cfg),
        "checkpoint": str(ckpt_path),
    }

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{args.eval_task}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
