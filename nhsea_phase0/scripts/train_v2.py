#!/usr/bin/env python
"""NHSEA v2 baseline training (no injection)."""

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
from torch import nn

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


def _epoch_permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "backward"], required=True)
    ap.add_argument("--variant", default="no_injection")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train_size", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--run_root", type=str, default="runs/v2")
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--n_symbols", type=int, default=16)
    ap.add_argument("--n_facts", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=200)
    args = ap.parse_args()

    if args.variant != "no_injection":
        raise ValueError("Only variant=no_injection is supported in v2 preflight")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)

    gen_cfg = V2MappingConfig(
        n_symbols=args.n_symbols,
        n_facts=args.n_facts,
        T=args.T,
        vocab_size=args.vocab_size,
    )
    data_cfg = V2DatasetConfig(
        task=args.task,
        split="train",
        size=args.train_size,
        seed=args.seed,
        T=args.T,
        n_symbols=args.n_symbols,
        n_facts=args.n_facts,
        vocab_size=args.vocab_size,
    )
    dataset = V2MappingDataset(data_cfg, gen_cfg)

    model_cfg = ModelConfig(vocab_size=len(dataset.vocab), T=args.T)
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.to(device)

    alpha = 0.0
    beta = 0.0
    gamma = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    run_id = args.run_id or f"v2_{args.task}_train_seed{args.seed}"
    out_dir = Path(args.run_root) / f"{args.task}_train" / args.variant / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task": args.task,
        "variant": args.variant,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "seed": args.seed,
        "train_size": args.train_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "run_root": args.run_root,
        "run_id": run_id,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_cfg": asdict(model_cfg),
        "gen_cfg": asdict(gen_cfg),
        "git_commit": _git_commit(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size
    perm = _epoch_permutation(dataset_size, args.seed, 0)
    epoch = 0
    batch_idx = 0

    model.train()
    for step in range(1, args.steps + 1):
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, args.seed, epoch)
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, dataset_size)
        idxs = perm[start:end]
        batch = [dataset[int(i)] for i in idxs]
        batch_idx += 1

        input_ids, attn_mask, labels, _ = collate_batch(batch)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant=args.variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        log_f.write(json.dumps({"step": step, "loss": float(loss.item())}) + "\n")
        if step % 1000 == 0:
            log_f.flush()

    log_f.flush()
    log_f.close()

    ckpt = {
        "task": args.task,
        "variant": args.variant,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "seed": args.seed,
        "model_cfg": asdict(model_cfg),
        "model_state": model.state_dict(),
        "vocab": dataset.vocab,
        "gen_cfg": asdict(gen_cfg),
        "train_size": args.train_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "run_id": run_id,
        "run_root": args.run_root,
        "git_commit": _git_commit(),
    }
    torch.save(ckpt, out_dir / "checkpoint_final.pt")
    print(f"Wrote checkpoint: {out_dir / 'checkpoint_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
