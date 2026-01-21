#!/usr/bin/env python
"""NHSEA v2 few-shot head-only adaptation (baseline only)."""

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


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _epoch_permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(size)


def _load_checkpoint(run_root: Path, source_task: str, seed: int) -> Path:
    ckpt = run_root / f"{source_task}_train" / "no_injection" / f"seed_{seed}" / "checkpoint_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _build_dataset(task: str, split: str, size: int, seed: int, gen_cfg: V2MappingConfig) -> V2MappingDataset:
    cfg = V2DatasetConfig(
        task=task,
        split=split,
        size=size,
        seed=seed,
        T=gen_cfg.T,
        n_symbols=gen_cfg.n_symbols,
        n_facts=gen_cfg.n_facts,
        vocab_size=gen_cfg.vocab_size,
    )
    return V2MappingDataset(cfg, gen_cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--source_task", choices=["forward", "backward"], required=True)
    ap.add_argument("--target_task", choices=["forward", "backward"], required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)

    ckpt_path = _load_checkpoint(Path(args.run_root), args.source_task, args.seed)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))
    gamma = float(ckpt.get("gamma", 1.0))
    variant = str(ckpt.get("variant", "no_injection"))

    if variant != "no_injection":
        raise ValueError("Only variant=no_injection is supported in v2 preflight")

    # Freeze everything except head.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    gen_cfg = V2MappingConfig(**ckpt["gen_cfg"])
    train_ds = _build_dataset(args.target_task, "adapt", args.n_train, args.seed, gen_cfg)
    eval_ds = _build_dataset(args.target_task, "eval", args.eval_size, args.seed, gen_cfg)

    opt = torch.optim.AdamW(model.head.parameters(), lr=ckpt.get("lr", 3e-4), weight_decay=ckpt.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss()

    out_dir = (
        Path(args.out_dir)
        / "no_injection"
        / f"seed{args.seed}"
        / f"{args.source_task}_to_{args.target_task}"
        / "head"
        / f"n{args.n_train}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "adapt_train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    best_eval = -1.0
    step_at_best = 0
    best_ci = (0.0, 1.0)

    dataset_size = len(train_ds)
    batches_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size
    perm = _epoch_permutation(dataset_size, args.seed, 0)
    epoch = 0
    batch_idx = 0

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, args.seed, epoch)
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, dataset_size)
        idxs = perm[start:end]
        batch = [train_ds[int(i)] for i in idxs]
        batch_idx += 1
        input_ids, attn_mask, labels, _ = collate_batch(batch)
        return input_ids, attn_mask, labels

    def _eval() -> Tuple[float, int, int]:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(eval_ds), args.batch_size):
                batch = [eval_ds[j] for j in range(i, min(i + args.batch_size, len(eval_ds)))]
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
        model.train()
        acc = correct / total if total else 0.0
        return acc, correct, total

    model.train()
    final_acc = 0.0
    final_correct = 0
    final_total = 0

    for step in range(1, args.steps + 1):
        input_ids, attn_mask, labels = _next_batch()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            eval_acc, correct, total = _eval()
            final_acc, final_correct, final_total = eval_acc, correct, total
            ci_low, ci_high = wilson_ci(correct, total)
            record = {
                "step": step,
                "loss": float(loss.item()),
                "eval_acc": float(eval_acc),
                "eval_ci_low": float(ci_low),
                "eval_ci_high": float(ci_high),
            }
            if eval_acc > best_eval:
                best_eval = eval_acc
                step_at_best = step
                best_ci = (ci_low, ci_high)
        else:
            record = {
                "step": step,
                "loss": float(loss.item()),
            }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    log_f.close()

    final_ci_low, final_ci_high = wilson_ci(final_correct, final_total)

    summary = {
        "variant": variant,
        "seed": args.seed,
        "source_task": args.source_task,
        "target_task": args.target_task,
        "n_train": args.n_train,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "final_acc": float(final_acc),
        "final_ci_low": float(final_ci_low),
        "final_ci_high": float(final_ci_high),
        "final_correct": int(final_correct),
        "final_total": int(final_total),
        "best_acc": float(best_eval),
        "best_ci_low": float(best_ci[0]),
        "best_ci_high": float(best_ci[1]),
        "step_at_best": int(step_at_best),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "checkpoint": str(ckpt_path),
        "model_cfg": asdict(model_cfg),
        "gen_cfg": asdict(gen_cfg),
    }

    summary_path = out_dir / "adapt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
