#!/usr/bin/env python
"""Phase 3c few-shot head-only adaptation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from nhsea.data import DatasetConfig, collate_batch
from nhsea.data_phase3 import Phase3BackwardDataset, Phase3ForwardDataset
from nhsea.generators_phase3 import Phase3ChainConfig
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


def _load_checkpoint(run_root: Path, source_task: str, variant: str, seed: int) -> Path:
    if source_task == "forward":
        ckpt = run_root / "forward_train" / variant / f"seed_{seed}" / "checkpoint_final.pt"
    else:
        ckpt = run_root / "train_backward" / variant / f"seed_{seed}" / "checkpoint_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _build_dataset(task: str, split: str, size: int, seed: int) -> object:
    cfg = DatasetConfig(task=task, split=split, size=size, seed=seed)
    gen_cfg = Phase3ChainConfig()
    if task == "forward":
        return Phase3ForwardDataset(cfg, gen_cfg)
    return Phase3BackwardDataset(cfg, gen_cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--variant", choices=["no_injection", "mechanism"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--source_task", choices=["forward", "backward"], required=True)
    ap.add_argument("--target_task", choices=["forward", "backward"], required=True)
    ap.add_argument("--tier", choices=["head"], default="head")
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)

    lock = json.loads(Path("phase2_lock.json").read_text())
    eval_size = int(lock["eval"]["forward"]["size"])
    batch_size = int(lock["batch_size"])
    lr = float(lock["optimizer"]["lr"])
    weight_decay = float(lock["optimizer"]["weight_decay"])

    out_dir = (
        Path(args.out_dir)
        / args.variant
        / f"seed{args.seed}"
        / f"{args.source_task}_to_{args.target_task}"
        / args.tier
        / f"n{args.n_train}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _load_checkpoint(Path(args.run_root), args.source_task, args.variant, args.seed)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))
    variant = str(ckpt.get("variant", args.variant))

    # Freeze everything except head.
    if args.tier != "head":
        raise ValueError("Only tier=head is supported in this phase")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    train_ds = _build_dataset(args.target_task, "adapt", args.n_train, args.seed)
    eval_ds = _build_dataset(args.target_task, "eval", eval_size, args.seed)

    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    log_path = out_dir / "adapt_train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    best_eval = -1.0
    step_at_best = 0
    final_eval = 0.0

    dataset_size = len(train_ds)
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    perm = _epoch_permutation(dataset_size, args.seed, 0)
    epoch = 0
    batch_idx = 0

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, args.seed, epoch)
        start = batch_idx * batch_size
        end = min(start + batch_size, dataset_size)
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
            for i in range(0, len(eval_ds), batch_size):
                batch = [eval_ds[j] for j in range(i, min(i + batch_size, len(eval_ds)))]
                input_ids, attn_mask, labels, _ = collate_batch(batch)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                logits, _, _ = model(
                    input_ids,
                    attn_mask=attn_mask,
                    variant=variant,
                    alpha=alpha,
                    beta=beta,
                )
                preds = torch.argmax(logits, dim=-1).cpu()
                correct += int((preds == labels).sum().item())
                total += int(labels.numel())
        model.train()
        acc = correct / total if total else 0.0
        return acc, correct, total

    model.train()
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
        )
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            eval_acc, correct, total = _eval()
            final_eval = eval_acc
            if eval_acc > best_eval:
                best_eval = eval_acc
                step_at_best = step
            ci_low, ci_high = wilson_ci(correct, total)
            record = {
                "step": step,
                "loss": float(loss.item()),
                "eval_acc": float(eval_acc),
                "eval_ci_low": float(ci_low),
                "eval_ci_high": float(ci_high),
            }
        else:
            record = {
                "step": step,
                "loss": float(loss.item()),
            }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    # Final eval stats
    final_acc, correct, total = _eval()
    ci_low, ci_high = wilson_ci(correct, total)

    config = {
        "run_root": args.run_root,
        "out_dir": str(out_dir),
        "variant": args.variant,
        "seed": args.seed,
        "source_task": args.source_task,
        "target_task": args.target_task,
        "tier": args.tier,
        "n_train": args.n_train,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "eval_size": eval_size,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "alpha": alpha,
        "beta": beta,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "checkpoint": str(ckpt_path),
        "model_cfg": asdict(model_cfg),
    }
    (out_dir / "adapt_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "source_task": args.source_task,
        "target_task": args.target_task,
        "tier": args.tier,
        "n_train": args.n_train,
        "final_acc": float(final_acc),
        "best_acc": float(best_eval),
        "step_at_best": int(step_at_best),
        "eval_size": int(total),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
    (out_dir / "adapt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "adapt_summary.csv").write_text("summary\n" + json.dumps(summary))

    print(f"Wrote {out_dir / 'adapt_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
