#!/usr/bin/env python
"""Phase 2/3 training entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nhsea.data import BackwardChainDataset, CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import BackwardChainConfig, CycleRegimeConfig, ForwardChainConfig
from nhsea.model import ModelConfig, TinyTransformer


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "cycle", "backward"], required=True)
    ap.add_argument("--variant", choices=["mechanism", "symmetric_control", "no_injection", "no_drift"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_size", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--init_checkpoint", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--run_dir", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_cfg = DatasetConfig(task=args.task, split="train", size=args.train_size, seed=args.seed)
    if args.task == "forward":
        gen_cfg = ForwardChainConfig()
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
        num_classes = 2
    elif args.task == "cycle":
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)
        num_classes = 4
    else:
        gen_cfg = BackwardChainConfig()
        dataset = BackwardChainDataset(data_cfg, gen_cfg)
        num_classes = 2

    model_cfg = ModelConfig(vocab_size=len(dataset.vocab))
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(num_classes)
    model.to(device)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    run_name = f"{args.task}_{args.variant}_seed{args.seed}"
    out_dir = Path(args.run_dir) if args.run_dir else Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    model.train()
    step = 0
    while step < args.steps:
        for input_ids, attn_mask, labels, _ in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant=args.variant,
                alpha=args.alpha,
                beta=args.beta,
            )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % args.log_every == 0:
                preds = torch.argmax(logits, dim=-1)
                acc = float((preds == labels).float().mean().item())
                record = {"step": step, "loss": float(loss.item()), "acc": acc}
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                print(f"step {step} loss {loss.item():.4f} acc {acc:.4f}")
            if step >= args.steps:
                break

    ckpt_path = out_dir / "checkpoint.pt"
    config_path = out_dir / "config.json"

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "variant": args.variant,
            "task": args.task,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "vocab": dataset.vocab,
            "git_commit": _git_commit(),
            "train_config": {
                "task": args.task,
                "variant": args.variant,
                "alpha": args.alpha,
                "beta": args.beta,
                "seed": args.seed,
                "train_size": args.train_size,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "init_checkpoint": args.init_checkpoint,
            },
        },
        ckpt_path,
    )
    config_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "variant": args.variant,
                "alpha": args.alpha,
                "beta": args.beta,
                "seed": args.seed,
                "train_size": args.train_size,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "init_checkpoint": args.init_checkpoint,
                "model_cfg": asdict(model_cfg),
                "gen_cfg": asdict(gen_cfg),
                "git_commit": _git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    log_f.close()
    print(f"Saved checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
