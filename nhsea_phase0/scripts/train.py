#!/usr/bin/env python
"""Phase 2/3 training entrypoint."""

from __future__ import annotations

import argparse
import json
import platform
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from nhsea.data import BackwardChainDataset, CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.data_phase3 import Phase3BackwardDataset, Phase3ForwardDataset
from nhsea.generators import BackwardChainConfig, CycleRegimeConfig, ForwardChainConfig
from nhsea.generators_phase3 import Phase3ChainConfig
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import rho_ratio, scale_norms


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


def _reserve_backup(path: Path) -> Path:
    base = path.name + ".bak"
    candidate = path.with_name(base)
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = path.with_name(f"{base}.{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    dataset_vocab: dict,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    batch_idx: int,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "model_cfg": asdict(model_cfg),
        "variant": args.variant,
        "task": args.task,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "phase3": args.phase3,
        "seed": args.seed,
        "vocab": dataset_vocab,
        "git_commit": _git_commit(),
        "train_state": {
            "step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "save_every_steps": args.save_every_steps,
        },
        "train_config": {
            "task": args.task,
            "variant": args.variant,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "phase3": args.phase3,
            "seed": args.seed,
            "train_size": args.train_size,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "init_checkpoint": args.init_checkpoint,
            "resume_from": args.resume_from,
            "resume_if_exists": args.resume_if_exists,
            "run_root": args.run_root,
            "run_id": args.run_id,
            "device": args.device,
        },
    }
    torch.save(payload, path)


def _u_symmetry_stats(
    model: nn.Module,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    variant: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _logits, _probe_logits, probe_U = model(
            input_ids,
            attn_mask=attn_mask,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            return_probe=True,
        )
    if was_training:
        model.train()
    U_np = probe_U.cpu().numpy()
    ratios = []
    for U in U_np:
        A, B0 = scale_norms(U, beta=1.0)
        ratios.append(rho_ratio(A, B0))
    if not ratios:
        return {"mean": 0.0, "median": 0.0}
    arr = np.asarray(ratios, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forward", "cycle", "backward"], required=True)
    ap.add_argument(
        "--variant",
        choices=[
            "mechanism",
            "symmetric_control",
            "symmetric_control_v2_normmatched",
            "no_injection",
            "no_drift",
        ],
        required=True,
    )
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_size", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--init_checkpoint", type=str, default="")
    ap.add_argument("--resume_if_exists", action="store_true")
    ap.add_argument("--resume_from", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--run_root", type=str, default="")
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--phase3", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)

    data_cfg = DatasetConfig(task=args.task, split="train", size=args.train_size, seed=args.seed)
    if args.task == "forward":
        if args.phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3ForwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = ForwardChainConfig()
            dataset = ForwardChainDataset(data_cfg, gen_cfg)
        num_classes = 2
    elif args.task == "cycle":
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)
        num_classes = 4
    else:
        if args.phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3BackwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = BackwardChainConfig()
            dataset = BackwardChainDataset(data_cfg, gen_cfg)
        num_classes = 2

    model_cfg = ModelConfig(vocab_size=len(dataset.vocab))
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(num_classes)
    model.to(device)

    run_name = f"{args.task}_{args.variant}_seed{args.seed}"
    run_id = args.run_id if args.run_id else run_name
    if args.run_dir:
        out_dir = Path(args.run_dir)
    elif args.run_root:
        out_dir = Path(args.run_root) / run_id
    else:
        out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path: Optional[Path] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume_if_exists:
        candidate = out_dir / "checkpoint_last.pt"
        if candidate.exists():
            resume_path = candidate

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    step = 0
    epoch = 0
    batch_idx = 0
    resume_loaded = False
    if resume_path is not None and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            if "opt_state" in ckpt:
                opt.load_state_dict(ckpt["opt_state"])
            train_state = ckpt.get("train_state", {})
            step = int(train_state.get("step", 0))
            epoch = int(train_state.get("epoch", 0))
            batch_idx = int(train_state.get("batch_idx", 0))
            resume_loaded = True
            print(f"Resuming from {resume_path} at step {step} (epoch {epoch}, batch {batch_idx})")
        except Exception as exc:
            print(f"Failed to load {resume_path}: {exc}. Starting fresh.")
            try:
                corrupt_path = resume_path.with_suffix(resume_path.suffix + ".corrupt")
                if not corrupt_path.exists():
                    resume_path.rename(corrupt_path)
            except Exception:
                pass
            resume_path = None

    if not resume_loaded and args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "variant": args.variant,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "phase3": args.phase3,
                "seed": args.seed,
                "train_size": args.train_size,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "init_checkpoint": args.init_checkpoint,
                "resume_from": args.resume_from,
                "resume_if_exists": args.resume_if_exists,
                "run_root": args.run_root,
                "run_id": run_id,
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "model_cfg": asdict(model_cfg),
                "gen_cfg": asdict(gen_cfg),
                "git_commit": _git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    log_path = out_dir / "train_log.jsonl"
    log_mode = "a" if resume_loaded else "w"
    if not resume_loaded and log_path.exists():
        try:
            backup = _reserve_backup(log_path)
            log_path.rename(backup)
        except Exception:
            pass
    log_f = log_path.open(log_mode, encoding="utf-8")

    drift_path = out_dir / "u_symmetry_drift.jsonl"
    drift_mode = "a" if resume_loaded else "w"
    drift_f = drift_path.open(drift_mode, encoding="utf-8")
    sample_count = min(args.batch_size, len(dataset))
    if sample_count > 0:
        sample_batch = [dataset[i] for i in range(sample_count)]
        sample_input_ids, sample_attn_mask, _labels, _meta = collate_batch(sample_batch)
        sample_input_ids = sample_input_ids.to(device)
        sample_attn_mask = sample_attn_mask.to(device)
    else:
        sample_input_ids = None
        sample_attn_mask = None

    stop_state = {"flag": False}

    def _handle_stop(_signum: int, _frame: object) -> None:
        stop_state["flag"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    def _log_symmetry(label: str, step_val: int) -> None:
        if sample_input_ids is None or sample_attn_mask is None:
            return
        stats = _u_symmetry_stats(
            model,
            sample_input_ids,
            sample_attn_mask,
            args.variant,
            args.alpha,
            args.beta,
            args.gamma,
        )
        record = {"step": step_val, "label": label, "ratio_mean": stats["mean"], "ratio_median": stats["median"]}
        drift_f.write(json.dumps(record) + "\n")
        drift_f.flush()

    if not resume_loaded:
        _log_symmetry("start", step)

    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size
    perm = _epoch_permutation(dataset_size, args.seed, epoch)
    mid_step = args.steps // 2
    logged_mid = False

    def _next_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
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
        return input_ids, attn_mask, labels

    last_ckpt = out_dir / "checkpoint_last.pt"
    final_ckpt = out_dir / "checkpoint_final.pt"
    compat_ckpt = out_dir / "checkpoint.pt"

    model.train()
    try:
        while step < args.steps:
            input_ids, attn_mask, labels = _next_batch()
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant=args.variant,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if not logged_mid and step >= mid_step:
                _log_symmetry("mid", step)
                logged_mid = True
            if step % args.log_every == 0:
                preds = torch.argmax(logits, dim=-1)
                acc = float((preds == labels).float().mean().item())
                record = {"step": step, "loss": float(loss.item()), "acc": acc}
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                print(f"step {step} loss {loss.item():.4f} acc {acc:.4f}")

            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                _save_checkpoint(
                    last_ckpt,
                    model,
                    opt,
                    model_cfg,
                    dataset.vocab,
                    args,
                    step,
                    epoch,
                    batch_idx,
                )

            if stop_state["flag"]:
                _save_checkpoint(
                    last_ckpt,
                    model,
                    opt,
                    model_cfg,
                    dataset.vocab,
                    args,
                    step,
                    epoch,
                    batch_idx,
                )
                _log_symmetry("stop", step)
                print("Stop requested; saved checkpoint_last.pt and exiting.")
                return 0

        _save_checkpoint(
            last_ckpt,
            model,
            opt,
            model_cfg,
            dataset.vocab,
            args,
            step,
            epoch,
            batch_idx,
        )
        _log_symmetry("end", step)
        _save_checkpoint(
            final_ckpt,
            model,
            opt,
            model_cfg,
            dataset.vocab,
            args,
            step,
            epoch,
            batch_idx,
        )
        _save_checkpoint(
            compat_ckpt,
            model,
            opt,
            model_cfg,
            dataset.vocab,
            args,
            step,
            epoch,
            batch_idx,
        )
        print(f"Saved checkpoint: {final_ckpt}")
    finally:
        log_f.close()
        drift_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
