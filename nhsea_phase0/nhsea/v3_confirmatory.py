"""NHSEA v3 confirmatory mechanism runner."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

from .data_v3 import V3Dataset, V3DatasetConfig, collate_batch
from .generators_v3 import V3Config
from .metrics import token_adj_loc_gap
from .model import ModelConfig, TinyTransformer
from .operator import OperatorSpec, build_run_operator, rho_ratio, scale_norms, token_weights
from .topk import diagzero, rownorm_plus, topk_rownorm


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


def _bootstrap_ci(values: List[float], seed: int = 0, n_boot: int = 10000) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(arr[idx]))
    low = float(np.quantile(boots, 0.025))
    high = float(np.quantile(boots, 0.975))
    return low, high


def _epoch_permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(size)


def _save_checkpoint(
    path: Path,
    model: TinyTransformer,
    opt: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    vocab: Dict[str, int],
    cfg: V3Config,
    meta: Dict[str, object],
) -> None:
    ckpt = {
        "task": meta["task"],
        "variant": meta["variant"],
        "alpha": meta["alpha"],
        "beta": meta["beta"],
        "gamma": meta["gamma"],
        "seed": meta["seed"],
        "model_cfg": asdict(model_cfg),
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "vocab": vocab,
        "gen_cfg": asdict(cfg),
        "train_size": meta["train_size"],
        "steps": meta["steps"],
        "batch_size": meta["batch_size"],
        "lr": meta["lr"],
        "weight_decay": meta["weight_decay"],
        "git_commit": _git_commit(),
        "train_state": {
            "step": meta["step"],
            "epoch": meta["epoch"],
            "batch_idx": meta["batch_idx"],
        },
    }
    torch.save(ckpt, path)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _train_model(
    task: str,
    variant: str,
    seed: int,
    cfg: V3Config,
    train_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
    run_root: Path,
    save_every: int,
    resume_if_exists: bool,
) -> Path:
    data_cfg = V3DatasetConfig(
        task=task,
        split="train",
        size=train_size,
        seed=seed,
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    dataset = V3Dataset(data_cfg, cfg)

    model_cfg = ModelConfig(vocab_size=len(dataset.vocab), T=cfg.T)
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    out_dir = run_root / "train_conclusion" / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = out_dir / "checkpoint_last.pt" if resume_if_exists else None
    step = 0
    epoch = 0
    batch_idx = 0
    if resume_path is not None and resume_path.exists():
        ckpt = _load_checkpoint(resume_path)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        train_state = ckpt.get("train_state", {})
        step = int(train_state.get("step", 0))
        epoch = int(train_state.get("epoch", 0))
        batch_idx = int(train_state.get("batch_idx", 0))

    config = {
        "task": task,
        "variant": variant,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "seed": seed,
        "train_size": train_size,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_cfg": asdict(model_cfg),
        "gen_cfg": asdict(cfg),
        "git_commit": _git_commit(),
        "resume_from": str(resume_path) if resume_path is not None else "",
        "resume_step": step,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    log_path = out_dir / "train_log.jsonl"
    log_mode = "a" if step > 0 and log_path.exists() else "w"
    log_f = log_path.open(log_mode, encoding="utf-8")

    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    perm = _epoch_permutation(dataset_size, seed, epoch)

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, seed, epoch)
        start = batch_idx * batch_size
        end = min(start + batch_size, dataset_size)
        idxs = perm[start:end]
        batch = [dataset[int(i)] for i in idxs]
        batch_idx += 1
        input_ids, attn_mask, labels, _ = collate_batch(batch)
        return input_ids, attn_mask, labels

    last_ckpt = out_dir / "checkpoint_last.pt"
    final_ckpt = out_dir / "checkpoint_final.pt"

    model.train()
    while step < steps:
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

        step += 1
        if step % 100 == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = float((preds == labels).float().mean().item())
            record = {"step": step, "loss": float(loss.item()), "acc": acc}
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

        if save_every > 0 and step % save_every == 0:
            meta = {
                "task": task,
                "variant": variant,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "seed": seed,
                "train_size": train_size,
                "steps": steps,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "step": step,
                "epoch": epoch,
                "batch_idx": batch_idx,
            }
            _save_checkpoint(last_ckpt, model, opt, model_cfg, dataset.vocab, cfg, meta)

    meta = {
        "task": task,
        "variant": variant,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "seed": seed,
        "train_size": train_size,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
    }
    _save_checkpoint(last_ckpt, model, opt, model_cfg, dataset.vocab, cfg, meta)
    _save_checkpoint(final_ckpt, model, opt, model_cfg, dataset.vocab, cfg, meta)
    log_f.close()
    return final_ckpt


def _eval_dataset(
    model: TinyTransformer,
    dataset: V3Dataset,
    device: torch.device,
    variant: str,
    alpha: float,
    beta: float,
    gamma: float,
    batch_size: int,
    instances_path: Path | None,
) -> Dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    records: List[dict] = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, attn_mask, labels, metas = collate_batch(batch)
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
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.numpy()
            correct += int((preds == labels_np).sum())
            total += int(labels_np.size)

            if instances_path is not None:
                for idx, meta in enumerate(metas):
                    instance_key = f"{meta.pair_id}:{meta.topology}"
                    records.append(
                        {
                            "instance_key": instance_key,
                            "pair_id": meta.pair_id,
                            "topology": meta.topology,
                            "label": int(labels_np[idx]),
                            "pred": int(preds[idx]),
                        }
                    )

    if instances_path is not None:
        instances_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(instances_path, "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    acc = correct / total if total else 0.0
    ci_low, ci_high = wilson_ci(correct, total)
    return {
        "acc": float(acc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "correct": int(correct),
        "total": int(total),
    }


def _eval_model(
    ckpt_path: Path,
    eval_task: str,
    eval_size: int,
    batch_size: int,
    device: torch.device,
    write_instances: bool = False,
) -> Dict[str, object]:
    ckpt = _load_checkpoint(ckpt_path)
    cfg = V3Config(**ckpt["gen_cfg"])

    data_cfg = V3DatasetConfig(
        task=eval_task,
        split="eval",
        size=eval_size,
        seed=int(ckpt["seed"]),
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    dataset = V3Dataset(data_cfg, cfg)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    variant = str(ckpt["variant"])
    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))
    gamma = float(ckpt.get("gamma", 1.0))

    out_path = ckpt_path.parent / f"eval_{eval_task}_summary.json"
    inst_path = ckpt_path.parent / f"eval_{eval_task}_instances.jsonl.gz" if write_instances else None
    summary = _eval_dataset(
        model=model,
        dataset=dataset,
        device=device,
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        batch_size=batch_size,
        instances_path=inst_path,
    )
    summary.update(
        {
            "train_task": str(ckpt.get("task", "")),
            "eval_task": eval_task,
            "mode": "in_task" if str(ckpt.get("task")) == eval_task else "zero_shot",
            "variant": variant,
            "seed": int(ckpt.get("seed", 0)),
            "eval_size": eval_size,
            "batch_size": batch_size,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": _git_commit(),
            "model_cfg": asdict(model_cfg),
            "gen_cfg": asdict(cfg),
            "checkpoint": str(ckpt_path),
        }
    )
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _fewshot_adapt(
    run_root: Path,
    variant: str,
    seed: int,
    n_train: int,
    steps: int,
    eval_every: int,
    eval_size: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    ckpt_path = run_root / "train_conclusion" / variant / f"seed_{seed}" / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path)

    cfg = V3Config(**ckpt["gen_cfg"])

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    train_size = n_train if n_train % 2 == 0 else n_train + 1
    train_cfg = V3DatasetConfig(
        task="topology",
        split="adapt",
        size=train_size,
        seed=seed,
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    eval_cfg = V3DatasetConfig(
        task="topology",
        split="eval",
        size=eval_size,
        seed=seed,
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )

    train_ds = V3Dataset(train_cfg, cfg)
    eval_ds = V3Dataset(eval_cfg, cfg)

    opt = torch.optim.AdamW(model.head.parameters(), lr=float(ckpt.get("lr", 3e-4)), weight_decay=float(ckpt.get("weight_decay", 0.01)))
    criterion = nn.CrossEntropyLoss()

    out_dir = run_root / "adapt" / variant / f"seed_{seed}" / f"n{n_train}" / "head"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "adapt_train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    best_eval = -1.0
    step_at_best = 0
    best_ci = (0.0, 1.0)

    dataset_size = len(train_ds)
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    perm = _epoch_permutation(dataset_size, seed, 0)
    epoch = 0
    batch_idx = 0

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal epoch, batch_idx, perm
        if batch_idx >= batches_per_epoch:
            epoch += 1
            batch_idx = 0
            perm = _epoch_permutation(dataset_size, seed, epoch)
        start = batch_idx * batch_size
        end = min(start + batch_size, dataset_size)
        idxs = perm[start:end]
        batch = [train_ds[int(i)] for i in idxs]
        batch_idx += 1
        input_ids, attn_mask, labels, _ = collate_batch(batch)
        return input_ids, attn_mask, labels

    variant_ckpt = str(ckpt.get("variant", "no_injection"))
    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))
    gamma = float(ckpt.get("gamma", 1.0))

    def _eval() -> Tuple[float, int, int]:
        summary = _eval_dataset(
            model=model,
            dataset=eval_ds,
            device=device,
            variant=variant_ckpt,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            batch_size=batch_size,
            instances_path=None,
        )
        return float(summary["acc"]), int(summary["correct"]), int(summary["total"])

    model.train()
    final_acc = 0.0
    final_correct = 0
    final_total = 0

    for step in range(1, steps + 1):
        input_ids, attn_mask, labels = _next_batch()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant=variant_ckpt,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps:
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
            record = {"step": step, "loss": float(loss.item())}
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    log_f.close()

    # Final eval with instance-level outputs.
    inst_path = out_dir / "eval_instances.jsonl.gz"
    eval_summary = _eval_dataset(
        model=model,
        dataset=eval_ds,
        device=device,
        variant=variant_ckpt,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        batch_size=batch_size,
        instances_path=inst_path,
    )

    final_ci_low, final_ci_high = wilson_ci(final_correct, final_total)

    summary = {
        "variant": variant_ckpt,
        "seed": seed,
        "source_task": "conclusion",
        "target_task": "topology",
        "n_train": n_train,
        "steps": steps,
        "eval_every": eval_every,
        "eval_size": eval_size,
        "batch_size": batch_size,
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
        "gen_cfg": asdict(cfg),
        "eval_instances": str(inst_path),
        "eval_summary": eval_summary,
    }

    summary_path = out_dir / "adapt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def _prem_tokens(meta) -> List[int]:
    tokens: List[int] = []
    for pid in meta.premises:
        start, end = meta.spans[pid]
        tokens.extend(list(range(start, end)))
    return tokens


def _candidate_tokens(meta, T: int) -> Tuple[List[int], List[int]]:
    if meta.true_index == 0:
        return [T - 2], [T - 1]
    return [T - 1], [T - 2]


def _collapse_to_prop(W_tok: np.ndarray, spans: List[Tuple[int, int]]) -> np.ndarray:
    sets = [np.arange(s, e, dtype=np.int64) for (s, e) in spans]
    M = len(sets)
    W_prop = np.zeros((M, M), dtype=np.float64)
    for a, idx_a in enumerate(sets):
        for b, idx_b in enumerate(sets):
            sub = W_tok[np.ix_(idx_a, idx_b)]
            if sub.size == 0:
                W_prop[a, b] = 0.0
            else:
                W_prop[a, b] = float(sub.mean())
    return W_prop


def _seed_vector(indices: Iterable[int], size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float64)
    idx = list(indices)
    if not idx:
        return v
    scale = 1.0 / np.sqrt(len(idx))
    v[np.asarray(idx, dtype=np.int64)] = scale
    return v


def _propagate_normalize(W: np.ndarray, v0: np.ndarray, steps: int) -> np.ndarray:
    if steps == 0:
        v = v0.astype(np.float64)
    else:
        v = np.linalg.matrix_power(W, steps) @ v0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.zeros_like(v, dtype=np.float64)
    return v / norm


def _compute_gamma_stats(
    dataset: V3Dataset,
    model: TinyTransformer,
    variant: str,
    alpha: float,
    beta: float,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    beta_eff = 0.0 if variant == "no_injection" else beta
    A_vals: List[float] = []
    B0_vals: List[float] = []
    zero_count = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, attn_mask, _labels, _metas = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _logits, _probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=1.0,
                return_probe=True,
            )
            U_np = probe_U.cpu().numpy()
            for idx in range(U_np.shape[0]):
                U = U_np[idx]
                A, B0 = scale_norms(U, beta_eff)
                A_vals.append(float(A))
                B0_vals.append(float(B0))
                if B0 == 0.0:
                    zero_count += 1
                total += 1

    A_median = float(np.median(np.asarray(A_vals))) if A_vals else 0.0
    B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
    gamma = 1.0 if B0_median == 0.0 else A_median / B0_median
    zero_rate = 0.0 if total == 0 else zero_count / float(total)

    if variant == "symmetric_control_v2_normmatched":
        rho_vals = [rho_ratio(a, gamma * b0) for a, b0 in zip(A_vals, B0_vals)]
        flag = not (B0_median > 0.0 and zero_rate <= 0.001)
    else:
        rho_vals = [rho_ratio(a, b0) for a, b0 in zip(A_vals, B0_vals)]
        flag = not (0.9 <= float(np.median(np.asarray(rho_vals))) <= 1.1)

    rho_median = float(np.median(np.asarray(rho_vals))) if rho_vals else 0.0

    return {
        "A_median": A_median,
        "B0_median": B0_median,
        "gamma": float(gamma),
        "zero_rate_B0": float(zero_rate),
        "rho_median": rho_median,
        "flag": bool(flag),
    }


def _sel_locgap_and_topology(
    ckpt_path: Path,
    eval_size: int,
    batch_size: int,
    k_tok: int,
    k_prop: int,
    device: torch.device,
    paired_dir: Path,
    bootstrap: int,
) -> Dict[str, object]:
    ckpt = _load_checkpoint(ckpt_path)
    cfg = V3Config(**ckpt["gen_cfg"])

    data_cfg = V3DatasetConfig(
        task="conclusion",
        split="eval",
        size=eval_size,
        seed=int(ckpt["seed"]),
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    dataset = V3Dataset(data_cfg, cfg)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    variant = str(ckpt["variant"])
    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))

    gamma_stats = _compute_gamma_stats(dataset, model, variant, alpha, beta, device, batch_size)
    gamma = float(gamma_stats["gamma"])

    spec_mech = OperatorSpec(alpha=alpha, beta=beta, variant="mechanism")
    spec_sym = OperatorSpec(alpha=alpha, beta=beta, gamma=gamma, variant="symmetric_control_v2_normmatched")
    spec_variant = OperatorSpec(
        alpha=alpha,
        beta=beta,
        gamma=gamma if variant == "symmetric_control_v2_normmatched" else None,
        variant=variant,
    )

    sel_by_topo: Dict[str, List[float]] = {"OBC": [], "PBC": []}
    pr_by_topo: Dict[str, List[float]] = {"OBC": [], "PBC": []}
    mass_by_topo: Dict[str, List[float]] = {"OBC": [], "PBC": []}
    per_pair: Dict[str, Dict[str, Dict[str, float]]] = {}

    model.eval()
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, attn_mask, _labels, metas = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _logits, probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=1.0,
                return_probe=True,
            )
            L_np = probe_logits.cpu().numpy()
            U_np = probe_U.cpu().numpy()

            for idx, meta in enumerate(metas):
                L = L_np[idx].mean(axis=0)
                U = U_np[idx]
                prem_tokens = _prem_tokens(meta)
                cand_true, cand_false = _candidate_tokens(meta, cfg.T)

                adj_mech = token_adj_loc_gap(
                    L=L,
                    U=U,
                    spec=spec_mech,
                    k_tok=k_tok,
                    prem_tokens=prem_tokens,
                    cand_true_tokens=cand_true,
                    cand_false_tokens=cand_false,
                    run_id=meta.run_id,
                    instance_id=meta.instance_id,
                )
                adj_sym = token_adj_loc_gap(
                    L=L,
                    U=U,
                    spec=spec_sym,
                    k_tok=k_tok,
                    prem_tokens=prem_tokens,
                    cand_true_tokens=cand_true,
                    cand_false_tokens=cand_false,
                    run_id=meta.run_id,
                    instance_id=meta.instance_id,
                )
                sel = float(adj_mech - adj_sym)
                sel_by_topo[meta.topology].append(sel)

                O_mat = build_run_operator(L, U, spec_variant)
                W_tok = token_weights(O_mat)
                W_tok_hat = topk_rownorm(W_tok, k_tok)
                W_prop = _collapse_to_prop(W_tok_hat, meta.spans)
                W_prop_hat = rownorm_plus(diagzero(topk_rownorm(W_prop, k_prop)))

                u0 = _seed_vector(meta.premises, W_prop_hat.shape[0])
                steps = min(cfg.K, W_prop_hat.shape[0])
                u = _propagate_normalize(W_prop_hat, u0, steps)

                denom = float(np.sum(np.abs(u) ** 4))
                pr = 0.0 if denom == 0.0 else float(1.0 / denom)
                conclusion_id = int(meta.candidates[int(meta.true_index)])
                mass = float(u[conclusion_id] ** 2) if conclusion_id < len(u) else 0.0

                pr_by_topo[meta.topology].append(pr)
                mass_by_topo[meta.topology].append(mass)

                per_pair.setdefault(meta.pair_id, {})[meta.topology] = {
                    "sel": sel,
                    "pr": pr,
                    "mass": mass,
                }

    pairs: List[dict] = []
    for pair_id, vals in per_pair.items():
        if "OBC" not in vals or "PBC" not in vals:
            continue
        sel_obc = vals["OBC"]["sel"]
        sel_pbc = vals["PBC"]["sel"]
        pr_obc = vals["OBC"]["pr"]
        pr_pbc = vals["PBC"]["pr"]
        mass_obc = vals["OBC"]["mass"]
        mass_pbc = vals["PBC"]["mass"]
        pairs.append(
            {
                "pair_id": pair_id,
                "sel_loc_gap_obc": float(sel_obc),
                "sel_loc_gap_pbc": float(sel_pbc),
                "sel_loc_gap_diff": float(sel_obc - sel_pbc),
                "pr_diff": float(pr_obc - pr_pbc),
                "mass_diff": float(mass_obc - mass_pbc),
            }
        )

    paired_dir.mkdir(parents=True, exist_ok=True)
    pair_path = paired_dir / f"{variant}_seed{ckpt['seed']}.json"
    pair_path.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n")

    sel_obc = sel_by_topo["OBC"]
    sel_pbc = sel_by_topo["PBC"]
    sel_diff = [p["sel_loc_gap_diff"] for p in pairs]
    pr_diff = [p["pr_diff"] for p in pairs]
    mass_diff = [p["mass_diff"] for p in pairs]

    sel_obc_ci = _bootstrap_ci(sel_obc, n_boot=bootstrap)
    sel_pbc_ci = _bootstrap_ci(sel_pbc, n_boot=bootstrap)
    sel_diff_ci = _bootstrap_ci(sel_diff, n_boot=bootstrap)
    pr_ci = _bootstrap_ci(pr_diff, n_boot=bootstrap)
    mass_ci = _bootstrap_ci(mass_diff, n_boot=bootstrap)

    summary = {
        "variant": variant,
        "seed": int(ckpt["seed"]),
        "sel_loc_gap": {
            "obc_mean": float(np.mean(sel_obc)) if sel_obc else 0.0,
            "obc_ci_low": float(sel_obc_ci[0]),
            "obc_ci_high": float(sel_obc_ci[1]),
            "pbc_mean": float(np.mean(sel_pbc)) if sel_pbc else 0.0,
            "pbc_ci_low": float(sel_pbc_ci[0]),
            "pbc_ci_high": float(sel_pbc_ci[1]),
            "delta_mean": float(np.mean(sel_diff)) if sel_diff else 0.0,
            "delta_ci_low": float(sel_diff_ci[0]),
            "delta_ci_high": float(sel_diff_ci[1]),
        },
        "topology": {
            "pr_diff_mean": float(np.mean(pr_diff)) if pr_diff else 0.0,
            "pr_diff_ci_low": float(pr_ci[0]),
            "pr_diff_ci_high": float(pr_ci[1]),
            "mass_diff_mean": float(np.mean(mass_diff)) if mass_diff else 0.0,
            "mass_diff_ci_low": float(mass_ci[0]),
            "mass_diff_ci_high": float(mass_ci[1]),
            "n_pairs": int(len(pairs)),
        },
        "gamma_stats": gamma_stats,
        "pair_path": str(pair_path),
    }

    summary_path = paired_dir / f"{variant}_seed{ckpt['seed']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _load_instances(path: Path) -> Dict[str, Tuple[int, int]]:
    records: Dict[str, Tuple[int, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records[row["instance_key"]] = (int(row["pred"]), int(row["label"]))
    return records


def _delta_from_instances(a_path: Path, b_path: Path) -> List[int]:
    a = _load_instances(a_path)
    b = _load_instances(b_path)
    keys = sorted(set(a.keys()) & set(b.keys()))
    diffs = []
    for key in keys:
        a_pred, a_label = a[key]
        b_pred, b_label = b[key]
        diffs.append(int(a_pred == a_label) - int(b_pred == b_label))
    return diffs


def _load_pairs(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["pair_id"]: row for row in data}


def _delta_sel_loc_between_variants(a_path: Path, b_path: Path) -> List[float]:
    a = _load_pairs(a_path)
    b = _load_pairs(b_path)
    keys = sorted(set(a.keys()) & set(b.keys()))
    diffs = []
    for key in keys:
        diffs.append(float(a[key]["sel_loc_gap_diff"] - b[key]["sel_loc_gap_diff"]))
    return diffs


def _collect_master_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in root.glob("paired/*_summary.json"):
        summary = json.loads(summary_path.read_text())
        variant = summary["variant"]
        seed = int(summary["seed"])
        train_dir = root / "train_conclusion" / variant / f"seed_{seed}"
        eval_conc = json.loads((train_dir / "eval_conclusion_summary.json").read_text())
        eval_topo = json.loads((train_dir / "eval_topology_summary.json").read_text())
        adapt = json.loads((root / "adapt" / variant / f"seed_{seed}" / "n512" / "head" / "adapt_summary.json").read_text())

        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "train_task": "conclusion",
                "in_task_acc": eval_conc["acc"],
                "in_task_ci_low": eval_conc["ci_low"],
                "in_task_ci_high": eval_conc["ci_high"],
                "zero_shot_acc": eval_topo["acc"],
                "zero_shot_ci_low": eval_topo["ci_low"],
                "zero_shot_ci_high": eval_topo["ci_high"],
                "fewshot_n": adapt["n_train"],
                "fewshot_acc": adapt["final_acc"],
                "fewshot_ci_low": adapt["final_ci_low"],
                "fewshot_ci_high": adapt["final_ci_high"],
                "fewshot_correct": adapt["final_correct"],
                "fewshot_total": adapt["final_total"],
                "sel_loc_gap_obc": summary["sel_loc_gap"]["obc_mean"],
                "sel_loc_gap_pbc": summary["sel_loc_gap"]["pbc_mean"],
                "delta_sel_loc": summary["sel_loc_gap"]["delta_mean"],
                "pr_diff": summary["topology"]["pr_diff_mean"],
                "pr_diff_ci_low": summary["topology"]["pr_diff_ci_low"],
                "pr_diff_ci_high": summary["topology"]["pr_diff_ci_high"],
                "mass_diff": summary["topology"]["mass_diff_mean"],
                "mass_diff_ci_low": summary["topology"]["mass_diff_ci_low"],
                "mass_diff_ci_high": summary["topology"]["mass_diff_ci_high"],
                "gamma": summary["gamma_stats"]["gamma"],
                "zero_rate_B0": summary["gamma_stats"]["zero_rate_B0"],
                "rho_median": summary["gamma_stats"]["rho_median"],
                "scale_flag": summary["gamma_stats"]["flag"],
            }
        )
    return rows


def _write_master_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "train_task",
                "in_task_acc",
                "in_task_ci_low",
                "in_task_ci_high",
                "zero_shot_acc",
                "zero_shot_ci_low",
                "zero_shot_ci_high",
                "fewshot_n",
                "fewshot_acc",
                "fewshot_ci_low",
                "fewshot_ci_high",
                "fewshot_correct",
                "fewshot_total",
                "sel_loc_gap_obc",
                "sel_loc_gap_pbc",
                "delta_sel_loc",
                "pr_diff",
                "pr_diff_ci_low",
                "pr_diff_ci_high",
                "mass_diff",
                "mass_diff_ci_low",
                "mass_diff_ci_high",
                "gamma",
                "zero_rate_B0",
                "rho_median",
                "scale_flag",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_report(root: Path, out_md: Path, out_json: Path, bootstrap: int) -> None:
    rows = _collect_master_rows(root)

    variants = sorted({r["variant"] for r in rows})

    def _subset(variant: str) -> List[Dict[str, object]]:
        return [r for r in rows if r["variant"] == variant]

    report_lines = ["# V3 Confirmatory Report", ""]

    # Primary endpoint E1
    diffs_e1: List[float] = []
    for seed in sorted({r["seed"] for r in rows}):
        a_path = root / "paired" / f"mechanism_seed{seed}.json"
        b_path = root / "paired" / f"symmetric_control_v2_normmatched_seed{seed}.json"
        if a_path.exists() and b_path.exists():
            diffs_e1.extend(_delta_sel_loc_between_variants(a_path, b_path))
    e1_mean = float(np.mean(diffs_e1)) if diffs_e1 else 0.0
    e1_ci = _bootstrap_ci(diffs_e1, n_boot=bootstrap)

    report_lines.append("## Primary endpoints")
    report_lines.append(
        f"- E1 ΔSelLoc(mech)-ΔSelLoc(sym): mean={e1_mean:.6f} CI=[{e1_ci[0]:.6f},{e1_ci[1]:.6f}] n_pairs={len(diffs_e1)}"
    )

    # Primary endpoint E2
    diffs_e2: List[int] = []
    for seed in sorted({r["seed"] for r in rows}):
        mech_path = root / "adapt" / "mechanism" / f"seed_{seed}" / "n512" / "head" / "eval_instances.jsonl.gz"
        base_path = root / "adapt" / "no_injection" / f"seed_{seed}" / "n512" / "head" / "eval_instances.jsonl.gz"
        if mech_path.exists() and base_path.exists():
            diffs_e2.extend(_delta_from_instances(mech_path, base_path))
    e2_mean = float(np.mean(diffs_e2)) if diffs_e2 else 0.0
    e2_ci = _bootstrap_ci(diffs_e2, n_boot=bootstrap)

    report_lines.append(
        f"- E2 acc_512(mech)-acc_512(base): mean={e2_mean:.6f} CI=[{e2_ci[0]:.6f},{e2_ci[1]:.6f}] n_inst={len(diffs_e2)}"
    )
    report_lines.append("")

    report_lines.append("## SelLocGap by variant")
    for variant in variants:
        vals = [r["delta_sel_loc"] for r in _subset(variant)]
        ci = _bootstrap_ci(vals, n_boot=bootstrap)
        report_lines.append(
            f"- {variant}: ΔSelLoc mean={float(np.mean(vals)):.6f} CI=[{ci[0]:.6f},{ci[1]:.6f}] n={len(vals)}"
        )
    report_lines.append("")

    report_lines.append("## Few-shot transfer (n=512)")
    for variant in variants:
        subset = _subset(variant)
        correct = sum(int(r["fewshot_correct"]) for r in subset)
        total = sum(int(r["fewshot_total"]) for r in subset)
        acc = correct / total if total else 0.0
        ci = wilson_ci(correct, total)
        report_lines.append(f"- {variant}: acc={acc:.6f} CI=[{ci[0]:.6f},{ci[1]:.6f}]")
    report_lines.append("")

    report_lines.append("## Topology sensitivity (PR and mass)")
    for variant in variants:
        subset = _subset(variant)
        pr_vals = [r["pr_diff"] for r in subset]
        mass_vals = [r["mass_diff"] for r in subset]
        pr_ci = _bootstrap_ci(pr_vals, n_boot=bootstrap)
        mass_ci = _bootstrap_ci(mass_vals, n_boot=bootstrap)
        report_lines.append(
            f"- {variant}: PR diff mean={float(np.mean(pr_vals)):.6f} CI=[{pr_ci[0]:.6f},{pr_ci[1]:.6f}]"
        )
        report_lines.append(
            f"- {variant}: mass diff mean={float(np.mean(mass_vals)):.6f} CI=[{mass_ci[0]:.6f},{mass_ci[1]:.6f}]"
        )
    report_lines.append("")

    report_lines.append("## Scale-match / gamma stats")
    for variant in variants:
        subset = _subset(variant)
        gamma_vals = [r["gamma"] for r in subset]
        zero_rates = [r["zero_rate_B0"] for r in subset]
        rho_vals = [r["rho_median"] for r in subset]
        flags = [r["scale_flag"] for r in subset]
        report_lines.append(
            f"- {variant}: gamma_median={float(np.median(gamma_vals)):.6f} zero_rate_B0_median={float(np.median(zero_rates)):.6f} rho_median={float(np.median(rho_vals)):.6f} flag_any={any(flags)}"
        )
    report_lines.append("")

    e1_pass = bool(e1_ci[0] > 0.0)
    e2_pass = bool(e2_mean <= -0.02 and e2_ci[1] < 0.0)
    overall_pass = bool(e1_pass and e2_pass)

    report_lines.append("## Pass/fail")
    report_lines.append(f"- E1_pass={e1_pass}")
    report_lines.append(f"- E2_pass={e2_pass}")
    report_lines.append(f"- overall_pass={overall_pass}")

    out_md.write_text("\n".join(report_lines) + "\n")

    out_json.write_text(
        json.dumps(
            {
                "E1": {"mean": e1_mean, "ci_low": e1_ci[0], "ci_high": e1_ci[1]},
                "E2": {"mean": e2_mean, "ci_low": e2_ci[0], "ci_high": e2_ci[1]},
                "gates": {"E1_pass": e1_pass, "E2_pass": e2_pass, "overall_pass": overall_pass},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--train_size", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--k_prop", type=int, default=4)
    ap.add_argument("--adapt_n", type=int, default=512)
    ap.add_argument("--adapt_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--resume_if_exists", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = _resolve_device(args.device)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = V3Config()

    variants = {
        "no_injection": {"alpha": args.alpha, "beta": 0.0, "gamma": 1.0},
        "mechanism": {"alpha": args.alpha, "beta": args.beta, "gamma": 1.0},
        "symmetric_control_v2_normmatched": {"alpha": args.alpha, "beta": args.beta, "gamma": 1.0},
    }

    paired_dir = out_root / "paired"

    for seed in seeds:
        for variant, params in variants.items():
            run_dir = out_root / "train_conclusion" / variant / f"seed_{seed}"
            final_ckpt = run_dir / "checkpoint_final.pt"
            if not final_ckpt.exists():
                _train_model(
                    task="conclusion",
                    variant=variant,
                    seed=seed,
                    cfg=cfg,
                    train_size=args.train_size,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    alpha=params["alpha"],
                    beta=params["beta"],
                    gamma=params["gamma"],
                    device=device,
                    run_root=out_root,
                    save_every=args.save_every,
                    resume_if_exists=args.resume_if_exists,
                )

            if not (run_dir / "eval_conclusion_summary.json").exists():
                _eval_model(final_ckpt, "conclusion", args.eval_size, args.batch_size, device)
            if not (run_dir / "eval_topology_summary.json").exists():
                _eval_model(final_ckpt, "topology", args.eval_size, args.batch_size, device)

            pair_summary_path = paired_dir / f"{variant}_seed{seed}_summary.json"
            if not pair_summary_path.exists():
                _sel_locgap_and_topology(
                    ckpt_path=final_ckpt,
                    eval_size=args.eval_size,
                    batch_size=args.batch_size,
                    k_tok=args.k_tok,
                    k_prop=args.k_prop,
                    device=device,
                    paired_dir=paired_dir,
                    bootstrap=args.bootstrap,
                )

            adapt_summary_path = out_root / "adapt" / variant / f"seed_{seed}" / "n512" / "head" / "adapt_summary.json"
            if not adapt_summary_path.exists():
                _fewshot_adapt(
                    run_root=out_root,
                    variant=variant,
                    seed=seed,
                    n_train=args.adapt_n,
                    steps=args.adapt_steps,
                    eval_every=args.eval_every,
                    eval_size=args.eval_size,
                    batch_size=args.batch_size,
                    device=device,
                )

    master_csv = out_root / "v3_confirmatory_master.csv"
    _write_master_csv(_collect_master_rows(out_root), master_csv)

    report_md = out_root / "v3_confirmatory_report.md"
    report_json = out_root / "v3_confirmatory_report.json"
    _aggregate_report(out_root, report_md, report_json, args.bootstrap)

    print(f"Wrote {master_csv}")
    print(f"Wrote {report_md}")
    print(f"Wrote {report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
