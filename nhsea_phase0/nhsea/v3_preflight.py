"""NHSEA v3 baseline-only preflight runner."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .data_v3 import V3DatasetConfig, V3Dataset, collate_batch
from .generators_v3 import V3Config
from .leak_gate_v3 import run_leak_gate
from .model import ModelConfig, TinyTransformer
from .operator import OperatorSpec, build_run_operator, token_weights
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


def _epoch_permutation(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch)
    return rng.permutation(size)


def _seed_vector(indices: List[int], size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float64)
    if not indices:
        return v
    scale = 1.0 / np.sqrt(len(indices))
    v[np.asarray(indices, dtype=np.int64)] = scale
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


def _topology_metrics(
    model: TinyTransformer,
    dataset: V3Dataset,
    device: torch.device,
    k_tok: int,
    k_prop: int,
    cfg: V3Config,
) -> List[Dict[str, float]]:
    model.eval()
    pairs: Dict[str, Dict[str, Tuple[float, float]]] = {}
    spec = OperatorSpec(alpha=0.0, beta=0.0, variant="no_injection")

    with torch.no_grad():
        for i in range(0, len(dataset), 32):
            batch = [dataset[j] for j in range(i, min(i + 32, len(dataset)))]
            input_ids, attn_mask, _labels, metas = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _logits, probe_logits, _probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant="no_injection",
                alpha=0.0,
                beta=0.0,
                gamma=1.0,
                return_probe=True,
            )

            probe_logits_np = probe_logits.cpu().numpy()
            for idx, meta in enumerate(metas):
                L = probe_logits_np[idx].mean(axis=0)
                O = build_run_operator(L, None, spec)
                W_tok = token_weights(O)
                W_tok_hat = topk_rownorm(W_tok, k_tok)
                W_prop = _collapse_to_prop(W_tok_hat, meta.spans)
                W_prop_hat = rownorm_plus(diagzero(topk_rownorm(W_prop, k_prop)))

                prem = list(meta.premises)
                u0 = _seed_vector(prem, W_prop_hat.shape[0])
                s = min(cfg.K, W_prop_hat.shape[0])
                u = _propagate_normalize(W_prop_hat, u0, s)
                pr = 0.0
                denom = float(np.sum(np.abs(u) ** 4))
                if denom > 0.0:
                    pr = 1.0 / denom
                conclusion_id = int(meta.candidates[int(meta.true_index)])
                mass = float(u[conclusion_id] ** 2) if conclusion_id < len(u) else 0.0

                pairs.setdefault(meta.pair_id, {})[meta.topology] = (pr, mass)

    diffs: List[Dict[str, float]] = []
    for pair_id, vals in pairs.items():
        if "OBC" not in vals or "PBC" not in vals:
            continue
        pr_obc, mass_obc = vals["OBC"]
        pr_pbc, mass_pbc = vals["PBC"]
        diffs.append(
            {
                "pair_id": pair_id,
                "pr_diff": float(pr_obc - pr_pbc),
                "mass_diff": float(mass_obc - mass_pbc),
            }
        )
    return diffs


def _train_model(
    task: str,
    seed: int,
    cfg: V3Config,
    train_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    run_root: Path,
    variant: str,
) -> Path:
    if variant != "baseline_only":
        raise ValueError("Only baseline_only variant is allowed in v3 preflight")

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

    out_dir = run_root / f"{task}_train" / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task": task,
        "variant": variant,
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
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    perm = _epoch_permutation(dataset_size, seed, 0)
    epoch = 0
    batch_idx = 0

    model.train()
    for step in range(1, steps + 1):
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
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant="no_injection",
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
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
        "task": task,
        "variant": variant,
        "seed": seed,
        "model_cfg": asdict(model_cfg),
        "model_state": model.state_dict(),
        "vocab": dataset.vocab,
        "gen_cfg": asdict(cfg),
        "train_size": train_size,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "git_commit": _git_commit(),
    }
    ckpt_path = out_dir / "checkpoint_final.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def _eval_model(
    ckpt_path: Path,
    eval_task: str,
    eval_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = V3Config(**ckpt["gen_cfg"])

    data_cfg = V3DatasetConfig(
        task=eval_task,
        split="eval",
        size=eval_size,
        seed=int(ckpt.get("seed", 0)),
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
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, attn_mask, labels, _ = collate_batch(batch)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits, _, _ = model(
                input_ids,
                attn_mask=attn_mask,
                variant="no_injection",
                alpha=0.0,
                beta=0.0,
                gamma=1.0,
            )
            preds = torch.argmax(logits, dim=-1).cpu()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    acc = correct / total if total else 0.0
    ci_low, ci_high = wilson_ci(correct, total)

    train_task = str(ckpt.get("task", ""))
    mode = "in_task" if train_task == eval_task else "zero_shot"

    summary = {
        "train_task": train_task,
        "eval_task": eval_task,
        "mode": mode,
        "variant": ckpt.get("variant", "baseline_only"),
        "seed": int(ckpt.get("seed", 0)),
        "eval_size": eval_size,
        "batch_size": batch_size,
        "acc": float(acc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "correct": int(correct),
        "total": int(total),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "model_cfg": asdict(model_cfg),
        "gen_cfg": asdict(cfg),
        "checkpoint": str(ckpt_path),
    }
    out_path = ckpt_path.parent / f"eval_{eval_task}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _fewshot_adapt(
    run_root: Path,
    source_task: str,
    target_task: str,
    seed: int,
    n_train: int,
    steps: int,
    eval_every: int,
    eval_size: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    ckpt_path = run_root / f"{source_task}_train" / "baseline_only" / f"seed_{seed}" / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

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

    train_cfg = V3DatasetConfig(
        task=target_task,
        split="adapt",
        size=n_train,
        seed=seed,
        T=cfg.T,
        M=cfg.M,
        K=cfg.K,
        L_min=cfg.L_min,
        L_max=cfg.L_max,
        vocab_size=cfg.vocab_size,
    )
    eval_cfg = V3DatasetConfig(
        task=target_task,
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

    opt = torch.optim.AdamW(model.head.parameters(), lr=ckpt.get("lr", 3e-4), weight_decay=ckpt.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss()

    out_dir = (
        run_root
        / "baseline_only"
        / f"seed{seed}"
        / f"{source_task}_to_{target_task}"
        / "head"
        / f"n{n_train}"
    )
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
                    variant="no_injection",
                    alpha=0.0,
                    beta=0.0,
                    gamma=1.0,
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

    for step in range(1, steps + 1):
        input_ids, attn_mask, labels = _next_batch()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits, _, _ = model(
            input_ids,
            attn_mask=attn_mask,
            variant="no_injection",
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
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
            record = {
                "step": step,
                "loss": float(loss.item()),
            }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    log_f.close()

    final_ci_low, final_ci_high = wilson_ci(final_correct, final_total)

    summary = {
        "variant": "baseline_only",
        "seed": seed,
        "source_task": source_task,
        "target_task": target_task,
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
    }

    summary_path = out_dir / "adapt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--variant", type=str, default="baseline_only")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--train_size", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--k_prop", type=int, default=4)
    ap.add_argument("--n_train", type=str, default="32,128,512")
    ap.add_argument("--adapt_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.variant != "baseline_only":
        raise ValueError("Only baseline_only variant is allowed in v3 preflight")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    n_trains = [int(s.strip()) for s in args.n_train.split(",") if s.strip()]

    if args.smoke:
        seeds = [0]
        n_trains = [8]
        args.train_size = 16
        args.steps = 5
        args.eval_size = 32
        args.batch_size = 8
        args.adapt_steps = 10
        args.eval_every = 5
        args.device = "cpu"

    device = _resolve_device(args.device)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = V3Config()

    # Leak gate (forward=conclusion, backward=topology).
    leak_dir = out_root / "leak"
    leak_dir.mkdir(parents=True, exist_ok=True)
    for label, task in (("forward", "conclusion"), ("backward", "topology")):
        report, features = run_leak_gate(task, n=4000 if not args.smoke else 128, seed=0, cfg=cfg, auroc_max=0.55)
        report_path = leak_dir / f"leak_gate_v3_{label}.json"
        feat_path = leak_dir / f"leak_gate_v3_{label}_features.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        feat_path.write_text(json.dumps({"features": features, "task": task}, indent=2, sort_keys=True) + "\n")

    # Train/eval tasks.
    for seed in seeds:
        ckpt_conc = _train_model(
            task="conclusion",
            seed=seed,
            cfg=cfg,
            train_size=args.train_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            run_root=out_root,
            variant=args.variant,
        )
        _eval_model(ckpt_conc, "conclusion", args.eval_size, args.batch_size, device)
        _eval_model(ckpt_conc, "topology", args.eval_size, args.batch_size, device)

        ckpt_topo = _train_model(
            task="topology",
            seed=seed,
            cfg=cfg,
            train_size=args.train_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            run_root=out_root,
            variant=args.variant,
        )
        _eval_model(ckpt_topo, "topology", args.eval_size, args.batch_size, device)
        _eval_model(ckpt_topo, "conclusion", args.eval_size, args.batch_size, device)

        # Few-shot adaptation in both directions.
        for n_train in n_trains:
            _fewshot_adapt(
                run_root=out_root,
                source_task="conclusion",
                target_task="topology",
                seed=seed,
                n_train=n_train,
                steps=args.adapt_steps,
                eval_every=args.eval_every,
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                device=device,
            )
            _fewshot_adapt(
                run_root=out_root,
                source_task="topology",
                target_task="conclusion",
                seed=seed,
                n_train=n_train,
                steps=args.adapt_steps,
                eval_every=args.eval_every,
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                device=device,
            )

        # Topology sensitivity (use conclusion-trained model).
        data_cfg = V3DatasetConfig(
            task="conclusion",
            split="eval",
            size=args.eval_size,
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
        ckpt = torch.load(ckpt_conc, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        diffs = _topology_metrics(
            model=model,
            dataset=dataset,
            device=device,
            k_tok=args.k_tok,
            k_prop=args.k_prop,
            cfg=cfg,
        )
        pair_path = out_root / f"topology_pairs_seed{seed}.json"
        pair_path.write_text(json.dumps(diffs, indent=2, sort_keys=True) + "\n")

    # Aggregate results.
    from .v3_aggregate import main as aggregate_main

    aggregate_main(["--root", str(out_root), "--out", str(out_root / "v3_preflight_master.csv"), "--report", str(out_root / "v3_preflight_report.md"), "--json", str(out_root / "v3_preflight_report.json")])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
