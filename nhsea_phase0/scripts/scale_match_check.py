#!/usr/bin/env python
"""Scale-match preflight check for symmetric control."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import ForwardChainConfig
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import rho_ratio, scale_norms


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


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


def _load_checkpoint(run_dir: Path) -> dict:
    for name in ("checkpoint_final.pt", "checkpoint.pt"):
        path = run_dir / name
        if path.exists():
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No checkpoint_final.pt or checkpoint.pt in {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="forward")
    ap.add_argument("--variant", type=str, default="symmetric_control")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--eval_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--config", type=str, default="phase2_lock.json")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mask", type=str, choices=["pad", "all"], default="pad")
    args = ap.parse_args()

    if args.task != "forward":
        raise ValueError("scale_match_check only supports task=forward")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    results = []

    if args.run_dir:
        run_dir = Path(args.run_dir)
        ckpt = _load_checkpoint(run_dir)
        if ckpt.get("task") != "forward":
            raise ValueError("run_dir checkpoint is not a forward task")
        seed = int(ckpt["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        eval_cfg_path = run_dir / "eval_config.json"
        eval_cfg = json.loads(eval_cfg_path.read_text()) if eval_cfg_path.exists() else {}
        gen_cfg_dict = eval_cfg.get("gen_cfg", {})
        gen_cfg = ForwardChainConfig(**gen_cfg_dict) if gen_cfg_dict else ForwardChainConfig()
        eval_size = int(eval_cfg.get("eval_size", args.eval_size))

        data_cfg = DatasetConfig(
            task="forward",
            split="eval",
            size=eval_size,
            seed=seed,
            T=gen_cfg.T,
            vocab_size=gen_cfg.vocab_size,
        )
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        model_cfg = ModelConfig(**ckpt["model_cfg"])
        model = TinyTransformer(model_cfg, probe_layer=2)
        model.set_num_classes(2)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        device = _resolve_device(args.device)
        model.to(device)

        alpha = float(ckpt["alpha"])
        beta = float(ckpt["beta"])
        variant = str(ckpt["variant"])
        beta_eff = 0.0 if variant == "no_injection" else beta

        A_vals: List[float] = []
        B0_vals: List[float] = []

        with torch.no_grad():
            for input_ids, attn_mask, _labels, _metas in loader:
                if args.mask == "all":
                    attn_mask = torch.ones_like(attn_mask)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                _logits, _probe_logits, probe_U = model(
                    input_ids,
                    attn_mask=attn_mask,
                    variant=variant,
                    alpha=alpha,
                    beta=beta,
                    return_probe=True,
                )
                for idx in range(probe_U.shape[0]):
                    U = probe_U[idx].cpu().numpy()
                    A, B0 = scale_norms(U, beta=beta_eff)
                    A_vals.append(float(A))
                    B0_vals.append(float(B0))

        A_median = float(np.median(np.asarray(A_vals))) if A_vals else 0.0
        B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
        gamma = 1.0 if B0_median == 0.0 else A_median / B0_median
        zero_rate_B0 = 0.0 if not B0_vals else float(np.mean(np.asarray(B0_vals) == 0.0))
        if variant == "symmetric_control_v2_normmatched":
            B_vals = [gamma * b0 for b0 in B0_vals]
            rho_vals = [rho_ratio(a, gamma * b0) for a, b0 in zip(A_vals, B0_vals)]
            flag = not (B0_median > 0.0 and zero_rate_B0 <= 0.001)
        else:
            B_vals = list(B0_vals)
            rho_vals = [rho_ratio(a, b0) for a, b0 in zip(A_vals, B0_vals)]
            flag = not (0.9 <= float(np.median(np.asarray(rho_vals))) <= 1.1)

        rho_median = float(np.median(np.asarray(rho_vals))) if rho_vals else 0.0
        B_median = float(np.median(np.asarray(B_vals))) if B_vals else 0.0

        record = {
            "seed": seed,
            "rho_median": rho_median,
            "A_median": A_median,
            "B_median": B_median,
            "B0_median": B0_median,
            "gamma": float(gamma),
            "zero_rate_B0": zero_rate_B0,
            "flag": flag,
            "run_dir": str(run_dir),
            "device": str(device),
            "mask": args.mask,
        }
        results.append(record)
        print(
            f"seed {seed}: rho_median={rho_median:.6f} "
            f"A_median={A_median:.6f} B_median={B_median:.6f} "
            f"gamma={gamma:.6f} zero_rate_B0={zero_rate_B0:.6f} flag={flag}"
        )
    else:
        alpha, beta = cfg["alpha_beta"][args.variant]
        gen_cfg = ForwardChainConfig(**cfg["forward_gen"])

        seeds = _parse_seeds(args.seeds)
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            data_cfg = DatasetConfig(
                task="forward",
                split="eval",
                size=args.eval_size,
                seed=seed,
                T=gen_cfg.T,
                vocab_size=gen_cfg.vocab_size,
            )
            dataset = ForwardChainDataset(data_cfg, gen_cfg)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

            model_cfg = ModelConfig(vocab_size=len(dataset.vocab), T=gen_cfg.T)
            model = TinyTransformer(model_cfg, probe_layer=2)
            model.set_num_classes(2)
            model.eval()
            device = _resolve_device(args.device)
            model.to(device)

            A_vals: List[float] = []
            B0_vals: List[float] = []

            with torch.no_grad():
                for input_ids, attn_mask, _labels, _metas in loader:
                    if args.mask == "all":
                        attn_mask = torch.ones_like(attn_mask)
                    input_ids = input_ids.to(device)
                    attn_mask = attn_mask.to(device)
                    _logits, _probe_logits, probe_U = model(
                        input_ids,
                        attn_mask=attn_mask,
                        variant=args.variant,
                        alpha=alpha,
                        beta=beta,
                        return_probe=True,
                    )
                    for idx in range(probe_U.shape[0]):
                        U = probe_U[idx].cpu().numpy()
                        A, B0 = scale_norms(U, beta=beta)
                        A_vals.append(float(A))
                        B0_vals.append(float(B0))

            A_median = float(np.median(np.asarray(A_vals))) if A_vals else 0.0
            B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
            gamma = 1.0 if B0_median == 0.0 else A_median / B0_median
            zero_rate_B0 = 0.0 if not B0_vals else float(np.mean(np.asarray(B0_vals) == 0.0))
            if args.variant == "symmetric_control_v2_normmatched":
                B_vals = [gamma * b0 for b0 in B0_vals]
                rho_vals = [rho_ratio(a, gamma * b0) for a, b0 in zip(A_vals, B0_vals)]
                flag = not (B0_median > 0.0 and zero_rate_B0 <= 0.001)
            else:
                B_vals = list(B0_vals)
                rho_vals = [rho_ratio(a, b0) for a, b0 in zip(A_vals, B0_vals)]
                flag = not (0.9 <= float(np.median(np.asarray(rho_vals))) <= 1.1)

            rho_median = float(np.median(np.asarray(rho_vals))) if rho_vals else 0.0
            B_median = float(np.median(np.asarray(B_vals))) if B_vals else 0.0

            record = {
                "seed": seed,
                "rho_median": rho_median,
                "A_median": A_median,
                "B_median": B_median,
                "B0_median": B0_median,
                "gamma": float(gamma),
                "zero_rate_B0": zero_rate_B0,
                "flag": flag,
            }
            results.append(record)
            print(
                f"seed {seed}: rho_median={rho_median:.6f} "
                f"A_median={A_median:.6f} B_median={B_median:.6f} "
                f"gamma={gamma:.6f} zero_rate_B0={zero_rate_B0:.6f} flag={flag}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "variant": args.variant,
            "eval_size": args.eval_size,
            "batch_size": args.batch_size,
            "seeds": results,
            "passed": all(not r["flag"] for r in results),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
