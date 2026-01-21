#!/usr/bin/env python
"""Reproduce scale-mismatch numbers for a completed forward run."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import ForwardChainConfig
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


def _load_checkpoint(run_dir: Path) -> Dict:
    for name in ("checkpoint_final.pt", "checkpoint.pt"):
        path = run_dir / name
        if path.exists():
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No checkpoint_final.pt or checkpoint.pt in {run_dir}")


def _load_eval_config(run_dir: Path) -> Dict:
    path = run_dir / "eval_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_eval_summary(run_dir: Path) -> Dict:
    path = run_dir / "eval_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dump", type=str, default="")
    ap.add_argument("--mask", type=str, choices=["pad", "all"], default="pad")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = _load_checkpoint(run_dir)
    eval_cfg = _load_eval_config(run_dir)
    eval_summary = _load_eval_summary(run_dir)

    if ckpt.get("task") != "forward":
        raise ValueError("repro_scale_mismatch only supports forward task runs")

    torch.manual_seed(int(ckpt["seed"]))
    np.random.seed(int(ckpt["seed"]))
    device = _resolve_device(args.device)

    eval_size = int(eval_cfg.get("eval_size", eval_summary.get("eval_size", 10000)))
    if args.n > 0:
        eval_size = max(eval_size, args.n)

    gen_cfg_dict = eval_cfg.get("gen_cfg", {})
    gen_cfg = ForwardChainConfig(**gen_cfg_dict) if gen_cfg_dict else ForwardChainConfig()

    data_cfg = DatasetConfig(task="forward", split="eval", size=eval_size, seed=int(ckpt["seed"]))
    dataset = ForwardChainDataset(data_cfg, gen_cfg)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    beta = float(ckpt["beta"])
    variant = str(ckpt["variant"])
    beta_eff = 0.0 if variant == "no_injection" else beta

    rho_vals: List[float] = []
    A_vals: List[float] = []
    B_vals: List[float] = []
    per_instance: List[Dict[str, float]] = []

    count = 0
    with torch.no_grad():
        for input_ids, attn_mask, _labels, metas in loader:
            if args.mask == "all":
                attn_mask = torch.ones_like(attn_mask)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _logits, _probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=float(ckpt["alpha"]),
                beta=beta,
                return_probe=True,
            )
            U_np = probe_U.cpu().numpy()
            for idx, meta in enumerate(metas):
                if args.n > 0 and count >= args.n:
                    break
                U = U_np[idx]
                A, B = scale_norms(U, beta_eff)
                rho = rho_ratio(A, B)
                A_vals.append(float(A))
                B_vals.append(float(B))
                rho_vals.append(float(rho))
                per_instance.append(
                    {
                        "instance_id": meta.instance_id,
                        "A": float(A),
                        "B": float(B),
                        "rho": float(rho),
                    }
                )
                count += 1
            if args.n > 0 and count >= args.n:
                break

    rho_median = float(np.median(np.asarray(rho_vals)))
    A_median = float(np.median(np.asarray(A_vals)))
    B_median = float(np.median(np.asarray(B_vals)))
    flag = not (0.9 <= rho_median <= 1.1)

    repro = {
        "run_dir": str(run_dir),
        "device": str(device),
        "mask": args.mask,
        "n": args.n,
        "rho_median_repro": rho_median,
        "A_median_repro": A_median,
        "B_median_repro": B_median,
        "flag_repro": flag,
        "rho_median_eval": eval_summary.get("rho_median"),
        "A_median_eval": eval_summary.get("A_median"),
        "B_median_eval": eval_summary.get("B_median"),
        "flag_eval": eval_summary.get("rho_flag"),
        "diff_rho": None if eval_summary.get("rho_median") is None else float(rho_median - eval_summary["rho_median"]),
        "diff_A": None if eval_summary.get("A_median") is None else float(A_median - eval_summary["A_median"]),
        "diff_B": None if eval_summary.get("B_median") is None else float(B_median - eval_summary["B_median"]),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    if args.dump:
        out_path = Path(args.dump)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": repro,
            "instances": per_instance,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(json.dumps(repro, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
