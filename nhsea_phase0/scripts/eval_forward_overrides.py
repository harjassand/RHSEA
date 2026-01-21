#!/usr/bin/env python
"""Evaluate forward SelLocGap with eval-only operator overrides."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import ForwardChainConfig, candidate_token_spans, premise_token_set
from nhsea.metrics import token_adj_loc_gap
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import OperatorSpec, rho_ratio, scale_norms


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    return {"low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


def _summary(values: List[float], seed: int = 0, n_boot: int = 10000) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    ci = _bootstrap_ci(arr, seed=seed, n_boot=n_boot)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": ci["low"],
        "ci_high": ci["high"],
    }


def _compute_gamma_stats(
    loader: DataLoader,
    model: TinyTransformer,
    variant: str,
    alpha: float,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    beta_eff = 0.0 if variant == "no_injection" else beta
    A_vals: List[float] = []
    B0_vals: List[float] = []
    zero_count = 0
    total = 0
    with torch.no_grad():
        for input_ids, attn_mask, _labels, _metas in loader:
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
    return {
        "A_median": A_median,
        "B0_median": B0_median,
        "gamma": float(gamma),
        "zero_rate_B0": float(zero_rate),
    }


def _sign_test_pvalue(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 1.0
    k = sum(1 for v in values if v > 0)
    mean = n * 0.5
    var = n * 0.25
    if var == 0:
        return 1.0
    z = (k - mean) / math.sqrt(var)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _load_model(ckpt_path: str, num_classes: int) -> TinyTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech_ckpt", type=str, required=True)
    ap.add_argument("--sym_ckpt", type=str, required=True)
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    mech_ckpt = torch.load(args.mech_ckpt, map_location="cpu")
    sym_ckpt = torch.load(args.sym_ckpt, map_location="cpu")
    if mech_ckpt["task"] != "forward" or sym_ckpt["task"] != "forward":
        raise ValueError("Both checkpoints must be forward task")
    if mech_ckpt["seed"] != sym_ckpt["seed"]:
        raise ValueError("Seed mismatch between mech and sym checkpoints")

    seed = int(mech_ckpt["seed"])
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    model_mech = _load_model(args.mech_ckpt, num_classes=2).to(device)
    model_sym = _load_model(args.sym_ckpt, num_classes=2).to(device)

    data_cfg = DatasetConfig(task="forward", split="eval", size=args.eval_size, seed=seed)
    gen_cfg = ForwardChainConfig()
    dataset = ForwardChainDataset(data_cfg, gen_cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    alpha_mech = float(mech_ckpt["alpha"])
    beta_mech = float(mech_ckpt["beta"])
    alpha_sym = float(sym_ckpt["alpha"])
    beta_sym = float(sym_ckpt["beta"])
    sym_variant = str(sym_ckpt["variant"])

    gamma = None
    if sym_variant == "symmetric_control_v2_normmatched":
        gamma = _compute_gamma_stats(loader, model_sym, sym_variant, alpha_sym, beta_sym, device)["gamma"]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    spec_mech = OperatorSpec(alpha=alpha_mech, beta=beta_mech, variant="mechanism")
    spec_sym = OperatorSpec(alpha=alpha_sym, beta=beta_sym, gamma=gamma, variant=sym_variant)

    overrides = {
        "beta_zero": OperatorSpec(alpha=alpha_mech, beta=0.0, variant="mechanism"),
        "alpha_zero": OperatorSpec(alpha=0.0, beta=beta_mech, variant="mechanism"),
        "beta_flip": OperatorSpec(alpha=alpha_mech, beta=-beta_mech, variant="mechanism"),
    }

    sel_vals = {"baseline": [], "beta_zero": [], "alpha_zero": [], "beta_flip": []}
    adj_mech_vals = {"baseline": [], "beta_zero": [], "alpha_zero": [], "beta_flip": []}
    adj_sym_vals: List[float] = []

    rho_vals: List[float] = []
    A_vals: List[float] = []
    B_vals: List[float] = []
    B0_vals: List[float] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inst_path = out_path.with_suffix(".jsonl.gz")
    inst_f = gzip.open(inst_path, "wt", encoding="utf-8")

    with torch.no_grad():
        for input_ids, attn_mask, labels, metas in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            _logits_mech, probe_logits_mech, probe_U_mech = model_mech(
                input_ids, attn_mask=attn_mask, variant="mechanism", alpha=alpha_mech, beta=beta_mech, return_probe=True
            )
            _logits_sym, probe_logits_sym, probe_U_sym = model_sym(
                input_ids,
                attn_mask=attn_mask,
                variant=sym_variant,
                alpha=alpha_sym,
                beta=beta_sym,
                gamma=1.0 if gamma is None else gamma,
                return_probe=True,
            )

            L_mech = probe_logits_mech.cpu().numpy()
            U_mech = probe_U_mech.cpu().numpy()
            L_sym = probe_logits_sym.cpu().numpy()
            U_sym = probe_U_sym.cpu().numpy()

            for i, meta in enumerate(metas):
                Lm = L_mech[i].mean(axis=0)
                Um = U_mech[i]
                Ls = L_sym[i].mean(axis=0)
                Us = U_sym[i]
                run_id = dataset.run_id
                instance_id = meta.instance_id

                cand1, cand2 = candidate_token_spans(meta)
                prem = list(premise_token_set(meta))
                true_cand = cand1 if meta.true_index == 0 else cand2
                false_cand = cand2 if meta.true_index == 0 else cand1

                adj_sym = token_adj_loc_gap(
                    L=Ls,
                    U=Us,
                    spec=spec_sym,
                    k_tok=args.k_tok,
                    prem_tokens=prem,
                    cand_true_tokens=true_cand,
                    cand_false_tokens=false_cand,
                    run_id=run_id,
                    instance_id=instance_id,
                )
                adj_sym_vals.append(adj_sym)

                adj_mech = token_adj_loc_gap(
                    L=Lm,
                    U=Um,
                    spec=spec_mech,
                    k_tok=args.k_tok,
                    prem_tokens=prem,
                    cand_true_tokens=true_cand,
                    cand_false_tokens=false_cand,
                    run_id=run_id,
                    instance_id=instance_id,
                )
                adj_mech_vals["baseline"].append(adj_mech)
                sel_vals["baseline"].append(adj_mech - adj_sym)

                for key, spec in overrides.items():
                    adj_override = token_adj_loc_gap(
                        L=Lm,
                        U=Um,
                        spec=spec,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=run_id,
                        instance_id=instance_id,
                    )
                    adj_mech_vals[key].append(adj_override)
                    sel_vals[key].append(adj_override - adj_sym)

                A, B0 = scale_norms(Us, beta_sym)
                if sym_variant == "symmetric_control_v2_normmatched":
                    B = (gamma if gamma is not None else 1.0) * B0
                    rho = rho_ratio(A, B)
                else:
                    B = B0
                    rho = rho_ratio(A, B0)
                A_vals.append(float(A))
                B0_vals.append(float(B0))
                B_vals.append(float(B))
                rho_vals.append(float(rho))

                inst_f.write(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "label": int(labels[i].item()),
                            "adj_sym": float(adj_sym),
                            "adj_mech": float(adj_mech),
                            "sel_baseline": float(adj_mech - adj_sym),
                            "sel_beta_zero": float(sel_vals["beta_zero"][-1]),
                            "sel_alpha_zero": float(sel_vals["alpha_zero"][-1]),
                            "sel_beta_flip": float(sel_vals["beta_flip"][-1]),
                        }
                    )
                    + "\n"
                )

    inst_f.close()

    rho_median = float(np.median(np.asarray(rho_vals)))
    if sym_variant == "symmetric_control_v2_normmatched":
        B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
        zero_rate_B0 = float(np.mean(np.asarray(B0_vals) == 0.0)) if B0_vals else 0.0
        flag = not (B0_median > 0.0 and zero_rate_B0 <= 0.001)
    else:
        flag = not (0.9 <= rho_median <= 1.1)
        B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
        zero_rate_B0 = float(np.mean(np.asarray(B0_vals) == 0.0)) if B0_vals else 0.0

    out = {
        "seed": seed,
        "mech_ckpt": args.mech_ckpt,
        "sym_ckpt": args.sym_ckpt,
        "alpha_mech": alpha_mech,
        "beta_mech": beta_mech,
        "alpha_sym": alpha_sym,
        "beta_sym": beta_sym,
        "k_tok": args.k_tok,
        "control_variant": sym_variant,
        "control_gamma": float(gamma if gamma is not None else 1.0),
        "rho_median": rho_median,
        "rho_flag": flag,
        "A_median": float(np.median(np.asarray(A_vals))),
        "B_median": float(np.median(np.asarray(B_vals))),
        "B0_median": float(B0_median),
        "zero_rate_B0": float(zero_rate_B0),
        "baseline": {
            "SelLocGap": _summary(sel_vals["baseline"], seed=seed, n_boot=args.bootstrap),
            "AdjLocGap_mech": _summary(adj_mech_vals["baseline"], seed=seed, n_boot=args.bootstrap),
            "AdjLocGap_sym": _summary(adj_sym_vals, seed=seed, n_boot=args.bootstrap),
            "sign_test_p": _sign_test_pvalue(sel_vals["baseline"]),
        },
        "overrides": {
            "beta_zero": {
                "SelLocGap": _summary(sel_vals["beta_zero"], seed=seed, n_boot=args.bootstrap),
                "AdjLocGap_mech": _summary(adj_mech_vals["beta_zero"], seed=seed, n_boot=args.bootstrap),
                "sign_test_p": _sign_test_pvalue(sel_vals["beta_zero"]),
            },
            "alpha_zero": {
                "SelLocGap": _summary(sel_vals["alpha_zero"], seed=seed, n_boot=args.bootstrap),
                "AdjLocGap_mech": _summary(adj_mech_vals["alpha_zero"], seed=seed, n_boot=args.bootstrap),
                "sign_test_p": _sign_test_pvalue(sel_vals["alpha_zero"]),
            },
            "beta_flip": {
                "SelLocGap": _summary(sel_vals["beta_flip"], seed=seed, n_boot=args.bootstrap),
                "AdjLocGap_mech": _summary(adj_mech_vals["beta_flip"], seed=seed, n_boot=args.bootstrap),
                "sign_test_p": _sign_test_pvalue(sel_vals["beta_flip"]),
            },
        },
        "git_commit": _git_commit(),
    }

    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
