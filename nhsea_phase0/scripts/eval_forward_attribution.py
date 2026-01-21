#!/usr/bin/env python
"""Attribution eval for forward SelLocGap across operator overrides."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

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


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


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
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    ci = _bootstrap_ci(arr, seed=seed, n_boot=n_boot)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": ci["low"],
        "ci_high": ci["high"],
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


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def _load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> TinyTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    model.set_num_classes(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def _load_gen_cfg(run_dir: Path) -> ForwardChainConfig:
    eval_cfg_path = run_dir / "eval_config.json"
    if eval_cfg_path.exists():
        eval_cfg = json.loads(eval_cfg_path.read_text())
        gen_cfg_dict = eval_cfg.get("gen_cfg", {})
        if gen_cfg_dict:
            return ForwardChainConfig(**gen_cfg_dict)
    return ForwardChainConfig()


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


def _summarize_seed(values: Dict[str, List[float]], seed: int, n_boot: int) -> Dict[str, Dict[str, float]]:
    return {k: _summary(v, seed=seed, n_boot=n_boot) for k, v in values.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, default="runs/phase2_claimA_v2")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--eval_n", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--shared_rand_runid", type=str, default="")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    seeds = _parse_seeds(args.seeds)
    device = _resolve_device(args.device)

    per_seed_outputs: List[Dict[str, str]] = []
    pooled = {
        "SelLocGap_baseline": [],
        "SelLocGap_beta0": [],
        "SelLocGap_alpha0": [],
        "SelLocGap_betaflip": [],
        "Delta_infer_mech": [],
        "Delta_infer_sym": [],
        "Delta_learn": [],
    }

    pooled_shared = {
        "SelLocGap_baseline": [],
        "SelLocGap_beta0": [],
    }

    for seed in seeds:
        mech_dir = run_root / f"phase2_v2_forward_mechanism_seed{seed}"
        sym_dir = run_root / f"phase2_v2_forward_symmetric_control_v2_normmatched_seed{seed}"
        noinj_dir = run_root / f"phase2_v2_forward_no_injection_seed{seed}"

        if not (mech_dir.exists() and sym_dir.exists() and noinj_dir.exists()):
            raise FileNotFoundError(f"Missing run dirs for seed {seed}")

        mech_ckpt = _load_checkpoint(mech_dir / "checkpoint_final.pt")
        sym_ckpt = _load_checkpoint(sym_dir / "checkpoint_final.pt")
        noinj_ckpt = _load_checkpoint(noinj_dir / "checkpoint_final.pt")

        gen_cfg = _load_gen_cfg(mech_dir)
        data_cfg = DatasetConfig(
            task="forward",
            split="eval",
            size=args.eval_n,
            seed=seed,
            T=gen_cfg.T,
            vocab_size=gen_cfg.vocab_size,
        )
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        model_mech = _load_model(mech_dir / "checkpoint_final.pt", num_classes=2, device=device)
        model_sym = _load_model(sym_dir / "checkpoint_final.pt", num_classes=2, device=device)
        model_noinj = _load_model(noinj_dir / "checkpoint_final.pt", num_classes=2, device=device)

        alpha_mech = float(mech_ckpt["alpha"])
        beta_mech = float(mech_ckpt["beta"])
        alpha_sym = float(sym_ckpt["alpha"])
        beta_sym = float(sym_ckpt["beta"])
        alpha_noinj = float(noinj_ckpt["alpha"])
        beta_noinj = float(noinj_ckpt["beta"])

        gamma_stats = _compute_gamma_stats(loader, model_sym, "symmetric_control_v2_normmatched", alpha_sym, beta_sym, device)
        gamma = gamma_stats["gamma"]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        spec_mech_base = OperatorSpec(alpha=alpha_mech, beta=beta_mech, variant="mechanism")
        spec_sym_base = OperatorSpec(
            alpha=alpha_sym, beta=beta_sym, gamma=gamma, variant="symmetric_control_v2_normmatched"
        )
        spec_noinj_base = OperatorSpec(alpha=alpha_noinj, beta=0.0, variant="no_injection")

        spec_mech_beta0 = OperatorSpec(alpha=alpha_mech, beta=0.0, variant="mechanism")
        spec_sym_beta0 = OperatorSpec(alpha=alpha_sym, beta=0.0, gamma=gamma, variant="symmetric_control_v2_normmatched")
        spec_noinj_beta0 = spec_noinj_base

        spec_mech_alpha0 = OperatorSpec(alpha=0.0, beta=beta_mech, variant="mechanism")
        spec_sym_alpha0 = OperatorSpec(alpha=0.0, beta=beta_sym, gamma=gamma, variant="symmetric_control_v2_normmatched")

        spec_mech_betaflip = OperatorSpec(alpha=alpha_mech, beta=-beta_mech, variant="mechanism")
        spec_sym_betaflip = OperatorSpec(
            alpha=alpha_sym, beta=-beta_sym, gamma=gamma, variant="symmetric_control_v2_normmatched"
        )

        sel_vals = {
            "baseline": [],
            "beta0": [],
            "alpha0": [],
            "betaflip": [],
        }
        sel_vals_shared = {
            "baseline": [],
            "beta0": [],
        }

        adj_mech_vals = {
            "baseline": [],
            "beta0": [],
            "alpha0": [],
            "betaflip": [],
        }
        adj_sym_vals = {
            "baseline": [],
            "beta0": [],
            "alpha0": [],
            "betaflip": [],
        }
        adj_noinj_vals = {
            "baseline": [],
            "beta0": [],
        }

        delta_infer_mech = []
        delta_infer_sym = []
        delta_learn = []

        rand_run_id = args.shared_rand_runid if args.shared_rand_runid else dataset.run_id

        with torch.no_grad():
            for input_ids, attn_mask, _labels, metas in loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                _logits_mech, probe_logits_mech, probe_U_mech = model_mech(
                    input_ids,
                    attn_mask=attn_mask,
                    variant="mechanism",
                    alpha=alpha_mech,
                    beta=beta_mech,
                    return_probe=True,
                )
                _logits_sym, probe_logits_sym, probe_U_sym = model_sym(
                    input_ids,
                    attn_mask=attn_mask,
                    variant="symmetric_control_v2_normmatched",
                    alpha=alpha_sym,
                    beta=beta_sym,
                    gamma=gamma,
                    return_probe=True,
                )
                _logits_noinj, probe_logits_noinj, probe_U_noinj = model_noinj(
                    input_ids,
                    attn_mask=attn_mask,
                    variant="no_injection",
                    alpha=alpha_noinj,
                    beta=0.0,
                    return_probe=True,
                )

                L_mech = probe_logits_mech.cpu().numpy()
                U_mech = probe_U_mech.cpu().numpy()
                L_sym = probe_logits_sym.cpu().numpy()
                U_sym = probe_U_sym.cpu().numpy()
                L_noinj = probe_logits_noinj.cpu().numpy()
                U_noinj = probe_U_noinj.cpu().numpy()

                for i, meta in enumerate(metas):
                    Lm = L_mech[i].mean(axis=0)
                    Um = U_mech[i]
                    Ls = L_sym[i].mean(axis=0)
                    Us = U_sym[i]
                    Ln = L_noinj[i].mean(axis=0)
                    Un = U_noinj[i]

                    instance_id = meta.instance_id
                    cand1, cand2 = candidate_token_spans(meta)
                    prem = list(premise_token_set(meta))
                    true_cand = cand1 if meta.true_index == 0 else cand2
                    false_cand = cand2 if meta.true_index == 0 else cand1

                    adj_mech_base = token_adj_loc_gap(
                        L=Lm,
                        U=Um,
                        spec=spec_mech_base,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_sym_base = token_adj_loc_gap(
                        L=Ls,
                        U=Us,
                        spec=spec_sym_base,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_noinj_base = token_adj_loc_gap(
                        L=Ln,
                        U=Un,
                        spec=spec_noinj_base,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )

                    adj_mech_b0 = token_adj_loc_gap(
                        L=Lm,
                        U=Um,
                        spec=spec_mech_beta0,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_sym_b0 = token_adj_loc_gap(
                        L=Ls,
                        U=Us,
                        spec=spec_sym_beta0,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_noinj_b0 = token_adj_loc_gap(
                        L=Ln,
                        U=Un,
                        spec=spec_noinj_beta0,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )

                    adj_mech_a0 = token_adj_loc_gap(
                        L=Lm,
                        U=Um,
                        spec=spec_mech_alpha0,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_sym_a0 = token_adj_loc_gap(
                        L=Ls,
                        U=Us,
                        spec=spec_sym_alpha0,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )

                    adj_mech_bf = token_adj_loc_gap(
                        L=Lm,
                        U=Um,
                        spec=spec_mech_betaflip,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )
                    adj_sym_bf = token_adj_loc_gap(
                        L=Ls,
                        U=Us,
                        spec=spec_sym_betaflip,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=rand_run_id,
                        instance_id=instance_id,
                    )

                    adj_mech_vals["baseline"].append(adj_mech_base)
                    adj_sym_vals["baseline"].append(adj_sym_base)
                    adj_noinj_vals["baseline"].append(adj_noinj_base)

                    adj_mech_vals["beta0"].append(adj_mech_b0)
                    adj_sym_vals["beta0"].append(adj_sym_b0)
                    adj_noinj_vals["beta0"].append(adj_noinj_b0)

                    adj_mech_vals["alpha0"].append(adj_mech_a0)
                    adj_sym_vals["alpha0"].append(adj_sym_a0)

                    adj_mech_vals["betaflip"].append(adj_mech_bf)
                    adj_sym_vals["betaflip"].append(adj_sym_bf)

                    sel_vals["baseline"].append(adj_mech_base - adj_sym_base)
                    sel_vals["beta0"].append(adj_mech_b0 - adj_sym_b0)
                    sel_vals["alpha0"].append(adj_mech_a0 - adj_sym_a0)
                    sel_vals["betaflip"].append(adj_mech_bf - adj_sym_bf)

                    if args.shared_rand_runid:
                        sel_vals_shared["baseline"].append(adj_mech_base - adj_sym_base)
                        sel_vals_shared["beta0"].append(adj_mech_b0 - adj_sym_b0)

                    delta_infer_mech.append(adj_mech_base - adj_mech_b0)
                    delta_infer_sym.append(adj_sym_base - adj_sym_b0)
                    delta_learn.append(adj_mech_b0 - adj_noinj_b0)

        seed_out = {
            "seed": seed,
            "run_root": str(run_root),
            "run_id": dataset.run_id,
            "shared_rand_runid": args.shared_rand_runid or None,
            "gamma_sym": gamma,
            "gamma_stats": gamma_stats,
            "AdjLocGap_mech": _summarize_seed(adj_mech_vals, seed, args.bootstrap),
            "AdjLocGap_sym": _summarize_seed(adj_sym_vals, seed, args.bootstrap),
            "AdjLocGap_noinj": _summarize_seed(adj_noinj_vals, seed, args.bootstrap),
            "SelLocGap": _summarize_seed(sel_vals, seed, args.bootstrap),
            "Delta_infer_mech": _summary(delta_infer_mech, seed=seed, n_boot=args.bootstrap),
            "Delta_infer_sym": _summary(delta_infer_sym, seed=seed, n_boot=args.bootstrap),
            "Delta_learn": _summary(delta_learn, seed=seed, n_boot=args.bootstrap),
            "sign_test_p": {
                "baseline": _sign_test_pvalue(sel_vals["baseline"]),
                "beta0": _sign_test_pvalue(sel_vals["beta0"]),
                "alpha0": _sign_test_pvalue(sel_vals["alpha0"]),
                "betaflip": _sign_test_pvalue(sel_vals["betaflip"]),
            },
            "git_commit": _git_commit(),
        }
        if args.shared_rand_runid:
            seed_out["SelLocGap_sharedRand"] = _summarize_seed(sel_vals_shared, seed, args.bootstrap)

        out_path = run_root / f"phase2_attribution_seed{seed}.json"
        out_path.write_text(json.dumps(seed_out, indent=2, sort_keys=True))

        per_seed_outputs.append(
            {
                "seed": str(seed),
                "SelLocGap_baseline_mean": seed_out["SelLocGap"]["baseline"]["mean"],
                "SelLocGap_baseline_ci_low": seed_out["SelLocGap"]["baseline"]["ci_low"],
                "SelLocGap_baseline_ci_high": seed_out["SelLocGap"]["baseline"]["ci_high"],
                "SelLocGap_beta0_mean": seed_out["SelLocGap"]["beta0"]["mean"],
                "SelLocGap_beta0_ci_low": seed_out["SelLocGap"]["beta0"]["ci_low"],
                "SelLocGap_beta0_ci_high": seed_out["SelLocGap"]["beta0"]["ci_high"],
                "SelLocGap_alpha0_mean": seed_out["SelLocGap"]["alpha0"]["mean"],
                "SelLocGap_alpha0_ci_low": seed_out["SelLocGap"]["alpha0"]["ci_low"],
                "SelLocGap_alpha0_ci_high": seed_out["SelLocGap"]["alpha0"]["ci_high"],
                "SelLocGap_betaflip_mean": seed_out["SelLocGap"]["betaflip"]["mean"],
                "SelLocGap_betaflip_ci_low": seed_out["SelLocGap"]["betaflip"]["ci_low"],
                "SelLocGap_betaflip_ci_high": seed_out["SelLocGap"]["betaflip"]["ci_high"],
                "Delta_infer_mech_mean": seed_out["Delta_infer_mech"]["mean"],
                "Delta_infer_mech_ci_low": seed_out["Delta_infer_mech"]["ci_low"],
                "Delta_infer_mech_ci_high": seed_out["Delta_infer_mech"]["ci_high"],
                "Delta_infer_sym_mean": seed_out["Delta_infer_sym"]["mean"],
                "Delta_infer_sym_ci_low": seed_out["Delta_infer_sym"]["ci_low"],
                "Delta_infer_sym_ci_high": seed_out["Delta_infer_sym"]["ci_high"],
                "Delta_learn_mean": seed_out["Delta_learn"]["mean"],
                "Delta_learn_ci_low": seed_out["Delta_learn"]["ci_low"],
                "Delta_learn_ci_high": seed_out["Delta_learn"]["ci_high"],
            }
        )

        pooled["SelLocGap_baseline"].extend(sel_vals["baseline"])
        pooled["SelLocGap_beta0"].extend(sel_vals["beta0"])
        pooled["SelLocGap_alpha0"].extend(sel_vals["alpha0"])
        pooled["SelLocGap_betaflip"].extend(sel_vals["betaflip"])
        pooled["Delta_infer_mech"].extend(delta_infer_mech)
        pooled["Delta_infer_sym"].extend(delta_infer_sym)
        pooled["Delta_learn"].extend(delta_learn)

        if args.shared_rand_runid:
            pooled_shared["SelLocGap_baseline"].extend(sel_vals_shared["baseline"])
            pooled_shared["SelLocGap_beta0"].extend(sel_vals_shared["beta0"])

    pooled_summary = {k: _summary(v, seed=0, n_boot=args.bootstrap) for k, v in pooled.items()}
    pooled_row = {
        "seed": "pooled",
        "SelLocGap_baseline_mean": pooled_summary["SelLocGap_baseline"]["mean"],
        "SelLocGap_baseline_ci_low": pooled_summary["SelLocGap_baseline"]["ci_low"],
        "SelLocGap_baseline_ci_high": pooled_summary["SelLocGap_baseline"]["ci_high"],
        "SelLocGap_beta0_mean": pooled_summary["SelLocGap_beta0"]["mean"],
        "SelLocGap_beta0_ci_low": pooled_summary["SelLocGap_beta0"]["ci_low"],
        "SelLocGap_beta0_ci_high": pooled_summary["SelLocGap_beta0"]["ci_high"],
        "SelLocGap_alpha0_mean": pooled_summary["SelLocGap_alpha0"]["mean"],
        "SelLocGap_alpha0_ci_low": pooled_summary["SelLocGap_alpha0"]["ci_low"],
        "SelLocGap_alpha0_ci_high": pooled_summary["SelLocGap_alpha0"]["ci_high"],
        "SelLocGap_betaflip_mean": pooled_summary["SelLocGap_betaflip"]["mean"],
        "SelLocGap_betaflip_ci_low": pooled_summary["SelLocGap_betaflip"]["ci_low"],
        "SelLocGap_betaflip_ci_high": pooled_summary["SelLocGap_betaflip"]["ci_high"],
        "Delta_infer_mech_mean": pooled_summary["Delta_infer_mech"]["mean"],
        "Delta_infer_mech_ci_low": pooled_summary["Delta_infer_mech"]["ci_low"],
        "Delta_infer_mech_ci_high": pooled_summary["Delta_infer_mech"]["ci_high"],
        "Delta_infer_sym_mean": pooled_summary["Delta_infer_sym"]["mean"],
        "Delta_infer_sym_ci_low": pooled_summary["Delta_infer_sym"]["ci_low"],
        "Delta_infer_sym_ci_high": pooled_summary["Delta_infer_sym"]["ci_high"],
        "Delta_learn_mean": pooled_summary["Delta_learn"]["mean"],
        "Delta_learn_ci_low": pooled_summary["Delta_learn"]["ci_low"],
        "Delta_learn_ci_high": pooled_summary["Delta_learn"]["ci_high"],
    }

    master_rows = per_seed_outputs + [pooled_row]
    master_path = run_root / "phase2_attribution_master.csv"
    keys = list(master_rows[0].keys())
    master_path.write_text(",".join(keys) + "\n")
    with master_path.open("a", encoding="utf-8") as f:
        for row in master_rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    report_lines = ["# Phase 2 Attribution Report", ""]
    for row in per_seed_outputs:
        report_lines.append(
            f"- seed {row['seed']}: SelLocGap_baseline mean={row['SelLocGap_baseline_mean']:.6f} "
            f"CI=[{row['SelLocGap_baseline_ci_low']:.6f},{row['SelLocGap_baseline_ci_high']:.6f}]"
        )
        report_lines.append(
            f"  Delta_infer_mech mean={row['Delta_infer_mech_mean']:.6f} "
            f"CI=[{row['Delta_infer_mech_ci_low']:.6f},{row['Delta_infer_mech_ci_high']:.6f}]"
        )
        report_lines.append(
            f"  Delta_infer_sym mean={row['Delta_infer_sym_mean']:.6f} "
            f"CI=[{row['Delta_infer_sym_ci_low']:.6f},{row['Delta_infer_sym_ci_high']:.6f}]"
        )
        report_lines.append(
            f"  Delta_learn mean={row['Delta_learn_mean']:.6f} "
            f"CI=[{row['Delta_learn_ci_low']:.6f},{row['Delta_learn_ci_high']:.6f}]"
        )
    report_lines.append("")
    report_lines.append(
        f"- pooled Delta_infer_mech mean={pooled_summary['Delta_infer_mech']['mean']:.6f} "
        f"CI=[{pooled_summary['Delta_infer_mech']['ci_low']:.6f},{pooled_summary['Delta_infer_mech']['ci_high']:.6f}]"
    )
    report_lines.append(
        f"- pooled Delta_infer_sym mean={pooled_summary['Delta_infer_sym']['mean']:.6f} "
        f"CI=[{pooled_summary['Delta_infer_sym']['ci_low']:.6f},{pooled_summary['Delta_infer_sym']['ci_high']:.6f}]"
    )
    report_lines.append(
        f"- pooled Delta_learn mean={pooled_summary['Delta_learn']['mean']:.6f} "
        f"CI=[{pooled_summary['Delta_learn']['ci_low']:.6f},{pooled_summary['Delta_learn']['ci_high']:.6f}]"
    )

    if args.shared_rand_runid:
        pooled_shared_summary = {k: _summary(v, seed=0, n_boot=args.bootstrap) for k, v in pooled_shared.items()}
        report_lines.append("")
        report_lines.append("## Exploratory: shared rand run_id")
        report_lines.append(
            f"- pooled SelLocGap_sharedRand_baseline mean={pooled_shared_summary['SelLocGap_baseline']['mean']:.6f} "
            f"CI=[{pooled_shared_summary['SelLocGap_baseline']['ci_low']:.6f},{pooled_shared_summary['SelLocGap_baseline']['ci_high']:.6f}]"
        )
        report_lines.append(
            f"- pooled SelLocGap_sharedRand_beta0 mean={pooled_shared_summary['SelLocGap_beta0']['mean']:.6f} "
            f"CI=[{pooled_shared_summary['SelLocGap_beta0']['ci_low']:.6f},{pooled_shared_summary['SelLocGap_beta0']['ci_high']:.6f}]"
        )

    report_path = run_root / "phase2_attribution_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
