#!/usr/bin/env python
"""Phase 2/3 evaluation entrypoint."""

from __future__ import annotations

import argparse
import platform
import gzip
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import BackwardChainDataset, CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.data_phase3 import Phase3BackwardDataset, Phase3ForwardDataset
from nhsea.generators import BackwardChainConfig, CycleRegimeConfig, ForwardChainConfig, candidate_token_spans, premise_token_set
from nhsea.generators_phase3 import Phase3ChainConfig
from nhsea.metrics import (
    delta_cyc_stats,
    participation_ratio,
    prop_pipeline_a,
    prop_pipeline_b,
    token_adj_loc_gap,
)
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import OperatorSpec, build_run_operator, scale_norms, rho_ratio, token_weights


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
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": _bootstrap_ci(arr, seed=seed, n_boot=n_boot)["low"],
        "ci_high": _bootstrap_ci(arr, seed=seed, n_boot=n_boot)["high"],
    }


def _variant_spec(variant: str, alpha: float, beta: float, gamma: float | None = None) -> OperatorSpec:
    if variant == "mechanism":
        return OperatorSpec(alpha=alpha, beta=beta, variant="mechanism")
    if variant == "symmetric_control":
        return OperatorSpec(alpha=alpha, beta=beta, variant="symmetric_control")
    if variant == "symmetric_control_v2_normmatched":
        return OperatorSpec(alpha=alpha, beta=beta, gamma=gamma, variant="symmetric_control_v2_normmatched")
    if variant == "no_injection":
        return OperatorSpec(alpha=alpha, beta=0.0, variant="no_injection")
    if variant == "no_drift":
        return OperatorSpec(alpha=0.0, beta=beta, variant="no_drift")
    raise ValueError(f"Unknown variant: {variant}")


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
            probe_U_np = probe_U.cpu().numpy()
            for idx in range(probe_U_np.shape[0]):
                U = probe_U_np[idx]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--eval_size", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--k_prop", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--phase3", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    if ckpt["task"] == "forward":
        model.set_num_classes(2)
    elif ckpt["task"] == "cycle":
        model.set_num_classes(4)
    else:
        model.set_num_classes(2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = _resolve_device(args.device)
    model.to(device)

    phase3 = bool(ckpt.get("phase3", False) or args.phase3)
    data_cfg = DatasetConfig(task=ckpt["task"], split="eval", size=args.eval_size, seed=ckpt["seed"])
    if ckpt["task"] == "forward":
        if phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3ForwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = ForwardChainConfig()
            dataset = ForwardChainDataset(data_cfg, gen_cfg)
    elif ckpt["task"] == "cycle":
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)
    else:
        if phase3:
            gen_cfg = Phase3ChainConfig()
            dataset = Phase3BackwardDataset(data_cfg, gen_cfg)
        else:
            gen_cfg = BackwardChainConfig()
            dataset = BackwardChainDataset(data_cfg, gen_cfg)

    alpha = float(ckpt["alpha"])
    beta = float(ckpt["beta"])
    variant = str(ckpt["variant"])
    beta_eff = 0.0 if variant == "no_injection" else beta

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    gamma = None
    if variant == "symmetric_control_v2_normmatched":
        gamma = _compute_gamma_stats(loader, model, variant, alpha, beta, device)["gamma"]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    all_preds: List[int] = []
    all_labels: List[int] = []
    adj_loc_gaps: List[float] = []
    rho_vals: List[float] = []
    normA_vals: List[float] = []
    normB_vals: List[float] = []
    normB0_vals: List[float] = []

    cyc_vals_a: Dict[int, Dict[int, List[float]]] = {r: {2: [], 3: [], 4: []} for r in range(4)}
    cyc_vals_b: Dict[int, Dict[int, List[float]]] = {r: {2: [], 3: [], 4: []} for r in range(4)}
    pr_vals_a: Dict[int, List[float]] = {r: [] for r in range(4)}
    pr_vals_b: Dict[int, List[float]] = {r: [] for r in range(4)}

    spec_variant = _variant_spec(variant, alpha, beta, gamma=gamma)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_path = out_dir / "eval_instances.jsonl.gz"
    inst_f = gzip.open(inst_path, "wt", encoding="utf-8")

    with torch.no_grad():
        for input_ids, attn_mask, labels, metas in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            logits, probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=1.0 if gamma is None else gamma,
                return_probe=True,
            )

            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

            probe_logits_np = probe_logits.cpu().numpy()  # (B, H, T, T)
            probe_U_np = probe_U.cpu().numpy()  # (B, T, T)

            for i, meta in enumerate(metas):
                L = probe_logits_np[i].mean(axis=0)
                U = probe_U_np[i]
                run_id = dataset.run_id
                instance_id = meta.instance_id

                A, B0 = scale_norms(U, beta_eff)
                if variant == "symmetric_control_v2_normmatched":
                    B = (gamma if gamma is not None else 1.0) * B0
                    rho = rho_ratio(A, B)
                else:
                    B = B0
                    rho = rho_ratio(A, B0)
                rho_vals.append(float(rho))
                normA_vals.append(float(A))
                normB_vals.append(float(B))
                normB0_vals.append(float(B0))

                record = {
                    "instance_id": instance_id,
                    "label": int(labels[i].item()),
                    "pred": int(preds[i]),
                    "A": float(A),
                    "B": float(B),
                    "rho": float(rho),
                }
                if variant == "symmetric_control_v2_normmatched":
                    record["B0"] = float(B0)

                if ckpt["task"] == "forward":
                    cand1, cand2 = candidate_token_spans(meta)
                    prem = list(premise_token_set(meta))
                    true_cand = cand1 if meta.true_index == 0 else cand2
                    false_cand = cand2 if meta.true_index == 0 else cand1
                    adj = token_adj_loc_gap(
                        L=L,
                        U=U,
                        spec=spec_variant,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=run_id,
                        instance_id=instance_id,
                    )
                    adj_loc_gaps.append(adj)
                    record["adj_loc_gap"] = float(adj)
                elif ckpt["task"] == "cycle":
                    O = build_run_operator(L, U, spec_variant)
                    W_tok = token_weights(O)
                    W_prop_a = prop_pipeline_a(W_tok, meta.prop_spans, k_tok=args.k_tok, k_prop=args.k_prop)
                    W_prop_b = prop_pipeline_b(W_tok, meta.prop_spans, k_prop=args.k_prop)
                    delta_a = delta_cyc_stats(
                        W_prop_a,
                        run_id=run_id,
                        instance_id=instance_id,
                        n_perm=100,
                        ells=(2, 3, 4),
                        salt_prefix="PERMROW_A",
                    )
                    delta_b = delta_cyc_stats(
                        W_prop_b,
                        run_id=run_id,
                        instance_id=instance_id,
                        n_perm=100,
                        ells=(2, 3, 4),
                        salt_prefix="PERMROW_B",
                    )
                    for ell in (2, 3, 4):
                        cyc_vals_a[meta.regime][ell].append(delta_a[ell])
                        cyc_vals_b[meta.regime][ell].append(delta_b[ell])
                    pr_vals_a[meta.regime].append(participation_ratio(W_prop_a, meta.premises))
                    pr_vals_b[meta.regime].append(participation_ratio(W_prop_b, meta.premises))
                    record["regime"] = int(meta.regime)
                    record["delta_cyc_a"] = {str(ell): float(delta_a[ell]) for ell in (2, 3, 4)}
                    record["delta_cyc_b"] = {str(ell): float(delta_b[ell]) for ell in (2, 3, 4)}
                    record["pr_a"] = float(participation_ratio(W_prop_a, meta.premises))
                    record["pr_b"] = float(participation_ratio(W_prop_b, meta.premises))
                else:
                    record["adj_loc_gap"] = None

                if inst_f is not None:
                    inst_f.write(json.dumps(record) + "\n")

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    rho_median = float(np.median(np.asarray(rho_vals)))
    if variant == "symmetric_control_v2_normmatched":
        B0_median = float(np.median(np.asarray(normB0_vals))) if normB0_vals else 0.0
        zero_rate_B0 = 0.0
        if normB0_vals:
            zero_rate_B0 = float(np.mean(np.asarray(normB0_vals) == 0.0))
        flag = not (B0_median > 0.0 and zero_rate_B0 <= 0.001)
    else:
        flag = not (0.9 <= rho_median <= 1.1)

    out = {
        "task": ckpt["task"],
        "variant": variant,
        "phase3": phase3,
        "alpha": alpha,
        "beta": beta,
        "k_tok": args.k_tok,
        "k_prop": args.k_prop,
        "accuracy": accuracy,
        "rho_median": rho_median,
        "A_median": float(np.median(np.asarray(normA_vals))),
        "B_median": float(np.median(np.asarray(normB_vals))),
        "rho_flag": flag,
        "git_commit": _git_commit(),
    }
    if variant == "symmetric_control_v2_normmatched":
        out["sym_control_v2"] = {
            "A_median": float(np.median(np.asarray(normA_vals))) if normA_vals else 0.0,
            "B0_median": float(np.median(np.asarray(normB0_vals))) if normB0_vals else 0.0,
            "gamma": float(gamma if gamma is not None else 1.0),
            "zero_rate_B0": float(np.mean(np.asarray(normB0_vals) == 0.0)) if normB0_vals else 0.0,
        }

    if ckpt["task"] == "forward":
        out["AdjLocGap"] = _summary(adj_loc_gaps, seed=ckpt["seed"], n_boot=args.bootstrap)
    elif ckpt["task"] == "cycle":
        cyc_summary_a = {str(r): {str(ell): _summary(cyc_vals_a[r][ell], seed=ckpt["seed"], n_boot=args.bootstrap) for ell in (2, 3, 4)} for r in range(4)}
        cyc_summary_b = {str(r): {str(ell): _summary(cyc_vals_b[r][ell], seed=ckpt["seed"], n_boot=args.bootstrap) for ell in (2, 3, 4)} for r in range(4)}
        pr_summary_a = {str(r): _summary(pr_vals_a[r], seed=ckpt["seed"], n_boot=args.bootstrap) for r in range(4)}
        pr_summary_b = {str(r): _summary(pr_vals_b[r], seed=ckpt["seed"], n_boot=args.bootstrap) for r in range(4)}
        out["DeltaCyc_A"] = cyc_summary_a
        out["DeltaCyc_B"] = cyc_summary_b
        out["PR_A"] = pr_summary_a
        out["PR_B"] = pr_summary_b

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    eval_json_path = out_dir / "eval_summary.json"
    eval_json_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    csv_path = out_dir / "summary.csv"
    csv_path.write_text("summary\n" + json.dumps(out))
    eval_csv_path = out_dir / "eval_summary.csv"
    eval_csv_path.write_text("summary\n" + json.dumps(out))

    eval_cfg = {
        "checkpoint": str(args.checkpoint),
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "k_tok": args.k_tok,
        "k_prop": args.k_prop,
        "bootstrap": args.bootstrap,
        "phase3": phase3,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gen_cfg": gen_cfg.__dict__,
    }
    if variant == "symmetric_control_v2_normmatched":
        eval_cfg["gamma"] = float(gamma if gamma is not None else 1.0)
    (out_dir / "eval_config.json").write_text(json.dumps(eval_cfg, indent=2, sort_keys=True))

    if inst_f is not None:
        inst_f.close()

    print(f"Wrote {json_path}")
    print(f"Wrote {eval_json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {eval_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
