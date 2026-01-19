#!/usr/bin/env python
"""Phase 1 evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import CycleRegimeDataset, DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import CycleRegimeConfig, ForwardChainConfig, candidate_token_spans, premise_token_set
from nhsea.metrics import (
    delta_cyc_stats,
    participation_ratio,
    prop_pipeline_a,
    prop_pipeline_b,
    token_sel_loc_gap,
)
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import OperatorSpec, build_run_operator, scale_norms, rho_ratio, token_weights


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 200) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    return {"low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


def _summary(values: List[float], seed: int = 0) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": _bootstrap_ci(arr, seed=seed)["low"],
        "ci_high": _bootstrap_ci(arr, seed=seed)["high"],
    }


def _variant_spec(variant: str, alpha: float, beta: float) -> OperatorSpec:
    if variant == "mechanism":
        return OperatorSpec(alpha=alpha, beta=beta, variant="mechanism")
    if variant == "symmetric_control":
        return OperatorSpec(alpha=alpha, beta=beta, variant="symmetric_control")
    if variant == "no_injection":
        return OperatorSpec(alpha=alpha, beta=0.0, variant="no_injection")
    if variant == "no_drift":
        return OperatorSpec(alpha=0.0, beta=beta, variant="no_drift")
    raise ValueError(f"Unknown variant: {variant}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--eval_size", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--k_prop", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyTransformer(model_cfg, probe_layer=2)
    if ckpt["task"] == "forward":
        model.set_num_classes(2)
    else:
        model.set_num_classes(4)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_cfg = DatasetConfig(task=ckpt["task"], split="eval", size=args.eval_size, seed=ckpt["seed"])
    if ckpt["task"] == "forward":
        gen_cfg = ForwardChainConfig()
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
    else:
        gen_cfg = CycleRegimeConfig()
        dataset = CycleRegimeDataset(data_cfg, gen_cfg)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    alpha = float(ckpt["alpha"])
    beta = float(ckpt["beta"])
    variant = str(ckpt["variant"])
    beta_eff = 0.0 if variant == "no_injection" else beta

    all_preds: List[int] = []
    all_labels: List[int] = []
    sel_loc_gaps: List[float] = []
    rho_vals: List[float] = []
    normA_vals: List[float] = []
    normB_vals: List[float] = []

    cyc_vals_a: Dict[int, Dict[int, List[float]]] = {r: {2: [], 3: [], 4: []} for r in range(4)}
    cyc_vals_b: Dict[int, Dict[int, List[float]]] = {r: {2: [], 3: [], 4: []} for r in range(4)}
    pr_vals_a: Dict[int, List[float]] = {r: [] for r in range(4)}
    pr_vals_b: Dict[int, List[float]] = {r: [] for r in range(4)}

    spec_variant = _variant_spec(variant, alpha, beta)
    spec_mech = OperatorSpec(alpha=alpha, beta=beta, variant="mechanism")
    spec_sym = OperatorSpec(alpha=alpha, beta=beta, variant="symmetric_control")

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

                A, B = scale_norms(U, beta_eff)
                rho = rho_ratio(A, B)
                rho_vals.append(float(rho))
                normA_vals.append(float(A))
                normB_vals.append(float(B))

                if ckpt["task"] == "forward":
                    cand1, cand2 = candidate_token_spans(meta)
                    prem = list(premise_token_set(meta))
                    true_cand = cand1 if meta.true_index == 0 else cand2
                    false_cand = cand2 if meta.true_index == 0 else cand1
                    sel = token_sel_loc_gap(
                        L=L,
                        U=U,
                        spec_mech=spec_mech,
                        spec_sym=spec_sym,
                        k_tok=args.k_tok,
                        prem_tokens=prem,
                        cand_true_tokens=true_cand,
                        cand_false_tokens=false_cand,
                        run_id=run_id,
                        instance_id=instance_id,
                    )
                    sel_loc_gaps.append(sel.sel_loc_gap)
                else:
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

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    rho_median = float(np.median(np.asarray(rho_vals)))
    flag = not (0.9 <= rho_median <= 1.1)

    out = {
        "task": ckpt["task"],
        "variant": variant,
        "alpha": alpha,
        "beta": beta,
        "k_tok": args.k_tok,
        "k_prop": args.k_prop,
        "accuracy": accuracy,
        "rho_median": rho_median,
        "A_median": float(np.median(np.asarray(normA_vals))),
        "B_median": float(np.median(np.asarray(normB_vals))),
        "rho_flag": flag,
    }

    if ckpt["task"] == "forward":
        out["SelLocGap"] = _summary(sel_loc_gaps, seed=ckpt["seed"])
    else:
        cyc_summary_a = {str(r): {str(ell): _summary(cyc_vals_a[r][ell], seed=ckpt["seed"]) for ell in (2, 3, 4)} for r in range(4)}
        cyc_summary_b = {str(r): {str(ell): _summary(cyc_vals_b[r][ell], seed=ckpt["seed"]) for ell in (2, 3, 4)} for r in range(4)}
        pr_summary_a = {str(r): _summary(pr_vals_a[r], seed=ckpt["seed"]) for r in range(4)}
        pr_summary_b = {str(r): _summary(pr_vals_b[r], seed=ckpt["seed"]) for r in range(4)}
        out["DeltaCyc_A"] = cyc_summary_a
        out["DeltaCyc_B"] = cyc_summary_b
        out["PR_A"] = pr_summary_a
        out["PR_B"] = pr_summary_b

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(args.checkpoint).stem
    json_path = out_dir / f"{run_name}_summary.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    csv_path = out_dir / f"{run_name}_summary.csv"
    csv_path.write_text("summary\n" + json.dumps(out))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
