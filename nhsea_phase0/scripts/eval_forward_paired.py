#!/usr/bin/env python
"""Paired SelLocGap evaluation for forward task (mech minus sym)."""

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
from nhsea.operator import OperatorSpec, scale_norms, rho_ratio


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


def _sign_test_pvalue(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 1.0
    k = sum(1 for v in values if v > 0)
    # Normal approximation, one-sided (greater than 0).
    mean = n * 0.5
    var = n * 0.25
    if var == 0:
        return 1.0
    z = (k - mean) / math.sqrt(var)
    # one-sided p-value
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
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    mech_ckpt = torch.load(args.mech_ckpt, map_location="cpu")
    sym_ckpt = torch.load(args.sym_ckpt, map_location="cpu")
    if mech_ckpt["task"] != "forward" or sym_ckpt["task"] != "forward":
        raise ValueError("Both checkpoints must be forward task")
    if mech_ckpt["seed"] != sym_ckpt["seed"]:
        raise ValueError("Seed mismatch between mech and sym checkpoints")

    seed = int(mech_ckpt["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    spec_mech = OperatorSpec(alpha=alpha_mech, beta=beta_mech, variant="mechanism")
    spec_sym = OperatorSpec(alpha=alpha_sym, beta=beta_sym, variant="symmetric_control")

    sel_loc_vals: List[float] = []
    adj_mech_vals: List[float] = []
    adj_sym_vals: List[float] = []
    rho_vals: List[float] = []
    normA_vals: List[float] = []
    normB_vals: List[float] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_path = out_dir / "eval_instances.jsonl.gz"
    inst_f = gzip.open(inst_path, "wt", encoding="utf-8")

    with torch.no_grad():
        for input_ids, attn_mask, labels, metas in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            logits_mech, probe_logits_mech, probe_U_mech = model_mech(
                input_ids, attn_mask=attn_mask, variant="mechanism", alpha=alpha_mech, beta=beta_mech, return_probe=True
            )
            logits_sym, probe_logits_sym, probe_U_sym = model_sym(
                input_ids, attn_mask=attn_mask, variant="symmetric_control", alpha=alpha_sym, beta=beta_sym, return_probe=True
            )

            preds_mech = torch.argmax(logits_mech, dim=-1).cpu().numpy().tolist()
            preds_sym = torch.argmax(logits_sym, dim=-1).cpu().numpy().tolist()

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
                sel = adj_mech - adj_sym

                A, B = scale_norms(Us, beta_sym)
                rho = rho_ratio(A, B)

                adj_mech_vals.append(adj_mech)
                adj_sym_vals.append(adj_sym)
                sel_loc_vals.append(sel)
                rho_vals.append(float(rho))
                normA_vals.append(float(A))
                normB_vals.append(float(B))

                record = {
                    "instance_id": instance_id,
                    "label": int(labels[i].item()),
                    "pred_mech": int(preds_mech[i]),
                    "pred_sym": int(preds_sym[i]),
                    "adj_locgap_mech": float(adj_mech),
                    "adj_locgap_sym": float(adj_sym),
                    "sel_loc_gap": float(sel),
                    "A": float(A),
                    "B": float(B),
                    "rho": float(rho),
                }
                inst_f.write(json.dumps(record) + "\n")

    inst_f.close()

    rho_median = float(np.median(np.asarray(rho_vals)))
    flag = not (0.9 <= rho_median <= 1.1)

    out = {
        "task": "forward",
        "variant": "paired_mech_minus_sym",
        "alpha_mech": alpha_mech,
        "beta_mech": beta_mech,
        "alpha_sym": alpha_sym,
        "beta_sym": beta_sym,
        "k_tok": args.k_tok,
        "adj_locgap_mech": _summary(adj_mech_vals, seed=seed, n_boot=args.bootstrap),
        "adj_locgap_sym": _summary(adj_sym_vals, seed=seed, n_boot=args.bootstrap),
        "SelLocGap": _summary(sel_loc_vals, seed=seed, n_boot=args.bootstrap),
        "sign_test_p": _sign_test_pvalue(sel_loc_vals),
        "rho_median": rho_median,
        "A_median": float(np.median(np.asarray(normA_vals))),
        "B_median": float(np.median(np.asarray(normB_vals))),
        "rho_flag": flag,
        "git_commit": _git_commit(),
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    csv_path = out_dir / "summary.csv"
    csv_path.write_text("summary\n" + json.dumps(out))

    eval_cfg = {
        "mech_checkpoint": args.mech_ckpt,
        "sym_checkpoint": args.sym_ckpt,
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "k_tok": args.k_tok,
        "bootstrap": args.bootstrap,
        "git_commit": _git_commit(),
        "gen_cfg": gen_cfg.__dict__,
    }
    (out_dir / "eval_config.json").write_text(json.dumps(eval_cfg, indent=2, sort_keys=True))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
