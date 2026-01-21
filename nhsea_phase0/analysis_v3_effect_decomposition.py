"""Decompose v3 SelLoc effects under eval-time operator overrides."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from nhsea.data_v3 import V3Dataset, V3DatasetConfig, collate_batch
from nhsea.generators_v3 import V3Config
from nhsea.metrics import token_adj_loc_gap
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import OperatorSpec, build_run_operator, rho_ratio, scale_norms


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def _load_eval_size(run_dir: Path, default: int) -> int:
    summary = run_dir / "eval_conclusion_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        return int(data.get("eval_size", default))
    return default


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _compute_gamma_stats(
    model: TinyTransformer,
    dataset: V3Dataset,
    variant: str,
    alpha: float,
    beta: float,
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
            logits, _probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=1.0,
                return_probe=True,
            )
            _ = logits
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


def _override_specs(variant: str, alpha: float, beta: float, gamma: float) -> Dict[str, OperatorSpec]:
    specs: Dict[str, OperatorSpec] = {}
    specs["L_only"] = OperatorSpec(alpha=alpha, beta=0.0, variant="no_injection")
    specs["U_antisym_only"] = OperatorSpec(alpha=0.0, beta=beta, variant="no_drift")
    specs["U_sym_only"] = OperatorSpec(alpha=0.0, beta=beta, gamma=gamma, variant="symmetric_control_v2_normmatched")

    if variant == "mechanism":
        specs["full"] = OperatorSpec(alpha=alpha, beta=beta, variant="mechanism")
    elif variant == "symmetric_control_v2_normmatched":
        specs["full"] = OperatorSpec(alpha=alpha, beta=beta, gamma=gamma, variant="symmetric_control_v2_normmatched")
    else:
        specs["full"] = OperatorSpec(alpha=alpha, beta=0.0, variant="no_injection")
    return specs


def _run_decomposition(
    ckpt_path: Path,
    eval_size: int,
    batch_size: int,
    k_tok: int,
    bootstrap: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    ckpt = _load_checkpoint(ckpt_path)
    cfg = V3Config(**ckpt["gen_cfg"])

    eval_size = eval_size if eval_size % 2 == 0 else eval_size + 1
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
    model.eval()

    variant = str(ckpt["variant"])
    alpha = float(ckpt.get("alpha", 0.0))
    beta = float(ckpt.get("beta", 0.0))

    gamma_stats = _compute_gamma_stats(model, dataset, variant, alpha, beta, batch_size)
    gamma = float(gamma_stats["gamma"])

    specs = _override_specs(variant, alpha, beta, gamma)
    overrides = {
        "L_only": {"spec": specs["L_only"], "zero_L": False},
        "U_antisym_only": {"spec": specs["U_antisym_only"], "zero_L": True},
        "U_sym_only": {"spec": specs["U_sym_only"], "zero_L": True},
        "full": {"spec": specs["full"], "zero_L": False},
    }

    per_pair: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in overrides}

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, attn_mask, _labels, metas = collate_batch(batch)
            logits, probe_logits, probe_U = model(
                input_ids,
                attn_mask=attn_mask,
                variant=variant,
                alpha=alpha,
                beta=beta,
                gamma=1.0,
                return_probe=True,
            )
            _ = logits
            L_np = probe_logits.cpu().numpy()
            U_np = probe_U.cpu().numpy()

            for idx, meta in enumerate(metas):
                L = L_np[idx].mean(axis=0)
                U = U_np[idx]
                L_zero = np.zeros_like(L, dtype=np.float64)
                prem_tokens = _prem_tokens(meta)
                cand_true, cand_false = _candidate_tokens(meta, cfg.T)

                for name, override in overrides.items():
                    L_use = L_zero if override["zero_L"] else L
                    val = token_adj_loc_gap(
                        L=L_use,
                        U=U,
                        spec=override["spec"],
                        k_tok=k_tok,
                        prem_tokens=prem_tokens,
                        cand_true_tokens=cand_true,
                        cand_false_tokens=cand_false,
                        run_id=meta.run_id,
                        instance_id=meta.instance_id,
                    )
                    per_pair[name].setdefault(meta.pair_id, {})[meta.topology] = float(val)

    rows: List[Dict[str, object]] = []
    summaries: Dict[str, Dict[str, float]] = {}

    for name in overrides:
        pairs = per_pair[name]
        diffs: List[float] = []
        obc_vals: List[float] = []
        pbc_vals: List[float] = []
        for pair_id, vals in pairs.items():
            if "OBC" not in vals or "PBC" not in vals:
                continue
            obc = vals["OBC"]
            pbc = vals["PBC"]
            obc_vals.append(obc)
            pbc_vals.append(pbc)
            diffs.append(obc - pbc)

        delta_mean = float(np.mean(diffs)) if diffs else 0.0
        delta_ci = _bootstrap_ci(diffs, n_boot=bootstrap)
        obc_mean = float(np.mean(obc_vals)) if obc_vals else 0.0
        pbc_mean = float(np.mean(pbc_vals)) if pbc_vals else 0.0

        summaries[name] = {
            "delta_mean": delta_mean,
            "delta_ci_low": float(delta_ci[0]),
            "delta_ci_high": float(delta_ci[1]),
            "obc_mean": obc_mean,
            "pbc_mean": pbc_mean,
            "n_pairs": len(diffs),
        }

        rows.append(
            {
                "variant": variant,
                "seed": int(ckpt["seed"]),
                "override": name,
                "delta_mean": delta_mean,
                "delta_ci_low": float(delta_ci[0]),
                "delta_ci_high": float(delta_ci[1]),
                "obc_mean": obc_mean,
                "pbc_mean": pbc_mean,
                "n_pairs": len(diffs),
                "eval_size": int(eval_size),
                "gamma": gamma_stats["gamma"],
                "zero_rate_B0": gamma_stats["zero_rate_B0"],
                "rho_median": gamma_stats["rho_median"],
                "scale_flag": gamma_stats["flag"],
            }
        )

    return rows, summaries


def _collect_ckpts(root: Path) -> List[Path]:
    ckpts = []
    for variant_dir in sorted((root / "train_conclusion").glob("*")):
        for seed_dir in sorted(variant_dir.glob("seed_*")):
            ckpt = seed_dir / "checkpoint_final.pt"
            if ckpt.exists():
                ckpts.append(ckpt)
    return ckpts


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "override",
                "delta_mean",
                "delta_ci_low",
                "delta_ci_high",
                "obc_mean",
                "pbc_mean",
                "n_pairs",
                "eval_size",
                "gamma",
                "zero_rate_B0",
                "rho_median",
                "scale_flag",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pooled_summary(rows: List[Dict[str, object]], bootstrap: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    pooled: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        key = f"{row['variant']}::{row['override']}"
        pooled.setdefault(key, []).append(float(row["delta_mean"]))

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, vals in pooled.items():
        variant, override = key.split("::", 1)
        ci = _bootstrap_ci(vals, n_boot=bootstrap)
        summary.setdefault(variant, {})[override] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "n": len(vals),
        }
    return summary


def _write_report(
    rows: List[Dict[str, object]],
    pooled: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> None:
    variants = sorted({row["variant"] for row in rows})
    report_lines = ["# V3 Effect Decomposition", ""]
    report_lines.append("## Per-seed ΔSelLoc (OBC - PBC)")
    for variant in variants:
        report_lines.append(f"### {variant}")
        for row in rows:
            if row["variant"] != variant:
                continue
            report_lines.append(
                f"- seed {row['seed']} {row['override']}: mean={row['delta_mean']:.6f} "
                f"CI=[{row['delta_ci_low']:.6f},{row['delta_ci_high']:.6f}] n_pairs={row['n_pairs']}"
            )
        report_lines.append("")

    report_lines.append("## Pooled summary by variant (mean over seeds)")
    for variant in variants:
        report_lines.append(f"### {variant}")
        overrides = pooled.get(variant, {})
        for name in sorted(overrides.keys()):
            entry = overrides[name]
            report_lines.append(
                f"- {name}: mean={entry['mean']:.6f} CI=[{entry['ci_low']:.6f},{entry['ci_high']:.6f}] n={entry['n']}"
            )
        report_lines.append("")

    report_lines.append("## Dominance check (mechanism variant)")
    mech = pooled.get("mechanism", {})
    if mech:
        ranked = sorted(mech.items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)
        if ranked:
            top_name, top_vals = ranked[0]
            report_lines.append(
                f"- largest |ΔSelLoc| override: {top_name} mean={top_vals['mean']:.6f} "
                f"CI=[{top_vals['ci_low']:.6f},{top_vals['ci_high']:.6f}]"
            )
    report_lines.append("- interpret with caution; override magnitudes may reflect baseline L asymmetry.")

    out_path.write_text("\n".join(report_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/v3_confirmatory")
    ap.add_argument("--eval_size", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_tok", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out_csv", type=str, default="analysis_v3_effect_decomposition.csv")
    ap.add_argument("--out_md", type=str, default="analysis_v3_effect_decomposition.md")
    args = ap.parse_args()

    root = Path(args.root)
    ckpts = _collect_ckpts(root)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {root}/train_conclusion")

    all_rows: List[Dict[str, object]] = []
    for ckpt_path in ckpts:
        run_dir = ckpt_path.parent
        eval_size = _load_eval_size(run_dir, args.eval_size)
        rows, _summaries = _run_decomposition(
            ckpt_path=ckpt_path,
            eval_size=eval_size,
            batch_size=args.batch_size,
            k_tok=args.k_tok,
            bootstrap=args.bootstrap,
        )
        all_rows.extend(rows)

    _write_csv(all_rows, Path(args.out_csv))
    pooled = _pooled_summary(all_rows, args.bootstrap)
    _write_report(all_rows, pooled, Path(args.out_md))

    meta = {
        "root": str(root),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    meta_path = Path(args.out_md).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
