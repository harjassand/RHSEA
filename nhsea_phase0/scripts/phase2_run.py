#!/usr/bin/env python
"""Phase 2 orchestrator: gates, training, evaluation, and aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    lock_path = Path("phase2_lock.json")
    cfg = json.loads(lock_path.read_text())

    seeds = cfg["seeds"]
    variants = ["mechanism", "symmetric_control", "no_injection", "no_drift"]
    tasks = ["forward", "cycle"]

    # Gates
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "scripts/diagnostic_stress.py"])

    leak_dir = Path("runs/phase2/leak_gate")
    leak_dir.mkdir(parents=True, exist_ok=True)
    leak_json = leak_dir / "leak_gate_report.json"
    run(
        [
            sys.executable,
            "scripts/leak_gate.py",
            "--n",
            "4000",
            "--seed",
            "0",
            "--run_id",
            "phase2",
            "--config",
            str(lock_path),
            "--report",
            str(leak_json),
        ]
    )
    leak_data = json.loads(leak_json.read_text())
    leak_txt = leak_dir / "leak_gate_report.txt"
    leak_txt.write_text(json.dumps(leak_data, indent=2, sort_keys=True) + "\n")
    if not leak_data.get("passed", False):
        raise SystemExit("Leak gate failed; aborting Phase 2.")

    # Runs
    for seed in seeds:
        for task in tasks:
            for variant in variants:
                alpha, beta = cfg["alpha_beta"][variant]
                train_size = cfg["train"][task]["size"]
                steps = cfg["train"][task]["steps"]
                eval_size = cfg["eval"][task]["size"]
                batch_size = cfg["batch_size"]
                lr = cfg["optimizer"]["lr"]
                weight_decay = cfg["optimizer"]["weight_decay"]
                k_tok = cfg["k_tok"]
                k_prop = cfg["k_prop"]

                run_dir = Path("runs/phase2") / task / variant / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                run(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--task",
                        task,
                        "--variant",
                        variant,
                        "--alpha",
                        str(alpha),
                        "--beta",
                        str(beta),
                        "--seed",
                        str(seed),
                        "--train_size",
                        str(train_size),
                        "--steps",
                        str(steps),
                        "--batch_size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--weight_decay",
                        str(weight_decay),
                        "--run_dir",
                        str(run_dir),
                    ]
                )

                run(
                    [
                        sys.executable,
                        "scripts/eval.py",
                        "--checkpoint",
                        str(run_dir / "checkpoint.pt"),
                        "--eval_size",
                        str(eval_size),
                        "--batch_size",
                        str(batch_size),
                        "--k_tok",
                        str(k_tok),
                        "--k_prop",
                        str(k_prop),
                        "--bootstrap",
                        "10000",
                        "--out_dir",
                        str(run_dir),
                    ]
                )

            # Paired SelLocGap for forward task
            if task == "forward":
                mech_ckpt = Path("runs/phase2/forward/mechanism") / f"seed_{seed}" / "checkpoint.pt"
                sym_ckpt = Path("runs/phase2/forward/symmetric_control") / f"seed_{seed}" / "checkpoint.pt"
                paired_dir = Path("runs/phase2/forward/paired_mech_minus_sym") / f"seed_{seed}"
                paired_dir.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        sys.executable,
                        "scripts/eval_forward_paired.py",
                        "--mech_ckpt",
                        str(mech_ckpt),
                        "--sym_ckpt",
                        str(sym_ckpt),
                        "--eval_size",
                        str(cfg["eval"]["forward"]["size"]),
                        "--batch_size",
                        str(cfg["batch_size"]),
                        "--k_tok",
                        str(cfg["k_tok"]),
                        "--bootstrap",
                        "10000",
                        "--out_dir",
                        str(paired_dir),
                    ]
                )

    run([sys.executable, "scripts/phase2_aggregate.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
