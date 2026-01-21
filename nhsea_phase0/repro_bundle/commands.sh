#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-mps}"
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

# Phase 2 Claim A v2 (forward task)
bash scripts/run_phase2_claimA_v2.sh

# Eval overrides (beta=0, alpha=0, beta flip) per seed
for SEED in 0 1 2; do
  $PYTHON_BIN scripts/eval_forward_overrides.py \
    --mech_ckpt "runs/phase2_claimA_v2/phase2_v2_forward_mechanism_seed${SEED}/checkpoint_final.pt" \
    --sym_ckpt "runs/phase2_claimA_v2/phase2_v2_forward_symmetric_control_v2_normmatched_seed${SEED}/checkpoint_final.pt" \
    --eval_size 10000 \
    --batch_size 64 \
    --k_tok 16 \
    --bootstrap 10000 \
    --device "$DEVICE" \
    --out "runs/phase2_claimA_v2/phase2_eval_overrides_forward_${SEED}.json"
done

# Attribution pass
$PYTHON_BIN scripts/eval_forward_attribution.py \
  --run_root runs/phase2_claimA_v2 \
  --seeds 0,1,2 \
  --eval_n 10000 \
  --batch_size 64 \
  --k_tok 16 \
  --bootstrap 10000 \
  --device "$DEVICE"

# Phase 2 Claim B (cycle task)
bash scripts/run_phase2_claimB.sh
$PYTHON_BIN scripts/claimB_report.py --root runs/phase2_claimB

# Phase 3 (reciprocity)
$PYTHON_BIN scripts/leak_gate_phase3.py --task forward --report runs/phase3/leak_gate_forward.json --features_out phase3_leak_features_forward.json
$PYTHON_BIN scripts/leak_gate_phase3.py --task backward --report runs/phase3/leak_gate_backward.json --features_out phase3_leak_features_backward.json
$PYTHON_BIN scripts/phase3_matrix.py
$PYTHON_BIN scripts/phase3_reciprocity.py
$PYTHON_BIN scripts/phase3_chance_check.py
$PYTHON_BIN scripts/phase3_learnability.py
$PYTHON_BIN scripts/phase3_multitask_runner.py

# Phase 3b (reverse-direction transfer)
$PYTHON_BIN scripts/phase3b_reverse.py

# Phase 3c (few-shot head adaptation)
OUT_DIR="runs/phase3c_fewshot"
for VARIANT in no_injection mechanism; do
  for SEED in 0 1 2; do
    for SOURCE in forward backward; do
      for TARGET in forward backward; do
        if [[ "$SOURCE" == "$TARGET" ]]; then
          continue
        fi
        if [[ "$SOURCE" == "forward" ]]; then
          RUN_ROOT="runs/phase3"
        else
          RUN_ROOT="runs/phase3b_reverse"
        fi
        for N in 32 128 512 2048; do
          $PYTHON_BIN scripts/phase3c_fewshot_adapt.py \
            --run_root "$RUN_ROOT" \
            --out_dir "$OUT_DIR" \
            --device "$DEVICE" \
            --variant "$VARIANT" \
            --seed "$SEED" \
            --source_task "$SOURCE" \
            --target_task "$TARGET" \
            --tier head \
            --n_train "$N" \
            --steps 2000 \
            --eval_every 250
        done
      done
    done
  done
done

$PYTHON_BIN scripts/phase3c_fewshot_aggregate.py \
  --root "$OUT_DIR" \
  --out phase3c_fewshot_master.csv \
  --report phase3c_fewshot_report.md
