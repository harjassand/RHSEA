#!/usr/bin/env bash
set -euo pipefail

RUNROOT="runs/phase2_claimA_v2"
CFG="phase2_lock.json"
DEVICE="${DEVICE:-mps}"

mkdir -p "$RUNROOT"

VENV=".venv"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

SEEDS=($($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(" ".join(str(s) for s in cfg["seeds"]))
PY
))

TRAIN_FORWARD=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["train"]["forward"]["size"])
PY
)
STEPS_FORWARD=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["train"]["forward"]["steps"])
PY
)
EVAL_FORWARD=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["eval"]["forward"]["size"])
PY
)
BATCH_SIZE=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["batch_size"])
PY
)
K_TOK=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["k_tok"])
PY
)
LR=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["optimizer"]["lr"])
PY
)
WEIGHT_DECAY=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["optimizer"]["weight_decay"])
PY
)

VARIANTS=(mechanism symmetric_control_v2_normmatched no_injection no_drift)

export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

for SEED in "${SEEDS[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    case "$VARIANT" in
      mechanism|symmetric_control_v2_normmatched)
        ALPHA=0.5
        BETA=1.0
        ;;
      no_injection)
        ALPHA=0.5
        BETA=0.0
        ;;
      no_drift)
        ALPHA=0.0
        BETA=1.0
        ;;
    esac

    RUN_ID="phase2_v2_forward_${VARIANT}_seed${SEED}"
    RUN_DIR="${RUNROOT}/${RUN_ID}"
    DONE_FILE="${RUN_DIR}/DONE"
    if [[ -f "$DONE_FILE" ]]; then
      echo "Skipping ${RUN_ID} (DONE exists)"
      continue
    fi

    mkdir -p "$RUN_DIR"

    $PYTHON_BIN scripts/train.py \
      --run_root "$RUNROOT" \
      --run_id "$RUN_ID" \
      --task forward \
      --variant "$VARIANT" \
      --seed "$SEED" \
      --alpha "$ALPHA" \
      --beta "$BETA" \
      --train_size "$TRAIN_FORWARD" \
      --steps "$STEPS_FORWARD" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --device "$DEVICE" \
      --save_every_steps 500 \
      --resume_if_exists \
      2>&1 | tee "$RUN_DIR/train_stdout.txt"

    $PYTHON_BIN scripts/eval.py \
      --checkpoint "$RUN_DIR/checkpoint_final.pt" \
      --eval_size "$EVAL_FORWARD" \
      --batch_size "$BATCH_SIZE" \
      --k_tok "$K_TOK" \
      --k_prop 4 \
      --bootstrap 10000 \
      --device "$DEVICE" \
      --out_dir "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/eval_stdout.txt"

    if [[ -f "$RUN_DIR/eval_summary.json" ]]; then
      touch "$DONE_FILE"
    else
      echo "Missing eval_summary.json for ${RUN_ID}"
      exit 1
    fi
  done

  PAIR_DIR="${RUNROOT}/paired_mech_minus_sym/seed_${SEED}"
  PAIR_DONE="${PAIR_DIR}/DONE"
  if [[ -f "$PAIR_DONE" ]]; then
    echo "Skipping paired eval for seed ${SEED} (DONE exists)"
    continue
  fi

  MECH_CKPT="${RUNROOT}/phase2_v2_forward_mechanism_seed${SEED}/checkpoint_final.pt"
  SYM_CKPT="${RUNROOT}/phase2_v2_forward_symmetric_control_v2_normmatched_seed${SEED}/checkpoint_final.pt"
  mkdir -p "$PAIR_DIR"
  $PYTHON_BIN scripts/eval_forward_paired.py \
    --mech_ckpt "$MECH_CKPT" \
    --sym_ckpt "$SYM_CKPT" \
    --eval_size "$EVAL_FORWARD" \
    --batch_size "$BATCH_SIZE" \
    --k_tok "$K_TOK" \
    --bootstrap 10000 \
    --out_dir "$PAIR_DIR" \
    2>&1 | tee "$PAIR_DIR/eval_stdout.txt"

  if [[ -f "$PAIR_DIR/summary.json" ]]; then
    touch "$PAIR_DONE"
  else
    echo "Missing paired summary.json for seed ${SEED}"
    exit 1
  fi
done
