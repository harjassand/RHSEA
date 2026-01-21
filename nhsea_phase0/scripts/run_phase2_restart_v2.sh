#!/usr/bin/env bash
set -euo pipefail

RUNROOT="runs/phase2_restart_v2"
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
TRAIN_CYCLE=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["train"]["cycle"]["size"])
PY
)
STEPS_FORWARD=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["train"]["forward"]["steps"])
PY
)
STEPS_CYCLE=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["train"]["cycle"]["steps"])
PY
)
EVAL_FORWARD=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["eval"]["forward"]["size"])
PY
)
EVAL_CYCLE=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["eval"]["cycle"]["size"])
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
K_PROP=$($PYTHON_BIN - <<'PY'
import json
cfg = json.load(open("phase2_lock.json"))
print(cfg["k_prop"])
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

TASKS=(forward cycle)
VARIANTS=(mechanism symmetric_control no_injection no_drift)

export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

for SEED in "${SEEDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
      case "$VARIANT" in
        mechanism|symmetric_control)
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

      if [[ "$TASK" == "forward" ]]; then
        TRAIN_SIZE=$TRAIN_FORWARD
        STEPS=$STEPS_FORWARD
        EVAL_SIZE=$EVAL_FORWARD
      else
        TRAIN_SIZE=$TRAIN_CYCLE
        STEPS=$STEPS_CYCLE
        EVAL_SIZE=$EVAL_CYCLE
      fi

      RUN_ID="phase2_v2_${TASK}_${VARIANT}_seed${SEED}"
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
        --task "$TASK" \
        --variant "$VARIANT" \
        --seed "$SEED" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --train_size "$TRAIN_SIZE" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --device "$DEVICE" \
        --save_every_steps 500 \
        --resume_if_exists \
        2>&1 | tee "$RUN_DIR/train_stdout.txt"

      $PYTHON_BIN scripts/eval.py \
        --checkpoint "$RUN_DIR/checkpoint_final.pt" \
        --eval_size "$EVAL_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --k_tok "$K_TOK" \
        --k_prop "$K_PROP" \
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
  done
done
