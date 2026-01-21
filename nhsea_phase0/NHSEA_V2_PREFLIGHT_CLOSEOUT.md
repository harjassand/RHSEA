# NHSEA v2 Preflight Closeout

## Locked provenance
- Preflight code commit: 5628089d54783a449426a46346cd02a66120291c
- NHSEA_V2_SPEC_LOCKED tag: 1aa767f5217738a471ea5e3970a7413e6f196f58

## Commands used (exact)
```bash
python scripts/leak_gate_v2.py --task forward --report runs/v2/leak_gate_v2_forward.json --features_out runs/v2/leak_gate_v2_forward_features.json
python scripts/leak_gate_v2.py --task backward --report runs/v2/leak_gate_v2_backward.json --features_out runs/v2/leak_gate_v2_backward_features.json
export PYTORCH_ENABLE_MPS_FALLBACK=1
for SEED in 0 1 2; do python scripts/train_v2.py --task forward --variant no_injection --seed "$SEED" --device mps --run_root runs/v2; done
for SEED in 0 1 2; do CKPT="runs/v2/forward_train/no_injection/seed_${SEED}/checkpoint_final.pt"; python scripts/eval_v2.py --checkpoint "$CKPT" --eval_task forward --device mps; python scripts/eval_v2.py --checkpoint "$CKPT" --eval_task backward --device mps; done
for SEED in 0 1 2; do python scripts/train_v2.py --task backward --variant no_injection --seed "$SEED" --device mps --run_root runs/v2; done
for SEED in 0 1 2; do CKPT="runs/v2/backward_train/no_injection/seed_${SEED}/checkpoint_final.pt"; python scripts/eval_v2.py --checkpoint "$CKPT" --eval_task backward --device mps; python scripts/eval_v2.py --checkpoint "$CKPT" --eval_task forward --device mps; done
for SEED in 0 1 2; do for SOURCE in forward backward; do for TARGET in forward backward; do if [ "$SOURCE" = "$TARGET" ]; then continue; fi; for N in 32 128 512; do python scripts/v2_fewshot_adapt.py --run_root runs/v2 --out_dir runs/v2 --device mps --seed "$SEED" --source_task "$SOURCE" --target_task "$TARGET" --n_train "$N"; done; done; done; done
python scripts/v2_preflight_aggregate.py --root runs/v2 --out runs/v2/v2_preflight_master.csv --report runs/v2/v2_preflight_report.md
```

## Preflight report (full table)
```text
# V2 Preflight Report

## Leak gate
- forward: AUROC=0.5000 pass=True
- backward: AUROC=0.5000 pass=True

## In-task accuracy
- forward: acc=0.7671 CI=[0.7622,0.7718] n=30000
- backward: acc=0.7674 CI=[0.7626,0.7721] n=30000

## Zero-shot transfer
- forward->backward: acc=0.4967 CI=[0.4910,0.5023] n=30000
- backward->forward: acc=0.5007 CI=[0.4950,0.5063] n=30000

## Few-shot transfer (head-only)
- forward->backward n=32: acc=0.5010 CI=[0.4953,0.5067] n=30000
- backward->forward n=32: acc=0.5010 CI=[0.4953,0.5067] n=30000
- forward->backward n=128: acc=0.4953 CI=[0.4896,0.5009] n=30000
- backward->forward n=128: acc=0.4996 CI=[0.4940,0.5053] n=30000
- forward->backward n=512: acc=0.4987 CI=[0.4930,0.5043] n=30000
- backward->forward n=512: acc=0.4985 CI=[0.4928,0.5041] n=30000

## Gate status
- leak_gate_pass=True
- in_task_pass=False
- transfer_pass=False
- overall_pass=False
```

## Status
overall_pass=False; no mechanism runs executed.
