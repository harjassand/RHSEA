# NHSEA v2 Task Spec (Draft, Option V2-A)

## 1. Scope and hypothesis
Goal: test a discriminative reciprocity prediction on a transfer-learnable task where forward and backward share an invertible latent mapping.

## 2. Task family (Option V2-A: invertible mapping)
### Latent structure
- Sample a bijection f: A -> B over N symbols (random permutation per instance).
- Forward task: given context facts and query A_i, predict B_{f(i)}.
- Backward task: given the same context facts and query B_j, predict A_{f^{-1}(j)}.

### Tokens and vocab
- Special tokens: PAD, SEP, QRY, TASK_FWD, TASK_BWD.
- Symbol tokens: A00..A(N-1), B00..B(N-1).
- Filler tokens: w0..w(V-1) for neutral padding/fillers.

### Instance layout (fixed length T)
- Start token: TASK_FWD or TASK_BWD in position 0.
- Context: K facts, each encoded as "Axx SEP Byy" (bidirectional facts in surface form).
- Query: "QRY Axx" (forward) or "QRY Byy" (backward), drawn from one of the K facts.
- Candidates: two candidate spans, each a single symbol token, placed at fixed slots (positions T-2 and T-1).
- Padding: PAD to length T, with PAD filling any gap between query and candidates.

### Candidate construction (anti-leak)
- True candidate is the correct mapping for the query under f.
- Decoy candidate is sampled from the same symbol set, uniformly among incorrect mappings.
- Match candidate span length (1 token) and position distribution.
- Ensure identical token-type distributions across candidates (A vs B depending on task).
 - Candidate order is randomized per instance; label is the true candidate index (0 or 1).

### Determinism and seeding (locked)
- All randomness is derived from instance_seed_u32(run_id, instance_id, salt="GEN_V2").
- The instance record must include run_id, instance_id, and the derived seed used.

### Generator parameters (locked)
- N (symbols per side): 16
- K (facts per instance): 8
- T (sequence length): 64
- V (filler vocab size): 200
- Seeds: 0, 1, 2

## 3. Labels
- Binary label: 0 if candidate 0 is correct, 1 if candidate 1 is correct.
- Output head: 2-way classification.

## 4. Model/training config (locked)
- Model: TinyTransformer(d_model=128, n_heads=4, d_ff=512, n_layers=4, dropout=0.0).
- Probe layer: 2 (matches v1 instrumentation).
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01.
- Batch size: 64.
- Train size: 50,000; steps: 10,000.
- Eval size: 10,000.

## 5. Metrics (locked)
- Accuracy (in-task, zero-shot transfer, and few-shot adaptation).
- SelLocGap using v1 definition on attention weights.
- Report transfer deltas: forward->backward and backward->forward.

## 6. Leak-gate invariants (locked)
- Candidate length and position symmetry across true/decoy.
- Token identity symmetry: decoy sampled from same symbol set (A or B) as true.
- Use leak_gate_v2 with permutation-invariant features only.
- Threshold: AUROC <= 0.55 (95% CI upper bound <= 0.60) required to pass.

## 7. Transfer protocol (locked)
- Train forward-only baseline; eval forward + backward.
- Train backward-only baseline; eval backward + forward.
- Zero-shot transfer: no adaptation.
- Few-shot adaptation: head-only, freeze encoder, n_train in {32, 128, 512}, steps=2000, eval_every=250.

## 8. Discriminative gate (must pass before NHSEA mechanism)
- Baseline in-task accuracy >= 0.95 for both tasks when trained directly.
- Baseline transfer >= 0.60 zero-shot OR >= 0.70 with head-only n_train=512.
- If gate fails, generator is rejected and redesigned; no mechanism runs.

## 9. Preflight plan (baseline only, no mechanism)
### Expected outputs
- v2_preflight_master.csv (per seed/task/transfer with accuracy and CI)
- v2_preflight_report.md (summary + gate status)

### Command plan (to run after spec is tagged)
- Leak gate:
  - python scripts/leak_gate_v2.py --task forward --report runs/v2/leak_gate_forward.json
  - python scripts/leak_gate_v2.py --task backward --report runs/v2/leak_gate_backward.json
- Train/eval forward:
  - python scripts/train_v2.py --task forward --variant no_injection --seed 0,1,2
  - python scripts/eval_v2.py --checkpoint <ckpt> --eval_task forward
  - python scripts/eval_v2.py --checkpoint <ckpt> --eval_task backward
- Train/eval backward:
  - python scripts/train_v2.py --task backward --variant no_injection --seed 0,1,2
  - python scripts/eval_v2.py --checkpoint <ckpt> --eval_task backward
  - python scripts/eval_v2.py --checkpoint <ckpt> --eval_task forward
- Few-shot head-only adaptation:
  - python scripts/v2_fewshot_adapt.py --source_task forward --target_task backward --n_train 32,128,512
  - python scripts/v2_fewshot_adapt.py --source_task backward --target_task forward --n_train 32,128,512
- Aggregate:
  - python scripts/v2_preflight_aggregate.py --root runs/v2 --out v2_preflight_master.csv --report v2_preflight_report.md
