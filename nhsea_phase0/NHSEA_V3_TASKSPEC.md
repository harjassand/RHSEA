# NHSEA v3 Task Spec (Baseline-Only Gates)

## D1.1 Generator definition (OBC vs PBC)
Paired instance generation with matched surface statistics and different topology:

- OBC: directed chain terminates in a sink conclusion span. The conclusion is only reachable via directed edges from premises.
- PBC: same chain, but inject a cycle that removes sink behavior (the walk closes on itself).
- Surface stats matched: proposition lengths, positions, token types, candidate placements, and edge token counts.

Token layout (fixed length T):
- Position 0: TASK token (TASK_CONC for conclusion-choice; TASK_TOPO for topology task).
- Proposition spans pi: [T] -> [M] with M propositions, each span length in [L_min, L_max].
- Edge encoding: EDGE token followed by (P_src, P_dst) pairs, inserted after propositions.
- Query: QRY token followed by a premise ID or conclusion ID as specified below.
- Candidates: two candidate proposition IDs placed at fixed slots (positions T-2, T-1).
- Padding: PAD to length T.

Paired instance rule:
- Generate a base OBC chain of length K ending at sink C.
- Generate a PBC twin by inserting an edge from C to an earlier node to close a cycle.
- All non-topology tokens and placements are identical between the pair.

Determinism:
- All randomness derived from instance_seed_u32(run_id, instance_id, salt="GEN_V3").
- Instance record includes run_id, instance_id, and derived seed.

## D1.2 Labels
Two tasks with explicit labels:

1) Conclusion-choice task:
- Given premises and query, choose the correct conclusion among two candidate propositions.
- Label y_conc in {0,1}: index of the true candidate.

2) Topology-regime task:
- Predict whether the instance is OBC (0) or PBC (1).
- Label y_topo in {0,1}.

Evaluation uses separate heads or separate runs, but labels are defined in the data for both tasks.

## D1.3 Anti-leak gate
- Allowed features are invariant to candidate ordering: min, max, absdiff on length, position, and character count.
- Fail generator if AUROC > 0.55 on either task.

## D1.4 Baseline-only gates (must pass to proceed)
Gate A (in-task learnability):
- Baseline in-task accuracy above chance with CI not overlapping 0.5 for both tasks.

Gate B (transfer learnability):
- Baseline exceeds chance with n_train=512 in at least one direction under the adaptation protocol.

Gate C (topology observable contrast):
- OBC vs PBC shows non-trivial difference in at least one observable (PR or mass-at-conclusion) with CI excluding 0.

## D1.5 Budgets
Fixed parameters:
- Seeds: 0, 1, 2.
- T=64, M=8, K=4, L_min=3, L_max=6, vocab_size=200.

Fast smoke:
- Train size: 5,000; steps: 1,000; eval size: 2,000.
- Purpose: check basic learnability and leak gate.

Confirmatory:
- Train size: 50,000; steps: 10,000; eval size: 10,000.
- Few-shot adaptation: head-only, n_train in {32, 128, 512}, steps=2,000, eval_every=250.

No mechanism runs until all gates pass.
