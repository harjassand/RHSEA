# NHSEA Executive Summary (One Page)

## Hypothesis (NHSEA)
Non-reciprocal directed operator induces conclusion-boundary localization (skin-like accumulation) in linear chains.
Cycles suppress localization and increase delocalization/uncertainty (PR increases) and are detectable via DeltaCyc^(ell).
Reciprocity test: forward-trained model should fail abductive/backward inference relative to baseline.

## What was implemented and locked
- v1 prereg phases 0-3 and v2 baseline preflight.
- Determinism rules, clamp, TopK, RowNorm, PermRow, and SelLocGap definition.
- v2 symmetric control addendum (norm-matched γ) to equalize injection scale in controls.

## Key results
- Baseline L is already directional (`asymmetry_report.md`, `asymmetry_master.csv`).
- Phase 3 reciprocity tests show forward->backward and backward->forward transfer at chance; no directional gap (`phase3_report.md`, `phase3b_reverse_report.md`).
- Cycle regime separation weak or null (`runs/phase2_claimB/phase2_claimB_report.md`).
- v2 preflight baseline passes leak gate, learns in-task, but transfer remains at chance; overall gate fails (`runs/v2_preflight/v2_preflight_report.md`).
- Solvability audit shows ambiguous_rate=0.0 and ceiling_acc=1.0 (`runs/v2_preflight/solvability_audit.json`, `runs/v2_preflight/solvability_audit.md`).

## Conclusion / stop decision
Under the implemented benchmark families and gates, the evidence does not support the NHSEA mechanism claims; compute is stopped under `NHSEA_ALL_STOP_LOCKED`.
This is not a metaphysical statement; it is an empirical outcome under these benchmark designs.

## Next step
Only acceptable next step is a v3 benchmark that explicitly creates an OBC vs PBC topology contrast and passes baseline discriminativity + transfer-learnability gates before any mechanism runs.
