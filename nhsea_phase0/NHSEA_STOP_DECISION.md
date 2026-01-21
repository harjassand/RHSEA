# NHSEA Stop Decision

## Hypothesis target prediction
Forward/backward generalization should be asymmetric due to irreversibility and boundary localization (skin-effect), producing a directional transfer penalty.

## What we observed
- In-task learnable, transfer at chance in both directions (see `runs/v2_preflight/v2_preflight_report.md` and `phase3c_fewshot_report.md`).
- Baseline L is already directional (see `asymmetry_report.md`).
- Cycle diagnostics are weak/null under the locked generator (see `runs/phase2_claimB/phase2_claimB_report.md`).

## Conclusion
Under this experimental family, the NHSEA hypothesis is **not supported**.

## Compute decision
No further NHSEA mechanism runs are authorized under v2; only redesign proposals with preflight proof of discriminativity will be considered.
