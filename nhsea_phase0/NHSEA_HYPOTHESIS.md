# Non-Hermitian Skin-Effect Automata (NHSEA): Tested Hypothesis (v1)

## Core hypothesis (verbatim, formalized)
Attention/propagation is fundamentally non-reciprocal; reasoning corresponds to boundary localization under non-Hermitian operators; and this directionality should induce asymmetric generalization (forward != backward).

## Operational predictions (must match gates)
- P1: Premise-seeded propagation localizes conclusions more than matched controls (SelLocGap > 0).
- P2: Cyclic structure induces delocalization (PR, DeltaCyc signals).
- P3 (critical): Forward->backward generalization is catastrophically worse than baseline (reciprocity test).

## Falsification criteria (explicit)
Hypothesis is falsified if:
- Baseline and mechanism show equivalent backward performance under matched conditions.
- Directional gap in transfer is indistinguishable from 0 within CI.
- Effects collapse once task ambiguity is controlled.

## What was not claimed
- No scaling-law violation claim.
- No claim of universal superiority.
- No claim that non-Hermiticity alone induces reasoning without task support.

## Mapping to experimental phases
- Phase 2 -> tests P1, P2.
- Phase 3 -> tests P3 (reciprocity).
- Status: P1 supported under v2 control but not uniquely attributable; P2 not supported; P3 not testable (transfer at chance).

## Status
Status under v1/v2 experimental family: **Not supported / not testable** (see closeout tags).
