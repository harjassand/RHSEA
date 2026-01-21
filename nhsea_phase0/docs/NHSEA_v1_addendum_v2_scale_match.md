# NHSEA v1 Addendum v2: Symmetric Control Scale Matching (Run-Level)

Status: APPROVED (implemented)
Date: 2026-01-20
Git commit: d483d4458059b6f4dd44441e3f751ae75c70118d
Run command: bash scripts/run_phase2_claimA_v2.sh

This amendment only changes the symmetric-control scaling. All other prereg definitions are unchanged.

## Motivation
Phase 2 shows symmetric-control scale mismatch (rho_tilde outside [0.9, 1.1]) after training despite preflight parity.
This makes Claim A untestable under the existing control. The mechanism definition remains unchanged; only the
symmetric-control scale is adjusted to restore matchability.

## Amendment (control-only change; run-level scale match)
Mechanism remains:
S = U - U^T
O_raw = L + alpha D + beta S

Control is replaced with a norm-matched symmetric surrogate with a run-level scalar.

For a fixed trained model checkpoint theta and fixed eval set D_eval (the same used for Claim A evaluation), define:

- a(x) = || beta * (U(x) - U(x)^T) ||_F
- b0(x) = || beta * (U(x) + U(x)^T) ||_F

Let:

- A_tilde = median_{x in D_eval} a(x)
- B0_tilde = median_{x in D_eval} b0(x)

Define the run-level scalar:

- gamma = 1 if B0_tilde = 0 else gamma = A_tilde / B0_tilde

Then the symmetric-control injection is:

- S_sym(x) = gamma * (U(x) + U(x)^T)
- O_raw = L + alpha D + beta * S_sym(x)

The rescaling is applied only to the symmetric-control term, and only before the clamp; all other definitions are unchanged.

## Updated gate criteria (control validity)
For the v2 norm-matched control, the validity gate is:

- B0_tilde > 0, and
- zero_rate_B0 <= 0.1% where zero_rate_B0 is the fraction of x in D_eval with b0(x) = 0.

Report rho_tilde for auditing (it should be approximately 1), but do not use it as a validity gate for the v2 control
since gamma enforces median norm match by construction.

## Scope and invariants
This addendum does not change:
- Clamp sentinel handling
- TopKOffDiag / RowNorm+ / DiagZero
- PermRow null definition
- Propagation definition
- SelLocGap definition (prem minus rand; mech minus sym)

This addendum only modifies the symmetric-control injection scale to restore a testable matched control.
