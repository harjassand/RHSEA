# Phase 3 Reciprocity Report

pooled_forward_mechanism=1.0000
pooled_forward_no_injection=1.0000
pooled_backward_mechanism=0.5057
pooled_backward_no_injection=0.5057
delta_backward(mech - baseline)=0.0000
falsification_threshold=-0.10
prediction_met=False
falsified=True

Per-seed backward accuracies:
- seed 0: mech=0.5160 baseline=0.5160 delta=0.0000
- seed 1: mech=0.5054 baseline=0.5054 delta=0.0000
- seed 2: mech=0.4956 baseline=0.4956 delta=0.0000

Chance check (Wilson 95% CI, backward):
- mechanism pooled CI [0.5000, 0.5113]; seeds 1/2 include 0.5; seed 0 excludes 0.5.
- no_injection pooled CI [0.5000, 0.5113]; seeds 1/2 include 0.5; seed 0 excludes 0.5.
Interpretation: backward remains effectively chance-level; no differential penalty between variants.

See Phase 3b reverse-direction test: `runs/phase3b_reverse/phase3b_reverse_report.md`.

Forward→Backward transfer fails for both baseline and mechanism.
Backward is learnable under supervision for both.
Phase 3b tests whether the failure is symmetric or orientation-specific.
