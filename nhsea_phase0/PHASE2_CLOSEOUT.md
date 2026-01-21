# Phase 2 Closeout (v2 control)

## Claim A (v2 control) — confirmatory result
- pooled SelLocGap mean 0.004760 CI [0.004412, 0.005104]
- seed 0 SelLocGap mean 0.004465 CI [0.003639, 0.005331] (rho_flag False, rho_median 1.002972)
- seed 1 SelLocGap mean 0.008414 CI [0.008043, 0.008784] (rho_flag False, rho_median 1.013125)
- seed 2 SelLocGap mean 0.001402 CI [0.000974, 0.001831] (rho_flag False, rho_median 0.998653)

## A2 eval overrides (mech eval-time ablations)
All overrides use the same eval instances and the same control; only the mechanism spec is changed at eval.

Seed 0:
- baseline SelLocGap mean 0.004465 CI [0.003639, 0.005331]
- beta=0 SelLocGap mean 0.002570 CI [0.001725, 0.003437]
- alpha=0 SelLocGap mean 0.004569 CI [0.003729, 0.005442]
- beta sign flip SelLocGap mean 0.002239 CI [0.001407, 0.003096]

Seed 1:
- baseline SelLocGap mean 0.008414 CI [0.008043, 0.008784]
- beta=0 SelLocGap mean 0.006556 CI [0.006161, 0.006958]
- alpha=0 SelLocGap mean 0.007983 CI [0.007627, 0.008341]
- beta sign flip SelLocGap mean 0.005924 CI [0.005599, 0.006245]

Seed 2:
- baseline SelLocGap mean 0.001402 CI [0.000974, 0.001831]
- beta=0 SelLocGap mean 0.000631 CI [0.000326, 0.000941]
- alpha=0 SelLocGap mean 0.000969 CI [0.000521, 0.001416]
- beta sign flip SelLocGap mean 0.000108 CI [-0.000150, 0.000365]

Interpretation note: beta=0 reduces SelLocGap but does not collapse to ~0 for seeds 0/1; this needs review under the stated stop condition.

## A2 attribution pass (decomposition)
- pooled Delta_infer_mech mean 0.001508 CI [0.001375, 0.001635]
- pooled Delta_infer_sym mean -0.001556 CI [-0.001974, -0.001149]
- pooled Delta_learn mean 0.003135 CI [0.002759, 0.003505]

## Control validity
- rho_flag False for all seeds (zero_rate_B0 = 0.0)

## Claim B (cycle diagnostics)
Claim B is not supported under the locked generator/training regime.
Pooled ΔCyc and PR differences are near 0 with CIs spanning 0 across pipelines.

Postmortem hypotheses (no new experiments):
- TopK may prune cyclic edges, suppressing signal after collapse.
- PR saturates near maximum with current M/k_prop, reducing sensitivity.
- Regime classifier accuracy near chance indicates generator/training mismatch.

## Phase 3 (reciprocity test)
- Forward-trained models show no differential backward penalty: mech == baseline at chance (Δ=0.0000).
- This falsifies the preregistered Phase 3 differential-penalty threshold (−0.10) but does not establish backward competence.
- Chance check: seed-level Wilson CIs include 0.5 for seeds 1/2; pooled CI is only marginally above 0.5.

## Phase 3b (reverse reciprocity)
- Backward-trained models also fail to transfer to forward: mech == baseline at chance (Δ=0.0000).
- reverse_gap_present=False under the exploratory −0.05 threshold; no orientation-specific transfer gap observed.

## Artifacts
- addendum: `docs/NHSEA_v1_addendum_v2_scale_match.md`
- claim A report: `runs/phase2_claimA_v2/phase2_claimA_v2_report.md`
- claim A master CSV: `runs/phase2_claimA_v2/phase2_claimA_v2_master.csv`
- paired evals: `runs/phase2_claimA_v2/paired_mech_minus_sym/seed_0/summary.json`, `runs/phase2_claimA_v2/paired_mech_minus_sym/seed_1/summary.json`, `runs/phase2_claimA_v2/paired_mech_minus_sym/seed_2/summary.json`
- overrides: `runs/phase2_claimA_v2/phase2_eval_overrides_forward_0.json`, `runs/phase2_claimA_v2/phase2_eval_overrides_forward_1.json`, `runs/phase2_claimA_v2/phase2_eval_overrides_forward_2.json`
- attribution: `runs/phase2_claimA_v2/phase2_attribution_seed0.json`, `runs/phase2_claimA_v2/phase2_attribution_seed1.json`, `runs/phase2_claimA_v2/phase2_attribution_seed2.json`
- attribution report: `runs/phase2_claimA_v2/phase2_attribution_report.md`
- u symmetry drift: `runs/phase2_claimA_v2/u_symmetry_drift.md`, `runs/phase2_claimA_v2/u_symmetry_drift.csv`
- Claim B report: `runs/phase2_claimB/phase2_claimB_report.md`
- Claim B master CSV: `runs/phase2_claimB/phase2_claimB_master.csv`
- Phase 3 report: `runs/phase3/phase3_report.md`
- Phase 3 master CSV: `runs/phase3/phase3_master.csv`
- Phase 3 chance check: `runs/phase3/phase3_chance_check.json`
- Phase 3 learnability report: `runs/phase3_learnability/phase3_learnability_report.md`
- Phase 3 multitask report: `runs/phase3_multitask/phase3_multitask_report.md`

## Tags
- NHSEA_v1_P2_ADDENDUM_V2_APPROVED
- NHSEA_v1_PHASE3_CLOSEOUT (d483d4458059b6f4dd44441e3f751ae75c70118d)
- NHSEA_v1_PHASE3_SANITY_LOCK (d483d4458059b6f4dd44441e3f751ae75c70118d)
