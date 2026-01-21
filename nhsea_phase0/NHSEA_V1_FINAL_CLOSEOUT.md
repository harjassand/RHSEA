# NHSEA v1 Final Closeout

## A. Locked artifacts + provenance

NOTE: NHSEA_V1_FINAL_CLOSEOUT_LOCKED_R2 exists because NHSEA_HYPOTHESIS.md was added after the first lock; the original NHSEA_V1_FINAL_CLOSEOUT_LOCKED tag remains preserved at its prior commit.

Repo HEAD (at closeout tag): d3d6a50b1b73a377c7e6e4c8c30807caeacb2faa
Final closeout tag: NHSEA_V1_FINAL_CLOSEOUT_LOCKED
Tagged commit: d3d6a50b1b73a377c7e6e4c8c30807caeacb2faa

Tags (tag -> commit):
- NHSEA_v1_P2_ADDENDUM_DRAFT: 0b4cf145c6c5015ed043a265f887ba09ebc67df6
- NHSEA_v1_P2_ADDENDUM_V2_APPROVED: 0b4cf145c6c5015ed043a265f887ba09ebc67df6
- NHSEA_v1_PHASE3_CLOSEOUT: 0b4cf145c6c5015ed043a265f887ba09ebc67df6
- NHSEA_v1_PHASE3_CLOSEOUT_LOCKED: 92361edaca446980d931fcfa6a681193df5fa34b
- NHSEA_v1_PHASE3_SANITY_LOCK: 0b4cf145c6c5015ed043a265f887ba09ebc67df6
- NHSEA_v1_PHASE3_SANITY_LOCKED: 92361edaca446980d931fcfa6a681193df5fa34b
- NHSEA_V1_FINAL_CLOSEOUT_LOCKED: d3d6a50b1b73a377c7e6e4c8c30807caeacb2faa

git show-ref --tags | sort:
```
0a2324da09a61fc4da82469db4b6e67ae7911d32 refs/tags/NHSEA_v1_PHASE3_SANITY_LOCKED
0b4cf145c6c5015ed043a265f887ba09ebc67df6 refs/tags/NHSEA_v1_P2_ADDENDUM_DRAFT
0b4cf145c6c5015ed043a265f887ba09ebc67df6 refs/tags/NHSEA_v1_P2_ADDENDUM_V2_APPROVED
0b4cf145c6c5015ed043a265f887ba09ebc67df6 refs/tags/NHSEA_v1_PHASE3_CLOSEOUT
0b4cf145c6c5015ed043a265f887ba09ebc67df6 refs/tags/NHSEA_v1_PHASE3_SANITY_LOCK
a4b39db20e3c5df4600ec84805c27aee8f7d6f70 refs/tags/NHSEA_v1_PHASE3_CLOSEOUT_LOCKED
d3d6a50b1b73a377c7e6e4c8c30807caeacb2faa refs/tags/NHSEA_V1_FINAL_CLOSEOUT_LOCKED
```

Phase reports and CSVs:
- runs/phase2_claimA_v2/phase2_claimA_v2_report.md
- runs/phase2_claimA_v2/phase2_claimA_v2_master.csv
- runs/phase2_claimA_v2/phase2_claimA_v2_aggregate.csv
- runs/phase2_claimA_v2/phase2_claimA_v2_aggregate.json
- runs/phase2_claimA_v2/phase2_attribution_report.md
- runs/phase2_claimA_v2/phase2_attribution_master.csv
- runs/phase2_claimA_v2/phase2_eval_overrides_forward_0.json
- runs/phase2_claimA_v2/phase2_eval_overrides_forward_1.json
- runs/phase2_claimA_v2/phase2_eval_overrides_forward_2.json
- runs/phase2_claimB/phase2_claimB_report.md
- runs/phase2_claimB/phase2_claimB_master.csv
- runs/phase3/phase3_report.md
- runs/phase3/phase3_master.csv
- runs/phase3b_reverse/phase3b_reverse_report.md
- runs/phase3b_reverse/phase3b_reverse_master.csv
- phase3c_fewshot_report.md
- phase3c_fewshot_master.csv
- asymmetry_report.md
- asymmetry_master.csv
- PHASE2_CLOSEOUT.md
- NHSEA_PHASE3_TASKSPEC.md
- docs/NHSEA_v1_addendum_v2_scale_match.md
- pytest_after_phase3_sanity.txt
- tags_snapshot.txt

## B. What is supported vs not supported

- Claim A (selective localization): supported with v2 norm-matched symmetric control. SelLocGap pooled CI > 0 and rho flags false. However, attribution shows SelLocGap remains non-zero under beta=0 for seeds 0 and 1, so the effect is not uniquely attributable to antisymmetric injection.
- Claim B (cycle diagnostic + PR direction): not supported. Pooled DeltaCyc and PR differences are near zero with CIs spanning 0 under the locked generator/training regime.
- Reciprocity: non-discriminative under current generator/protocol. Forward-to-backward and backward-to-forward transfer is at chance for both baseline and mechanism; no differential penalty is observed.

## C. Mechanism attribution summary (numbers)

SelLocGap overrides per seed (means with 95% CI):

| seed | SelLocGap baseline mean [CI] | beta=0 mean [CI] | alpha=0 mean [CI] | beta flip mean [CI] |
| --- | --- | --- | --- | --- |
| 0 | 0.004465 [0.003639, 0.005331] | 0.002570 [0.001725, 0.003437] | 0.004569 [0.003729, 0.005442] | 0.002239 [0.001407, 0.003096] |
| 1 | 0.008414 [0.008043, 0.008784] | 0.006556 [0.006161, 0.006958] | 0.007983 [0.007627, 0.008341] | 0.005924 [0.005599, 0.006245] |
| 2 | 0.001402 [0.000974, 0.001831] | 0.000631 [0.000326, 0.000941] | 0.000969 [0.000521, 0.001416] | 0.000108 [-0.000150, 0.000365] |

Attribution deltas (means with 95% CI):

| seed | Delta_infer_mech mean [CI] | Delta_learn mean [CI] |
| --- | --- | --- |
| 0 | 0.001895 [0.001666, 0.002123] | 0.009424 [0.008436, 0.010411] |
| 1 | 0.001858 [0.001620, 0.002093] | 0.000092 [-0.000316, 0.000503] |
| 2 | 0.000771 [0.000562, 0.000982] | -0.000111 [-0.000376, 0.000069] |

## C2. Asymmetry summary (baseline L directionality)

- L already directional: yes; baseline asym_L median=0.8796.
- Baseline asym_L range: min=0.5439, max=1.0255.
- Mechanism asym_L median=0.8209 (min=0.6372, max=1.0158).
- Symmetric_control_v2 asym_L median=0.8690 (min=0.4741, max=1.0698).
- Directionality persists across variants; L is strongly non-symmetric without injection.

## D. Stop reasons

- Cycle diagnostics do not separate regimes reliably: DeltaCyc and PR differences are near 0 with CIs spanning 0 across pipelines and variants.
- Reciprocity tests are non-discriminative: transfer (forward->backward and backward->forward) remains at chance for both mechanism and baseline, and few-shot head adaptation does not reliably exceed chance at n_train=512.

## E. Actionable next step

Proceed only under NHSEA v2 task redesign with transfer-learnable tasks that make reciprocity discriminative. No further NHSEA v1 sweeps are recommended.

## F. Final gates

pytest -q output:
```
.........................                                                [100%]
25 passed in 2.37s
```

python -m py_compile $(git ls-files '*.py'):
```
(no output)
```
