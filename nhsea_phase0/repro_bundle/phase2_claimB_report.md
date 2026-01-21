# Phase 2 Claim B Report

## Variant: symmetric_control_v2_normmatched
- seed 0: accuracy=0.4906
- seed 1: accuracy=0.4667
- seed 2: accuracy=0.6955
- pooled across seeds:
  PR_A(cyc-DAG) mean=0.004973 CI=[-0.012022,0.022123]
  ΔCyc2_A(cyc-DAG) mean=-0.000070 CI=[-0.001462,0.001312]
  ΔCyc3_A(cyc-DAG) mean=-0.000255 CI=[-0.000811,0.000318]
  ΔCyc4_A(cyc-DAG) mean=0.000012 CI=[-0.000263,0.000284]
  PR_B(cyc-DAG) mean=0.016260 CI=[-0.008909,0.042548]
  ΔCyc2_B(cyc-DAG) mean=-0.000105 CI=[-0.000288,0.000073]
  ΔCyc3_B(cyc-DAG) mean=0.000014 CI=[-0.000055,0.000080]
  ΔCyc4_B(cyc-DAG) mean=-0.000020 CI=[-0.000048,0.000008]

## Variant: no_drift
- seed 0: accuracy=0.8702
- seed 1: accuracy=0.7064
- seed 2: accuracy=0.8446
- pooled across seeds:
  PR_A(cyc-DAG) mean=0.001661 CI=[-0.007704,0.011572]
  ΔCyc2_A(cyc-DAG) mean=0.001568 CI=[0.000299,0.002822]
  ΔCyc3_A(cyc-DAG) mean=-0.000636 CI=[-0.001223,-0.000050]
  ΔCyc4_A(cyc-DAG) mean=0.000137 CI=[-0.000082,0.000364]
  PR_B(cyc-DAG) mean=-0.001031 CI=[-0.014599,0.013002]
  ΔCyc2_B(cyc-DAG) mean=0.000990 CI=[0.000654,0.001328]
  ΔCyc3_B(cyc-DAG) mean=-0.000415 CI=[-0.000552,-0.000278]
  ΔCyc4_B(cyc-DAG) mean=0.000050 CI=[0.000007,0.000092]

## Variant: no_injection
- seed 2: accuracy=0.6998
- seed 1: accuracy=0.6240
- seed 0: accuracy=0.6370
- pooled across seeds:
  PR_A(cyc-DAG) mean=-0.072058 CI=[-0.103640,-0.039794]
  ΔCyc2_A(cyc-DAG) mean=0.001261 CI=[-0.000856,0.003348]
  ΔCyc3_A(cyc-DAG) mean=-0.001562 CI=[-0.002785,-0.000309]
  ΔCyc4_A(cyc-DAG) mean=0.001154 CI=[0.000478,0.001830]
  PR_B(cyc-DAG) mean=-0.028182 CI=[-0.064190,0.008325]
  ΔCyc2_B(cyc-DAG) mean=0.000367 CI=[0.000200,0.000535]
  ΔCyc3_B(cyc-DAG) mean=-0.000179 CI=[-0.000243,-0.000116]
  ΔCyc4_B(cyc-DAG) mean=0.000068 CI=[0.000048,0.000089]

## Variant: mechanism
- seed 2: accuracy=0.8046
- seed 0: accuracy=0.8792
- seed 1: accuracy=0.5504
- pooled across seeds:
  PR_A(cyc-DAG) mean=-0.004728 CI=[-0.012227,0.003504]
  ΔCyc2_A(cyc-DAG) mean=0.000026 CI=[-0.001412,0.001409]
  ΔCyc3_A(cyc-DAG) mean=-0.000311 CI=[-0.000994,0.000364]
  ΔCyc4_A(cyc-DAG) mean=0.000190 CI=[-0.000087,0.000463]
  PR_B(cyc-DAG) mean=0.011758 CI=[-0.002969,0.026906]
  ΔCyc2_B(cyc-DAG) mean=-0.000323 CI=[-0.000693,0.000042]
  ΔCyc3_B(cyc-DAG) mean=0.000093 CI=[-0.000065,0.000248]
  ΔCyc4_B(cyc-DAG) mean=0.000060 CI=[0.000011,0.000109]

