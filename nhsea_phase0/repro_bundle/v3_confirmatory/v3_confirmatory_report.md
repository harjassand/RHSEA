# V3 Confirmatory Report

## Primary endpoints
- E1 ΔSelLoc(mech)-ΔSelLoc(sym): mean=0.000064 CI=[0.000022,0.000105] n_pairs=15000
- E2 acc_512(mech)-acc_512(base): mean=-0.016600 CI=[-0.023767,-0.009333] n_inst=30000

## SelLocGap by variant
- mechanism: ΔSelLoc mean=0.000015 CI=[-0.000019,0.000080] n=3
- no_injection: ΔSelLoc mean=0.000000 CI=[0.000000,0.000000] n=3
- symmetric_control_v2_normmatched: ΔSelLoc mean=-0.000049 CI=[-0.000098,0.000004] n=3

## Few-shot transfer (n=512)
- mechanism: acc=0.536567 CI=[0.530919,0.542205]
- no_injection: acc=0.553167 CI=[0.547534,0.558785]
- symmetric_control_v2_normmatched: acc=0.541733 CI=[0.536090,0.547366]

## Topology sensitivity (PR and mass)
- mechanism: PR diff mean=0.004577 CI=[-0.011628,0.026848]
- mechanism: mass diff mean=-0.000093 CI=[-0.000351,0.000235]
- no_injection: PR diff mean=0.030095 CI=[0.007464,0.044620]
- no_injection: mass diff mean=0.000751 CI=[0.000368,0.001470]
- symmetric_control_v2_normmatched: PR diff mean=-0.014172 CI=[-0.035777,0.000798]
- symmetric_control_v2_normmatched: mass diff mean=0.000125 CI=[-0.000344,0.000911]

## Scale-match / gamma stats
- mechanism: gamma_median=2.420910 zero_rate_B0_median=0.000000 rho_median=2.416187 flag_any=True
- no_injection: gamma_median=1.000000 zero_rate_B0_median=1.000000 rho_median=1.000000 flag_any=False
- symmetric_control_v2_normmatched: gamma_median=0.330387 zero_rate_B0_median=0.000000 rho_median=0.999501 flag_any=False

## Pass/fail
- E1_pass=True
- E2_pass=False
- overall_pass=False
