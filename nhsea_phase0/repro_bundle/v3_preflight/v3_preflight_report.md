# V3 Preflight Report

## Leak gate
- forward: AUROC=0.5000 pass=True
- backward: AUROC=0.5000 pass=True

## In-task accuracy
- conclusion: acc=1.0000 CI=[0.9999,1.0000] n=30000
- topology: acc=0.6509 CI=[0.6455,0.6562] n=30000

## Zero-shot transfer
- conclusion→topology: acc=0.5000 CI=[0.4943,0.5057] n=30000
- topology→conclusion: acc=0.4971 CI=[0.4914,0.5028] n=30000

## Few-shot transfer (head-only)
- conclusion→topology n=32: acc=0.5327 CI=[0.5270,0.5383] n=30000
- topology→conclusion n=32: acc=0.5031 CI=[0.4975,0.5088] n=30000
- conclusion→topology n=128: acc=0.5374 CI=[0.5317,0.5430] n=30000
- topology→conclusion n=128: acc=0.5018 CI=[0.4962,0.5075] n=30000
- conclusion→topology n=512: acc=0.5405 CI=[0.5349,0.5461] n=30000
- topology→conclusion n=512: acc=0.5027 CI=[0.4971,0.5084] n=30000

## Topology sensitivity (paired OBC vs PBC)
- PR diff (OBC-PBC): mean=-0.0187 CI=[-0.0324,-0.0044] n_pairs=15000
- Mass diff (OBC-PBC): mean=-0.0004 CI=[-0.0007,-0.0001] n_pairs=15000

## Gate status
- leak_gate_pass=True
- in_task_pass=True
- transfer_pass=True (n_train_gate=512)
- topology_pass=True (expect PR diff < 0 or mass diff > 0)
- overall_pass=True

## Why v3 answers the NHSEA hypothesis better than v1/v2
- v1/v2 transfer sat at chance, so reciprocity tests were non-discriminative.
- v3 enforces an explicit OBC vs PBC topology contrast tied to boundary localization observables.
- Baseline transfer-learnability is gated before any mechanism compute is allowed.
