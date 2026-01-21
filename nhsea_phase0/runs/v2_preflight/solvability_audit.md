# V2 Solvability Audit

This audit uses only generator logic; no model is involved.

Config source: runs/v2_preflight/forward_train/no_injection/seed_0/checkpoint_final.pt
Config: n_symbols=16, n_facts=8, T=64, vocab_size=200
Seeds: 0, 1, 2
Eval size per seed/task: 10000

## Summary
- forward ambiguous_rate=0.000000, ceiling_acc=1.000000
- backward ambiguous_rate=0.000000, ceiling_acc=1.000000
- overall ambiguous_rate=0.000000, ceiling_acc=1.000000

## By n_facts
- n_facts=8: ambiguous_rate=0.000000, ceiling_acc=1.000000
