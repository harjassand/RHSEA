# NHSEA Phase 3 Task Specification (Reciprocity Test)

Status: locked

## Overview
Phase 3 evaluates backward/abductive reasoning against forward-trained models. The backward task must be matched to the
forward task in surface statistics and must pass the generator-only leak gate (AUROC <= 0.55 with invariant features).

## Forward task (reference)
Input: premises + two candidate conclusions (c1, c2)
Label: true conclusion index in {0,1}

This is the existing forward-chain generator (ForwardChainConfig). The Phase 3 backward task uses the same underlying
proposition graph and tokenization constraints.

## Backward/abductive task (new)
Input: conclusion tokens + two candidate antecedents (a1, a2), matched in surface stats
Label: true antecedent index in {0,1}

### Decoy construction (abductive)
Given a proposition graph G over M propositions:
1) Sample a true antecedent chain that implies the conclusion (c*), using the same forward-chain logic.
2) Construct a decoy chain with matched token/proposition counts and matched span lengths, but that does not entail c*.
3) Ensure the conclusion tokens are identical regardless of which antecedent is correct (conclusion provided separately,
   not embedded inside candidate spans).

Reject and resample if:
- decoy chain implies c* (path exists in G)
- any candidate spans violate anti-leak constraints

## Tokenization and spans (shared)
- Fixed T=64.
- Proposition spans are contiguous, length 3–6.
- Candidate antecedent spans must match in length distribution and token stats.
- Premise/conclusion markers must be symmetric across options.

## Anti-leak constraints (hard)
For both forward and backward tasks:
- candidate spans matched by length, token frequency, and marker identity
- candidate positions matched in distribution
- identical padding and positional patterns
- reject if any invariant feature predicts label above AUROC 0.55

## Leak gate features (generator-only)
Allowed invariant features computed from candidate-specific scalar f1, f2:
- min(f1, f2)
- max(f1, f2)
- abs(f1 - f2)
No ordered or raw (f1, f2) features permitted.

## Label encoding
- Forward: 0 if c1 true, 1 if c2 true
- Backward: 0 if a1 true antecedent, 1 if a2 true antecedent

## Stop conditions
- If backward generator fails leak gate (AUROC > 0.55), Phase 3 is blocked.
- If abductive decoys cannot be constructed without leakage under these constraints, revise generator before any training.

## Falsification threshold (locked)
Mechanism backward accuracy must be at least 10 percentage points worse than the no-injection baseline:
- acc_abductive_mechanism <= acc_abductive_no_injection - 0.10 (pooled across seeds)
If this criterion is not met, the directional reciprocity hypothesis is falsified for this instantiation.
