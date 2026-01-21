# NHSEA v3 Benchmark Design Memo

## A. Why v1/v2 failed to test the claim
- Baseline L is already directional, so non-reciprocity is not unique to the mechanism.
- The transfer task does not isolate boundary localization from generic feature learning; transfer remains at chance.

## B. Discriminative task family (must include OBC and PBC)
Propose a dataset with matched surface statistics but different graph topology:

- OBC instances: directed chains that terminate in a sink conclusion span; the conclusion is only reachable via directed edges from premises.
- PBC instances: same chains but with a forced cycle that removes sink behavior.
- Surface features must be matched: proposition lengths, positions, token types, and candidate placements.

Definition details:
- Propositions are token spans with fixed-length distribution; edges are encoded as pairs between proposition IDs.
- Labels encode the correct conclusion span index (binary candidate selection) and/or a topology class (OBC vs PBC).
- Boundary is the sink conclusion span in OBC; in PBC no sink exists by construction.

## C. Primary observables tied to the hypothesis
Not accuracy; use localization and topology-sensitive observables:
- Localization at boundary: participation ratio (PR) and mass-at-conclusion.
- Topology contrast: Delta(metric) between matched OBC vs PBC instances.

## D. Preflight discriminativity gate (baseline-only)
Before any mechanism runs:
- Baseline must show non-trivial ability to classify OBC vs PBC (or predict conclusion) above chance.
- Baseline must show measurable change in localization observables between OBC and PBC.
- If either fails, stop and redesign the generator.
