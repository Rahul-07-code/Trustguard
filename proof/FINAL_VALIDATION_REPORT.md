# Final Validation Report

## Executive summary
The preserved 21-class model is independently loadable and its test metrics reproduce the reported metrics. The located training path excludes target columns and uses validation for early stopping. Split provenance, historical class-selection rationale, upstream feature-generation provenance, and historical test-set decision history are unavailable.

## Exact reported and reproduced result
Reported accuracy: 0.999681549882874; reproduced accuracy: 0.999681549882874; errors: 368 of 1155597. See `reproduction/reproduction_comparison.csv`.

## Findings
- Dataset and 21-class support: PASS for preserved files.
- Preprocessing leakage in located training code: PASS; upstream preparation: WARNING.
- Exact duplicates: WARNING; one-for-one cross-split feature and complete-row hash overlap affects 56,693 train/validation rows, 56,836 train/test rows, and 12,178 validation/test rows. No rows were removed.
- Near duplicates: WARNING; bounded sample only.
- Feature leakage: WARNING; no direct target column is used, but upstream provenance is absent.
- Split methodology: NOT VERIFIABLE, with observed cross-split duplicate contamination.
- Test contamination history: NOT VERIFIABLE beyond the located code path.
- Class-selection rationale: NOT VERIFIABLE.
- Independent evaluation and reproduction: PASS.
- Error analysis and uncertainty: PASS for computed test-row analysis.
- Hardware and resource validation: PASS for the recorded run; audit use was bounded.

## Publication-readiness assessment
Status: NOT VALIDATED. The numerical result is reproducible for the preserved artifact set, but the observed cross-split duplicate overlap prevents a leakage-free, independent-test claim. Split provenance, class-selection rationale, and upstream feature provenance must be reconstructed before publication use.

Audit runtime: 83.4 seconds.
