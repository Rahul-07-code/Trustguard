# Evidence Index

| Claim | Evidence | Status |
|---|---|---|
| Test has 1,155,597 rows and 21 classes | data_audit/dataset_summary.json | PASS |
| 46 target-excluded features | data_audit/dataset_summary.json; leakage/feature_leakage_audit.csv | PASS |
| Reported metrics reproduce | reproduction/reproduction_comparison.csv | PASS |
| Independent evaluator used | independent_evaluation/metrics.json | PASS |
| Split process documented | split_audit/split_methodology.md | NOT VERIFIABLE |
| No fitted preprocessing leakage in located trainer | leakage/preprocessing_leakage_audit.md | PASS/WARNING |
| Exact duplicates audited | leakage/cross_split_duplicate_counts.csv | WARNING: one-for-one cross-split overlap observed |
| Near duplicates audited | leakage/near_duplicate_results.csv | WARNING: sampled |
| Class rationale documented | class_selection/class_selection_audit.md | NOT VERIFIABLE |
| Test history audited | evaluation_integrity/test_contamination_audit.md | NOT VERIFIABLE historically |
| Model and dataset hashes recorded | reproducibility/artifact_hashes.json | PASS |
| Error count and pairs recorded | error_analysis/ | PASS |
