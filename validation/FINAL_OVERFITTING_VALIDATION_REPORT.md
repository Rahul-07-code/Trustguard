# Final Overfitting Validation Report

## Objective
Evaluate the frozen 21-class model for conventional overfitting and quantify exact train/test feature overlap without retraining or modifying artifacts.

## Metrics
Train accuracy=0.999894119222127, macro F1=0.999922982432491.
Validation accuracy=0.999651229610346, macro F1=0.996413488470589.
Test accuracy=0.999681549882874, macro F1=0.997024707232144.

## Overlap
Test rows overlapping train=56836 (4.9183%). Non-overlapping=1098761 (95.0817%).

## Interpretation
Q1. Conventional overfitting: **No strong standalone evidence**, although nonzero gaps exist. Train-to-validation Macro F1 gap is `0.00350949`; train-to-test Macro F1 gap is `0.00289828`.

Q2. Generalization gaps: train-to-validation accuracy gap is `0.00024289` and Macro F1 gap is `0.00350949`; train-to-test accuracy gap is `0.00021257` and Macro F1 gap is `0.00289828`. Full details are in `overfitting/generalization_gaps.csv`.

Q3. Overlap performance: overlapping test accuracy/F1 are `0.99989443`/`0.99993209`; non-overlapping test accuracy/F1 are `0.99967054`/`0.99690851`.

Q4. Non-overlap performance: the model remains very strong, but is lower than the overlapping subset. See `overlap/nonoverlapping_test_metrics.json`.

Q5. Unseen-data claim: **NO**. The full `0.99968155` test accuracy cannot be treated as completely unseen traffic because 56,836 test feature rows occur in training.

Q6. Generalizes well: only with qualification. Aggregate performance is strong across the available partitions, but the test estimate is affected by exact train/test overlap and cannot support a clean unseen-traffic claim.

Q7. Additional validation: obtain raw/source data and perform group-aware or independently sourced evaluation.

## Final verdict
**D. PERFORMANCE AFFECTED BY TRAIN-TEST OVERLAP**

The overlap finding is distinct from conventional model overfitting.
