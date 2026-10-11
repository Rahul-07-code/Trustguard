# Independent Evaluation

This evaluator independently loads the preserved model, streams the untouched test CSV, and computes metrics and confusion matrices without calling the original `evaluate` function.
