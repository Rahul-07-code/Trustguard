"""Validate the federated histogram booster against a centralized reference.

The script builds a small synthetic multi-class dataset, splits it into three
non-overlapping client shards, runs the simulated federated protocol (real
request / histogram / decisions / tree messages over queues), and runs the
centralized reference on the concatenated data using the *same* histogram and
split code. It asserts that federated and centralized training agree.

Exit code is non-zero if equivalence fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.federated.histogram_booster import (  # noqa: E402
    BoosterParams,
    ClientSession,
    aggregate_histograms,
    bin_edges_from_ranges,
    bin_features,
    count_leaves,
    initial_margins,
    node_histograms,
    run_simulated_federation,
    softmax,
    train_centralized,
    tree_depth,
)

RESULTS = ROOT / "results" / "fl_xgboost" / "validation"


def make_dataset(seed: int, rows: int = 3000, classes: int = 3, features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(classes, features))
    labels = rng.integers(0, classes, size=rows)
    matrix = centers[labels] + rng.normal(0.0, 1.0, size=(rows, features))
    return matrix.astype(np.float32), labels.astype(np.int64)


def tree_signature(tree: dict[int, dict[str, object]]) -> list[tuple[int, bool, int, int]]:
    return [tuple((int(node_id), bool(node["is_leaf"]), int(node["feature"]), int(node["bin_index"]))) for node_id, node in sorted(tree.items())]


def main() -> None:
    matrix, labels = make_dataset(seed=42)
    classes = 3
    params = BoosterParams(max_depth=4, max_bin=16, learning_rate=0.3, min_samples_leaf=5, num_class=classes, min_child_weight=0.5)
    rounds = 6

    bounds = np.linspace(0, matrix.shape[0], 4, dtype=int)
    shards = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(3)]

    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    edges = bin_edges_from_ranges(mins, maxs, params.max_bin)
    binned_all = bin_features(matrix, edges)
    base_margins = initial_margins(np.bincount(labels, minlength=classes), classes)

    trees_central, session_central = train_centralized(binned_all, labels, params, base_margins, rounds=rounds)
    probabilities_central = session_central.probabilities()

    sessions = [ClientSession(binned_all[shard], labels[shard], params, base_margins, client_id=f"client_{index + 1}") for index, shard in enumerate(shards)]
    trees_federated = run_simulated_federation(sessions, params, rounds=rounds)

    global_margins = np.zeros((matrix.shape[0], classes))
    for session, shard in zip(sessions, shards):
        global_margins[shard] = session.margins
    probabilities_federated = softmax(global_margins)

    # Direct check that histogram aggregation is exact.
    probe_params = BoosterParams(max_bin=params.max_bin, num_class=classes)
    single = ClientSession(binned_all, labels, probe_params, base_margins, client_id="all")
    single.begin_class(0)
    gradient = single.probabilities()[:, 0] - (labels == 0).astype(np.float64)
    hessian = np.maximum(single.probabilities()[:, 0] * (1.0 - single.probabilities()[:, 0]), 1e-12)
    full_hist = node_histograms(binned_all, gradient, hessian, np.arange(matrix.shape[0]), params.max_bin)
    shard_hist = [node_histograms(binned_all[shard], gradient[shard], hessian[shard], np.arange(int(shard.stop - shard.start)), params.max_bin) for shard in shards]
    aggregated = aggregate_histograms(shard_hist)
    histogram_max_abs_diff = float(np.max(np.abs(np.array(full_hist) - np.array(aggregated))))

    structure_match = all(
        tree_signature(trees_central[r][k]) == tree_signature(trees_federated[r][k]) for r in range(rounds) for k in range(classes)
    )
    max_probability_diff = float(np.max(np.abs(probabilities_central - probabilities_federated)))
    agreement = float(np.mean(np.argmax(probabilities_central, axis=1) == np.argmax(probabilities_federated, axis=1)))

    evidence = {
        "seed": 42,
        "rows": int(matrix.shape[0]),
        "clients": len(shards),
        "classes": classes,
        "rounds": rounds,
        "params": params.as_dict(),
        "histogram_max_abs_diff": histogram_max_abs_diff,
        "histogram_exact_within_tolerance": histogram_max_abs_diff < 1e-9,
        "tree_structure_identical": bool(structure_match),
        "max_probability_diff": max_probability_diff,
        "prediction_agreement": agreement,
        "leaves_per_tree_central": [count_leaves(trees_central[0][k]) for k in range(classes)],
        "tree_depths_central": [tree_depth(trees_central[0][k]) for k in range(classes)],
        "passed": bool(structure_match and histogram_max_abs_diff < 1e-9 and max_probability_diff < 1e-9 and agreement == 1.0),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "federated_vs_centralized_equivalence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("=" * 64)
    print("FEDERATED HISTOGRAM BOOSTER EQUIVALENCE VALIDATION")
    print("=" * 64)
    print(f"clients                : {len(shards)}  rows={matrix.shape[0]}  classes={classes}  rounds={rounds}")
    print(f"histogram exactness    : max|full - sum(shards)| = {histogram_max_abs_diff}")
    print(f"tree structure identical: {structure_match}")
    print(f"max probability diff   : {max_probability_diff}")
    print(f"prediction agreement   : {agreement}")
    print(f"RESULT                 : {'PASS' if evidence['passed'] else 'FAIL'}")
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()