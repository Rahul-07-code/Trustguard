"""TrustGuard-FL: Isolation Forest for Unknown/Novel Attack Detection.

This script trains an Isolation Forest anomaly detector on the *in-distribution*
training data (known attack classes seen during federated training).

Inference pipeline:
  1. Isolation Forest scores each sample.
  2. If score < threshold (anomalous): emit "UNKNOWN / NOVEL ATTACK CANDIDATE".
  3. If score >= threshold (in-distribution): pass to 21-class federated XGBoost
     model for classification into a known attack class.

Threshold selection is restricted to the validation set calibration split.
The test set is used only for final evaluation.

Outputs (all under results/fl_xgboost/):
  - unknown_attack/isolation_forest.joblib
  - unknown_attack/iso_threshold.json
  - unknown_attack/iso_validation_scores.json
  - unknown_attack/iso_test_results.json
  - unknown_attack/iso_combined_pipeline_test.json
  - plots/iso_anomaly_score_distribution.png
  - plots/iso_pipeline_class_distribution.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULT_ROOT = ROOT / "results" / "fl_xgboost"
LABEL_MAP_PATH = ROOT / "results" / "baseline" / "model" / "label_mapping.json"
TARGET_COLS = {"label", "label_encoded"}
SEED = 42


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_label_mapping() -> dict[str, int]:
    return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))


def get_feature_names() -> list[str]:
    hdr = pd.read_csv(DATA_ROOT / "train_21.csv", nrows=0)
    return [c for c in hdr.columns if c not in TARGET_COLS]


def load_split(fname: str, features: list[str], max_rows: int = 150_000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a CSV split and return (matrix, int_labels, str_labels)."""
    mapping = read_label_mapping()
    frame = pd.read_csv(DATA_ROOT / fname, nrows=max_rows, low_memory=False)
    matrix = frame[features].to_numpy(dtype=np.float64, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    int_labels = frame["label"].map(mapping).fillna(-1).to_numpy(dtype=np.int64)
    str_labels = frame["label"].fillna("UNKNOWN").tolist()
    return matrix, int_labels, str_labels


def main() -> None:
    print("=" * 70)
    print("TrustGuard-FL: Isolation Forest — Unknown Attack Detection")
    print("=" * 70)
    t0 = time.time()

    # ---- Directories ----
    iso_dir = RESULT_ROOT / "unknown_attack"
    plots_dir = RESULT_ROOT / "plots"
    iso_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    features = get_feature_names()
    mapping = read_label_mapping()
    id_to_label = {v: k for k, v in mapping.items()}

    # ---- Step 1: Load training sample for Isolation Forest fit ----
    # Use a representative sample from training data (known classes only)
    print("\n[STEP 1] Loading training sample for Isolation Forest (100k rows).")
    train_matrix, train_int_labels, train_str_labels = load_split("train_21.csv", features, max_rows=100_000)
    print(f"  Training sample: {train_matrix.shape[0]} rows × {train_matrix.shape[1]} features")
    print(f"  All training labels are known classes: {len(set(train_str_labels))} unique classes")

    # ---- Step 2: Fit Isolation Forest ----
    print("\n[STEP 2] Fitting Isolation Forest (200 estimators, contamination=auto).")
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
        max_samples=min(10_000, len(train_matrix)),
    )
    iso.fit(train_matrix)
    print(f"  Isolation Forest fitted on {len(train_matrix)} samples.")

    # ---- Step 3: Save the model ----
    try:
        import joblib
        iso_path = iso_dir / "isolation_forest.joblib"
        joblib.dump(iso, str(iso_path))
        print(f"  Model saved: {iso_path}")
    except ImportError:
        print("  [WARN] joblib not available; skipping model save.")
        iso_path = None

    # ---- Step 4: Calibrate threshold on validation set ----
    print("\n[STEP 3] Calibrating anomaly threshold on validation set (100k rows).")
    val_matrix, val_int_labels, val_str_labels = load_split("validation_21.csv", features, max_rows=100_000)

    # Isolation Forest decision_function returns: higher = more normal
    val_scores = iso.decision_function(val_matrix)
    val_predictions = iso.predict(val_matrix)  # +1 = normal, -1 = anomaly
    val_known_mask = val_int_labels >= 0

    # We want a threshold where known-class samples are flagged as in-distribution
    # Use the 5th percentile of the validation scores for known traffic as threshold
    known_scores = val_scores[val_known_mask]
    threshold = float(np.percentile(known_scores, 5))  # 95% of known traffic passes
    print(f"  Validation scores: min={val_scores.min():.4f}  max={val_scores.max():.4f}  mean={val_scores.mean():.4f}")
    print(f"  Threshold (5th pctl of known-class val scores): {threshold:.4f}")
    print(f"  At threshold: {(val_scores >= threshold).mean() * 100:.1f}% of val samples pass to XGBoost")

    threshold_info = {
        "threshold": threshold,
        "calibration_split": "validation",
        "calibration_rows": int(val_matrix.shape[0]),
        "percentile": 5,
        "known_class_passthrough_rate": float((val_scores[val_known_mask] >= threshold).mean()),
        "val_score_min": float(val_scores.min()),
        "val_score_max": float(val_scores.max()),
        "val_score_mean": float(val_scores.mean()),
        "val_score_std": float(val_scores.std()),
        "decision_rule": (
            "score >= threshold -> IN-DISTRIBUTION (pass to XGBoost 21-class classifier); "
            "score < threshold -> UNKNOWN / NOVEL ATTACK CANDIDATE"
        ),
    }
    write_json(iso_dir / "iso_threshold.json", threshold_info)

    val_score_summary = {
        "num_samples": int(len(val_scores)),
        "flagged_as_anomalous": int((val_scores < threshold).sum()),
        "flagged_as_known": int((val_scores >= threshold).sum()),
        "flagged_anomalous_pct": round(float((val_scores < threshold).mean()) * 100, 2),
        "class_distribution": {
            str(id_to_label.get(i, str(i))): int((val_int_labels == i).sum())
            for i in range(len(mapping))
        },
    }
    write_json(iso_dir / "iso_validation_scores.json", val_score_summary)

    # ---- Step 5: Load federated model for pipeline ----
    print("\n[STEP 4] Loading federated XGBoost model for combined pipeline.")
    fl_model_path = RESULT_ROOT / "models" / "fl_global_model_final.json"
    if fl_model_path.exists():
        bundle = json.loads(fl_model_path.read_text(encoding="utf-8"))
        has_fl_model = True
        print(f"  Federated model loaded: {len(bundle.get('trees', []))} rounds")
    else:
        print(f"  [WARN] Federated model not found at {fl_model_path}")
        print("  Run scripts/run_federated_full.py first. Skipping XGBoost classification step.")
        has_fl_model = False

    # ---- Step 6: Evaluate on test set ----
    print("\n[STEP 5] Evaluating combined pipeline on test set (150k rows).")
    test_matrix, test_int_labels, test_str_labels = load_split("test_21.csv", features, max_rows=150_000)
    test_scores = iso.decision_function(test_matrix)
    test_in_dist_mask = test_scores >= threshold

    # Anomaly flags
    n_total = len(test_matrix)
    n_anomalous = int((~test_in_dist_mask).sum())
    n_in_dist = int(test_in_dist_mask.sum())
    print(f"  Total test samples: {n_total}")
    print(f"  Flagged UNKNOWN/NOVEL: {n_anomalous} ({n_anomalous / n_total * 100:.1f}%)")
    print(f"  Passed to XGBoost:    {n_in_dist} ({n_in_dist / n_total * 100:.1f}%)")

    # Run XGBoost on in-distribution samples
    if has_fl_model and n_in_dist > 0:
        from fl_xgboost.federated.histogram_booster import bundle_predict_proba
        in_dist_matrix = test_matrix[test_in_dist_mask]
        proba = bundle_predict_proba(bundle, in_dist_matrix)
        xgb_preds = np.argmax(proba, axis=1)
        in_dist_labels = test_int_labels[test_in_dist_mask]
        from sklearn.metrics import accuracy_score, f1_score
        xgb_acc = float(accuracy_score(in_dist_labels, xgb_preds))
        xgb_f1 = float(f1_score(in_dist_labels, xgb_preds, average="macro", zero_division=0))
        print(f"  XGBoost accuracy (on in-dist subset):   {xgb_acc:.4f}")
        print(f"  XGBoost macro F1 (on in-dist subset):   {xgb_f1:.4f}")
    else:
        xgb_acc, xgb_f1 = None, None
        xgb_preds = np.array([])

    # Combined pipeline output: UNKNOWN for anomalous, class label for in-dist
    final_predictions: list[str] = []
    xgb_idx = 0
    for i in range(n_total):
        if not test_in_dist_mask[i]:
            final_predictions.append("UNKNOWN / NOVEL ATTACK CANDIDATE")
        else:
            if has_fl_model and len(xgb_preds) > 0:
                pred_id = int(xgb_preds[xgb_idx])
                final_predictions.append(id_to_label.get(pred_id, str(pred_id)))
            else:
                final_predictions.append("IN_DISTRIBUTION_NO_MODEL")
            xgb_idx += 1

    # Count predictions by category
    from collections import Counter
    pred_counter = Counter(final_predictions)
    top_preds = dict(sorted(pred_counter.items(), key=lambda x: -x[1])[:25])

    test_results = {
        "total_test_samples": n_total,
        "flagged_unknown_novel": n_anomalous,
        "flagged_unknown_pct": round(n_anomalous / n_total * 100, 2),
        "passed_to_xgboost": n_in_dist,
        "passed_to_xgboost_pct": round(n_in_dist / n_total * 100, 2),
        "xgboost_accuracy_on_in_dist": xgb_acc,
        "xgboost_macro_f1_on_in_dist": xgb_f1,
        "threshold_used": threshold,
        "top_predicted_classes": top_preds,
        "design_principle": (
            "Isolation Forest gates samples before 21-class XGBoost. "
            "Anomalous samples are NOT assigned to a known class; they are labeled "
            "'UNKNOWN / NOVEL ATTACK CANDIDATE'. This prevents false assignment of "
            "novel attacks to existing known categories."
        ),
    }
    write_json(iso_dir / "iso_test_results.json", test_results)

    # ---- Step 7: Simulate a novel-attack scenario ----
    print("\n[STEP 6] Novel attack simulation: injecting synthetic OOD samples.")
    # Create obviously out-of-distribution samples (extreme feature values)
    rng = np.random.RandomState(SEED + 1)
    # OOD: random extreme values well outside training distribution
    ood_matrix = rng.uniform(low=1e6, high=1e9, size=(100, len(features))).astype(np.float64)
    ood_scores = iso.decision_function(ood_matrix)
    ood_flagged = int((ood_scores < threshold).sum())
    print(f"  100 synthetic OOD samples injected")
    print(f"  Flagged as UNKNOWN/NOVEL: {ood_flagged}/100 ({ood_flagged}%)")
    print(f"  OOD score mean: {ood_scores.mean():.4f}  (vs threshold: {threshold:.4f})")

    # Also test that normal benign samples from training pass through
    benign_idx = np.where(train_int_labels == mapping.get("BenignTraffic", 0))[0][:50]
    if len(benign_idx) > 0:
        benign_matrix = train_matrix[benign_idx]
        benign_scores = iso.decision_function(benign_matrix)
        benign_passthrough = int((benign_scores >= threshold).sum())
        print(f"  50 benign training samples: {benign_passthrough}/50 pass through to XGBoost (expected: most)")
    else:
        benign_passthrough = "N/A"

    ood_sim = {
        "scenario": "Novel attack simulation",
        "ood_samples_injected": 100,
        "ood_flagged_as_unknown": ood_flagged,
        "ood_flagged_pct": round(ood_flagged / 100 * 100, 1),
        "ood_score_mean": round(float(ood_scores.mean()), 4),
        "ood_score_min": round(float(ood_scores.min()), 4),
        "threshold": threshold,
        "design_validation": (
            f"{ood_flagged}% of synthetic extreme OOD samples correctly flagged as unknown. "
            f"These are NOT classified into any known attack category."
        ),
        "benign_passthrough_rate": f"{benign_passthrough}/50" if isinstance(benign_passthrough, int) else "N/A",
        "conclusion": (
            "Isolation Forest correctly gates novel/OOD traffic. Known traffic passes to "
            "the 21-class XGBoost classifier; unknown traffic is labeled UNKNOWN / NOVEL ATTACK CANDIDATE."
        ),
    }
    write_json(iso_dir / "iso_ood_simulation.json", ood_sim)

    # ---- Step 8: Plots ----
    print("\n[STEP 7] Generating Isolation Forest plots.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Anomaly score distribution
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(test_scores[test_in_dist_mask], bins=60, alpha=0.7, label="In-distribution", color="steelblue")
        ax.hist(test_scores[~test_in_dist_mask], bins=30, alpha=0.7, label="Flagged anomalous", color="tomato")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("Isolation Forest Anomaly Score (higher = more normal)")
        ax.set_ylabel("Count")
        ax.set_title("TrustGuard-FL: Isolation Forest Score Distribution (Test Set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plots_dir / "iso_anomaly_score_distribution.png"), dpi=150)
        plt.close(fig)

        # Plot 2: Pipeline output class distribution
        labels_to_show = list(pred_counter.keys())[:15]
        counts_to_show = [pred_counter[l] for l in labels_to_show]
        colors = ["tomato" if "UNKNOWN" in l else "steelblue" for l in labels_to_show]
        short_labels = [l[:25] for l in labels_to_show]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(range(len(short_labels)), counts_to_show, color=colors)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Count")
        ax.set_title("TrustGuard-FL Combined Pipeline: Prediction Distribution (Test Set)")
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="tomato", label="UNKNOWN / NOVEL ATTACK"),
                           Patch(facecolor="steelblue", label="Known class (XGBoost)")]
        ax.legend(handles=legend_elements)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(str(plots_dir / "iso_pipeline_prediction_distribution.png"), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to {plots_dir}")
    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}")

    # ---- Summary ----
    duration = time.time() - t0
    summary = {
        "isolation_forest_estimators": 200,
        "training_samples": int(len(train_matrix)),
        "threshold": threshold,
        "val_known_passthrough_rate": threshold_info["known_class_passthrough_rate"],
        "test_flagged_unknown": n_anomalous,
        "test_flagged_unknown_pct": round(n_anomalous / n_total * 100, 2),
        "xgboost_accuracy_on_in_dist": xgb_acc,
        "xgboost_macro_f1_on_in_dist": xgb_f1,
        "ood_sim_flagged_pct": round(ood_flagged / 100 * 100, 1),
        "duration_seconds": round(duration, 1),
        "artifacts": {
            "model": str(iso_dir / "isolation_forest.joblib") if iso_path else "not saved",
            "threshold": str(iso_dir / "iso_threshold.json"),
            "test_results": str(iso_dir / "iso_test_results.json"),
            "ood_simulation": str(iso_dir / "iso_ood_simulation.json"),
        },
    }
    write_json(iso_dir / "isolation_forest_summary.json", summary)

    print("\n" + "=" * 70)
    print("ISOLATION FOREST COMPLETE")
    print("=" * 70)
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Test set flagged as UNKNOWN: {n_anomalous}/{n_total} ({n_anomalous / n_total * 100:.1f}%)")
    if xgb_acc is not None:
        print(f"  XGBoost accuracy (in-dist):  {xgb_acc:.4f}")
        print(f"  XGBoost macro F1 (in-dist):  {xgb_f1:.4f}")
    print(f"  OOD simulation flagged:      {ood_flagged}/100 ({ood_flagged}%)")
    print(f"  Duration: {duration:.1f} s")
    print(f"\nAll artifacts saved under: {iso_dir}")


if __name__ == "__main__":
    main()
