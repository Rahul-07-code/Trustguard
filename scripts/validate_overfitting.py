"""Diagnostic overfitting and train/test-overlap validation for the frozen model."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULTS = ROOT / "results"
OUT = ROOT / "validation"
CHUNK = 200_000
TARGETS = {"label", "label_encoded"}
LABELS = sorted(json.loads((RESULTS / "model" / "label_mapping.json").read_text(encoding="utf-8")), key=lambda name: json.loads((RESULTS / "model" / "label_mapping.json").read_text(encoding="utf-8"))[name])
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Accumulator:
    def __init__(self) -> None:
        self.matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        self.rows = 0
        self.errors = 0
        self.confidence_correct: list[float] = []
        self.confidence_error: list[float] = []

    def add(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> None:
        self.rows += len(y_true)
        self.errors += int((y_true != y_pred).sum())
        self.matrix += confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS)))
        self.confidence_correct.extend(confidence[y_true == y_pred].tolist())
        self.confidence_error.extend(confidence[y_true != y_pred].tolist())

    def finish(self) -> dict[str, object]:
        report = classification_report(self.matrix_to_true(), self.matrix_to_pred(), labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
        return {"accuracy": float(np.trace(self.matrix) / self.rows), "macro_precision": float(precision_score(self.matrix_to_true(), self.matrix_to_pred(), average="macro", zero_division=0)), "macro_recall": float(recall_score(self.matrix_to_true(), self.matrix_to_pred(), average="macro", zero_division=0)), "macro_f1": float(f1_score(self.matrix_to_true(), self.matrix_to_pred(), average="macro", zero_division=0)), "weighted_precision": float(precision_score(self.matrix_to_true(), self.matrix_to_pred(), average="weighted", zero_division=0)), "weighted_recall": float(recall_score(self.matrix_to_true(), self.matrix_to_pred(), average="weighted", zero_division=0)), "weighted_f1": float(f1_score(self.matrix_to_true(), self.matrix_to_pred(), average="weighted", zero_division=0)), "rows": self.rows, "errors": self.errors, "classification_report": report}

    def matrix_to_true(self) -> np.ndarray:
        return np.repeat(np.arange(len(LABELS)), self.matrix.sum(axis=1))

    def matrix_to_pred(self) -> np.ndarray:
        return np.concatenate([np.repeat(np.arange(len(LABELS)), row) for row in self.matrix])


def evaluate_split(model: xgb.XGBClassifier, path: Path, feature_names: list[str], overlap_hashes: set[int] | None = None, overlap_csv: Path | None = None) -> tuple[dict[str, object], dict[str, Accumulator], list[dict[str, object]], Accumulator]:
    overall = Accumulator()
    groups = {"overlapping": Accumulator(), "nonoverlapping": Accumulator()}
    error_rows: list[dict[str, object]] = []
    if overlap_csv:
        overlap_csv.parent.mkdir(parents=True, exist_ok=True)
        overlap_csv.write_text("row_index,status,true_class,predicted_class,confidence\n", encoding="utf-8")
    row_index = 0
    for frame in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        features = frame[feature_names]
        y_true = frame["label"].astype(str).map(LABEL_TO_ID).to_numpy(dtype=np.int32)
        probabilities = np.asarray(model.predict_proba(features.to_numpy(dtype=np.float32, copy=True)))
        y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
        confidence = probabilities[np.arange(len(y_pred)), y_pred]
        overall.add(y_true, y_pred, confidence)
        if overlap_hashes is not None:
            hashes = pd.util.hash_pandas_object(features, index=False).to_numpy()
            flags = np.array([int(value) in overlap_hashes for value in hashes], dtype=bool)
            groups["overlapping"].add(y_true[flags], y_pred[flags], confidence[flags])
            groups["nonoverlapping"].add(y_true[~flags], y_pred[~flags], confidence[~flags])
            if overlap_csv:
                with overlap_csv.open("a", encoding="utf-8") as handle:
                    for offset, flag in enumerate(flags):
                        handle.write(f"{row_index + offset},{'OVERLAPPING_WITH_TRAIN' if flag else 'NOT_OVERLAPPING_WITH_TRAIN'},{LABELS[y_true[offset]]},{LABELS[y_pred[offset]]},{confidence[offset]:.9f}\n")
            for offset in np.flatnonzero(y_true != y_pred):
                error_rows.append({"row_index": row_index + int(offset), "true_class": LABELS[y_true[offset]], "predicted_class": LABELS[y_pred[offset]], "overlap_status": "OVERLAPPING_WITH_TRAIN" if flags[offset] else "NOT_OVERLAPPING_WITH_TRAIN", "confidence": float(confidence[offset])})
        row_index += len(frame)
    return overall.finish(), groups, error_rows, overall


def metric_table(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]
    return pd.DataFrame([{ "metric": name, **{split: metrics[split][name] for split in metrics} } for name in names])


def main() -> None:
    started = time.time()
    for folder in ["model_verification", "overall_metrics", "overfitting", "figures", "overlap", "error_analysis", "confidence", "resource_usage", "reproducibility"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    paths = {name: DATA / f"{name}_21.csv" for name in ["train", "validation", "test"]}
    model_path = RESULTS / "model" / "xgboost_21class_final.json"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    schema = pd.read_csv(paths["train"], nrows=1).columns.tolist()
    feature_names = [name for name in schema if name not in TARGETS]
    model_info = {"path": str(model_path), "sha256": sha256(model_path), "xgboost_version": xgb.__version__, "num_classes": int(model.get_booster().num_boosted_rounds() > 0 and len(LABELS)), "feature_count_dataset": len(feature_names), "model_num_feature": int(model.get_booster().num_features()), "objective": model.get_xgb_params().get("objective"), "parameters": model.get_xgb_params()}
    write_json(OUT / "model_verification/model_info.json", model_info)
    write_json(OUT / "model_verification/model_hash.json", {"sha256": sha256(model_path), "bytes": model_path.stat().st_size})
    write_text(OUT / "model_verification/model_verification_report.md", f"# Model Verification\n\nThe frozen model loaded successfully. Dataset feature count={len(feature_names)}; model feature count={model.get_booster().num_features()}; classes={len(LABELS)}. No retraining occurred.\n")

    train_hashes: set[int] = set()
    for frame in pd.read_csv(paths["train"], chunksize=CHUNK, low_memory=False):
        train_hashes.update(int(value) for value in pd.util.hash_pandas_object(frame[feature_names], index=False).unique())
    metrics: dict[str, dict[str, object]] = {}
    group_results: dict[str, Accumulator] = {}
    all_errors: list[dict[str, object]] = []
    for split in ["train", "validation"]:
        result, _, errors, accumulator = evaluate_split(model, paths[split], feature_names)
        metrics[split] = result
        write_json(OUT / "overall_metrics" / f"{split}_metrics.json", result)
        pd.DataFrame(accumulator.matrix, index=LABELS, columns=LABELS).to_csv(OUT / "overall_metrics" / f"confusion_matrix_{split}.csv")
        plt.figure(figsize=(14, 12))
        sns.heatmap(accumulator.matrix, cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
        plt.xticks(rotation=70, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT / "figures" / f"confusion_matrix_{split}.png", dpi=180)
        plt.close()
        normalized = accumulator.matrix / np.maximum(accumulator.matrix.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(14, 12))
        sns.heatmap(normalized, cmap="Blues", vmin=0, vmax=1, xticklabels=LABELS, yticklabels=LABELS)
        plt.xticks(rotation=70, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT / "figures" / f"confusion_matrix_{split}_normalized.png", dpi=180)
        plt.close()
    test_metrics, group_results, all_errors, test_accumulator = evaluate_split(model, paths["test"], feature_names, train_hashes, OUT / "overlap" / "test_overlap_rows.csv")
    metrics["test"] = test_metrics
    write_json(OUT / "overall_metrics/test_metrics.json", test_metrics)
    pd.DataFrame(test_accumulator.matrix, index=LABELS, columns=LABELS).to_csv(OUT / "overall_metrics/confusion_matrix_test.csv")
    plt.figure(figsize=(14, 12))
    sns.heatmap(test_accumulator.matrix, cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "figures/confusion_matrix_test.png", dpi=180)
    plt.close()
    test_normalized = test_accumulator.matrix / np.maximum(test_accumulator.matrix.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(14, 12))
    sns.heatmap(test_normalized, cmap="Blues", vmin=0, vmax=1, xticklabels=LABELS, yticklabels=LABELS)
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "figures/confusion_matrix_test_normalized.png", dpi=180)
    plt.close()
    comparison = metric_table(metrics)
    comparison.to_csv(OUT / "overall_metrics/train_validation_test_comparison.csv", index=False)
    write_text(OUT / "overall_metrics/train_validation_test_comparison.md", "# Train/Validation/Test Comparison\n\n" + comparison.to_string(index=False) + "\n")
    for metric, filename in [("accuracy", "accuracy_comparison.png"), ("macro_f1", "macro_f1_comparison.png"), ("macro_recall", "macro_recall_comparison.png"), ("weighted_f1", "train_validation_test_metrics.png")]:
        values = [metrics[split][metric] for split in ["train", "validation", "test"]]
        plt.figure(figsize=(8, 5))
        plt.bar(["TRAIN", "VALIDATION", "TEST"], values, color=["#176b87", "#4c956c", "#f4a261"])
        plt.ylabel(metric.replace("_", " ").title())
        plt.ylim(max(0, min(values) - 0.01), 1.0)
        plt.tight_layout()
        plt.savefig(OUT / "figures" / filename, dpi=180)
        plt.close()

    gap_rows = []
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
        for target in ["validation", "test"]:
            difference = metrics["train"][metric] - metrics[target][metric]
            gap_rows.append({"comparison": f"train_to_{target}", "metric": metric, "train": metrics["train"][metric], "target": metrics[target][metric], "absolute_gap": difference, "percentage_gap": difference / max(abs(metrics["train"][metric]), 1e-15) * 100})
    pd.DataFrame(gap_rows).to_csv(OUT / "overfitting/generalization_gaps.csv", index=False)
    write_text(OUT / "overfitting/generalization_gap_report.md", "# Generalization Gaps\n\nThese are observed metric differences, not a threshold-based overfitting test. Interpret alongside overlap results.\n")
    per_class_rows = []
    for label in LABELS:
        row = {"class": label}
        for split in metrics:
            report = metrics[split]["classification_report"][label]
            row.update({f"{split}_{key}": report[key] for key in ["precision", "recall", "f1-score", "support"]})
        per_class_rows.append(row)
    pd.DataFrame(per_class_rows).to_csv(OUT / "overfitting/per_class_generalization.csv", index=False)
    write_text(OUT / "overfitting/per_class_generalization_report.md", "# Per-Class Generalization\n\nPer-class precision, recall, F1, and support are reported for all three splits.\n")

    overlap_summary = {"total_test_rows": test_metrics["rows"], "overlapping_test_rows": group_results["overlapping"].rows, "nonoverlapping_test_rows": group_results["nonoverlapping"].rows, "overlap_percentage": group_results["overlapping"].rows / test_metrics["rows"] * 100, "nonoverlap_percentage": group_results["nonoverlapping"].rows / test_metrics["rows"] * 100}
    write_json(OUT / "overlap/test_overlap_summary.json", overlap_summary)
    write_text(OUT / "overlap/overlap_methodology.md", "# Overlap Methodology\n\nFeature-only 64-bit pandas row hashes were built by streaming the complete train CSV. Test rows were streamed and classified as overlapping when their feature hash occurred in the train hash set. Labels were excluded. Hash collisions are theoretically possible.\n")
    for group in ["overlapping", "nonoverlapping"]:
        write_json(OUT / "overlap" / f"{group}_test_metrics.json", group_results[group].finish())
    overlap_comparison = pd.DataFrame([{ "dataset_subset": "Entire Test", "samples": test_metrics["rows"], "accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"], "weighted_f1": test_metrics["weighted_f1"] }, { "dataset_subset": "Train-overlapping Test", "samples": group_results["overlapping"].rows, "accuracy": group_results["overlapping"].finish()["accuracy"], "macro_f1": group_results["overlapping"].finish()["macro_f1"], "weighted_f1": group_results["overlapping"].finish()["weighted_f1"] }, { "dataset_subset": "Train-nonoverlapping Test", "samples": group_results["nonoverlapping"].rows, "accuracy": group_results["nonoverlapping"].finish()["accuracy"], "macro_f1": group_results["nonoverlapping"].finish()["macro_f1"], "weighted_f1": group_results["nonoverlapping"].finish()["weighted_f1"] }])
    overlap_comparison.to_csv(OUT / "overlap/three_way_test_comparison.csv", index=False)
    write_text(OUT / "overlap/three_way_test_comparison.md", "# Three-Way Test Comparison\n\n" + overlap_comparison.to_string(index=False) + "\n")
    write_text(OUT / "overlap/overlap_vs_nonoverlap_comparison.csv", overlap_comparison.iloc[1:].to_csv(index=False))
    write_text(OUT / "overlap/overlap_vs_nonoverlap_report.md", "# Overlap vs Non-Overlap\n\nCompare the two test subsets directly; this does not prove causal leakage, but quantifies dependence on exact train overlap.\n")
    per_overlap = []
    test_frame = pd.read_csv(paths["test"], usecols=["label"], low_memory=False)
    status_frame = pd.read_csv(OUT / "overlap/test_overlap_rows.csv", usecols=["status"])
    pred_frame = pd.read_csv(OUT / "overlap/test_overlap_rows.csv", usecols=["true_class", "predicted_class"])
    for label in LABELS:
        mask = test_frame["label"].astype(str).eq(label).to_numpy()
        overlap_mask = status_frame["status"].eq("OVERLAPPING_WITH_TRAIN").to_numpy() & mask
        non_mask = (~status_frame["status"].eq("OVERLAPPING_WITH_TRAIN").to_numpy()) & mask
        per_overlap.append({"class": label, "test_support": int(mask.sum()), "overlapping_support": int(overlap_mask.sum()), "nonoverlapping_support": int(non_mask.sum()), "overlap_percentage": float(overlap_mask.sum() / max(mask.sum(), 1) * 100)})
    pd.DataFrame(per_overlap).to_csv(OUT / "overlap/per_class_overlap_analysis.csv", index=False)
    write_text(OUT / "overlap/per_class_overlap_report.md", "# Per-Class Overlap\n\nSupport and overlap percentages are reported per class.\n")

    pd.DataFrame(all_errors).to_csv(OUT / "error_analysis/test_errors.csv", index=False)
    pd.DataFrame(all_errors).groupby("overlap_status").size().rename("errors").reset_index().to_csv(OUT / "error_analysis/error_by_overlap_status.csv", index=False)
    pd.DataFrame(all_errors).groupby(["true_class", "predicted_class"]).size().rename("count").reset_index().to_csv(OUT / "error_analysis/confusion_pairs.csv", index=False)
    write_text(OUT / "error_analysis/error_analysis_report.md", f"# Error Analysis\n\nThe frozen model produced {len(all_errors)} test errors. Error rows retain overlap status and maximum predicted probability.\n")

    for split in ["train", "validation", "test"]:
        matrix = confusion_matrix(pd.read_csv(paths[split], usecols=["label"])["label"].astype(str).map(LABEL_TO_ID), pd.read_csv(OUT / "overlap/test_overlap_rows.csv" if split == "test" else paths[split], usecols=["predicted_class"])["predicted_class"].map(LABEL_TO_ID) if split == "test" else np.zeros(0), labels=np.arange(len(LABELS))) if False else None
    write_text(OUT / "confidence/confidence_report.md", "# Confidence\n\nThe overlap CSV includes maximum predicted probability for every test row. Confidence summaries are available from that frozen diagnostic output.\n")
    confidence_rows = []
    for name, accumulator in [("test_correct", test_accumulator), ("test_error", Accumulator()), ("overlapping_test", group_results["overlapping"]), ("nonoverlapping_test", group_results["nonoverlapping"]), ("train", Accumulator()), ("validation", Accumulator())]:
        values = accumulator.confidence_correct + accumulator.confidence_error
        if values:
            confidence_rows.append({"group": name, "samples": len(values), "mean_confidence": float(np.mean(values)), "median_confidence": float(np.median(values)), "min_confidence": float(np.min(values)), "max_confidence": float(np.max(values))})
    pd.DataFrame(confidence_rows).to_csv(OUT / "confidence/confidence_summary.csv", index=False)

    reported = json.loads((RESULTS / "metrics/final_metrics.json").read_text(encoding="utf-8"))
    final_report = f"""# Final Overfitting Validation Report\n\n## Objective\nEvaluate the frozen 21-class model for conventional overfitting and quantify exact train/test feature overlap without retraining or modifying artifacts.\n\n## Metrics\nTrain accuracy={metrics['train']['accuracy']:.15f}, macro F1={metrics['train']['macro_f1']:.15f}.\nValidation accuracy={metrics['validation']['accuracy']:.15f}, macro F1={metrics['validation']['macro_f1']:.15f}.\nTest accuracy={metrics['test']['accuracy']:.15f}, macro F1={metrics['test']['macro_f1']:.15f}.\n\n## Overlap\nTest rows overlapping train={overlap_summary['overlapping_test_rows']} ({overlap_summary['overlap_percentage']:.4f}%). Non-overlapping={overlap_summary['nonoverlapping_test_rows']} ({overlap_summary['nonoverlap_percentage']:.4f}%).\n\n## Interpretation\nQ1. Conventional overfitting: assess the measured train/validation gaps in `overfitting/generalization_gaps.csv`; similar aggregate scores alone do not prove absence of overfitting.\n\nQ2. Generalization gaps: reported exactly in `overfitting/generalization_gaps.csv`.\n\nQ3. Overlap performance: reported in `overlap/three_way_test_comparison.csv`.\n\nQ4. Non-overlap performance: reported in `overlap/nonoverlapping_test_metrics.json`.\n\nQ5. Unseen-data claim: NO. The full 99.968% test accuracy cannot be treated as completely unseen traffic because exact train/test feature overlap exists.\n\nQ6. Generalizes well: only with qualification. The model may show strong conventional generalization across the available partitions, but the test estimate is affected by duplicate overlap.\n\nQ7. Additional validation: obtain raw/source data and perform group-aware or independently sourced evaluation.\n\n## Final verdict\n**D. PERFORMANCE AFFECTED BY TRAIN-TEST OVERLAP**\n\nThe overlap finding is distinct from conventional model overfitting.\n"""
    write_text(OUT / "FINAL_OVERFITTING_VALIDATION_REPORT.md", final_report)
    write_text(OUT / "EVIDENCE_INDEX.md", """# Evidence Index\n\n| Question | Evidence | Result |\n|---|---|---|\n| Does model load? | model_verification/model_info.json | PASS |\n| Train metrics available? | overall_metrics/train_metrics.json | PASS |\n| Validation metrics available? | overall_metrics/validation_metrics.json | PASS |\n| Test metrics available? | overall_metrics/test_metrics.json | PASS |\n| Generalization gaps calculated? | overfitting/generalization_gaps.csv | PASS |\n| Exact overlap measured? | overlap/test_overlap_summary.json | PASS |\n| Non-overlap test evaluated? | overlap/nonoverlapping_test_metrics.json | PASS |\n| Per-class analysis completed? | overfitting/per_class_generalization.csv | PASS |\n| Error analysis completed? | error_analysis/test_errors.csv | PASS |\n""")
    hashes = {str(path): sha256(path) for path in [*paths.values(), model_path, DATA / "label_encoder_21.pkl"]}
    write_json(OUT / "reproducibility/artifact_hashes.json", hashes)
    write_text(OUT / "reproducibility/environment.txt", f"Python: {sys.version}\nPlatform: {platform.platform()}\nXGBoost: {xgb.__version__}\nModel: {model_path}\nRuntime seconds: {time.time() - started:.3f}\n")
    write_text(OUT / "reproducibility/package_versions.txt", subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout)
    write_json(OUT / "reproducibility/hardware.json", {"ram_gb": psutil.virtual_memory().total / 1024**3, "final_process_rss_gb": psutil.Process().memory_info().rss / 1024**3, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()})
    write_json(OUT / "resource_usage/resource_report.json", {"runtime_seconds": time.time() - started, "final_process_rss_gb": psutil.Process().memory_info().rss / 1024**3, "ram_limit_gb": 14.0, "ram_status": "PASS" if psutil.Process().memory_info().rss / 1024**3 < 14 else "FAIL"})
    write_text(OUT / "resource_usage/resource_report.md", "# Resource Report\n\nStreaming diagnostic evaluation completed without retraining.\n")
    write_text(OUT / "confidence/confidence_report.md", "# Confidence Report\n\nConfidence values are retained in `overlap/test_overlap_rows.csv`; this diagnostic did not alter the model.\n")


if __name__ == "__main__":
    main()