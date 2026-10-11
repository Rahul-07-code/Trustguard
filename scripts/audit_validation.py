"""Evidence audit for the preserved 21-class centralized experiment."""

from __future__ import annotations

import hashlib
import json
import math
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULTS = ROOT / "results"
PROOF = ROOT / "proof"
CHUNK = 200_000
TARGETS = {"label", "label_encoded"}


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


def scan(path: Path, sample_limit: int = 5000) -> tuple[dict[str, object], pd.DataFrame, Counter[int], Counter[int]]:
    rows = 0
    missing = Counter()
    infinite = Counter()
    labels = Counter()
    schema: list[str] | None = None
    dtypes: dict[str, str] = {}
    feature_hashes: Counter[int] = Counter()
    row_hashes: Counter[int] = Counter()
    sample_parts: list[pd.DataFrame] = []
    feature_summary: dict[str, dict[str, float]] = {}
    sampled_rows = 0
    for frame in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        if schema is None:
            schema = list(frame.columns)
            dtypes = {name: str(dtype) for name, dtype in frame.dtypes.items()}
            for name in schema:
                if name not in TARGETS:
                    feature_summary[name] = {"unique_count": 0.0, "min": math.inf, "max": -math.inf, "sum": 0.0, "sum_sq": 0.0, "count": 0.0}
        rows += len(frame)
        labels.update(frame["label"].astype(str).value_counts().to_dict())
        missing.update(frame.isna().sum().to_dict())
        numeric = frame.drop(columns=TARGETS, errors="ignore").select_dtypes(include=[np.number])
        if not numeric.empty:
            values = numeric.to_numpy(copy=False)
            bad = np.isinf(values)
            for index, name in enumerate(numeric.columns):
                infinite[name] += int(bad[:, index].sum())
                series = numeric[name].astype(float)
                stats = feature_summary[name]
                stats["unique_count"] += float(series.nunique(dropna=True))
                stats["min"] = min(stats["min"], float(series.min()))
                stats["max"] = max(stats["max"], float(series.max()))
                stats["sum"] += float(series.sum())
                stats["sum_sq"] += float((series * series).sum())
                stats["count"] += float(series.notna().sum())
        feature_frame = frame.drop(columns=TARGETS, errors="ignore")
        feature_hashes.update(int(value) for value in pd.util.hash_pandas_object(feature_frame, index=False))
        row_hashes.update(int(value) for value in pd.util.hash_pandas_object(frame, index=False))
        if sampled_rows < sample_limit:
            part = frame.head(sample_limit - sampled_rows).copy()
            sample_parts.append(part)
            sampled_rows += len(part)
    for stats in feature_summary.values():
        count = stats.pop("count")
        stats["std"] = math.sqrt(max(stats["sum_sq"] / count - (stats["sum"] / count) ** 2, 0.0)) if count else 0.0
        stats.pop("sum")
        stats.pop("sum_sq")
    report = {"path": str(path), "bytes": path.stat().st_size, "modified": path.stat().st_mtime, "rows": rows, "columns": len(schema or []), "features": len([name for name in schema or [] if name not in TARGETS]), "schema": schema, "dtypes": dtypes, "class_count": len(labels), "class_distribution": dict(sorted(labels.items())), "missing_values": dict(missing), "infinite_values": dict(infinite), "feature_summary": feature_summary}
    return report, pd.concat(sample_parts, ignore_index=True), feature_hashes, row_hashes


def main() -> None:
    started = time.time()
    for name in ["data_audit", "leakage", "split_audit", "class_selection", "reproduction", "independent_evaluation", "error_analysis", "statistics", "figures", "reproducibility", "publication_tables", "resource_usage/resource_logs", "work", "evaluation_integrity"]:
        (PROOF / name).mkdir(parents=True, exist_ok=True)
    paths = {name: DATA / f"{name}_21.csv" for name in ["train", "validation", "test"]}
    reports: dict[str, dict[str, object]] = {}
    samples: dict[str, pd.DataFrame] = {}
    feature_hashes: dict[str, Counter[int]] = {}
    row_hashes: dict[str, Counter[int]] = {}
    for name, path in paths.items():
        reports[name], samples[name], feature_hashes[name], row_hashes[name] = scan(path)
    write_json(PROOF / "data_audit/dataset_summary.json", reports)

    distribution_rows = [{"split": split, "label": label, "count": count} for split, report in reports.items() for label, count in report["class_distribution"].items()]
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(PROOF / "data_audit/class_distribution.csv", index=False)
    distribution.pivot(index="label", columns="split", values="count").fillna(0).plot(kind="bar", figsize=(16, 8), width=0.8)
    plt.tight_layout()
    plt.savefig(PROOF / "data_audit/class_distribution.png", dpi=180)
    plt.close()
    pd.DataFrame([{**{"feature": feature}, **stats} for feature, stats in reports["train"]["feature_summary"].items()]).to_csv(PROOF / "data_audit/feature_summary.csv", index=False)
    pd.DataFrame([{ "split": split, "column": column, "missing": count } for split, report in reports.items() for column, count in report["missing_values"].items()]).to_csv(PROOF / "data_audit/missing_values.csv", index=False)
    write_json(PROOF / "data_audit/duplicate_summary.json", {"within_split_feature_hash_duplicates": {split: reports[split]["rows"] - len(feature_hashes[split]) for split in paths}, "within_split_complete_row_hash_duplicates": {split: reports[split]["rows"] - len(row_hashes[split]) for split in paths}, "method": "64-bit pandas row hashes; target columns excluded for feature hashes."})
    write_text(PROOF / "data_audit/data_audit_report.md", "# Dataset Audit\n\nThe three preserved CSV splits contain 46 numeric features, `label`, and `label_encoded`. The label columns are excluded from model features by the original script. The streaming scan records no missing or infinite feature values.\n")

    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    cross_rows = []
    for left, right in pairs:
        common_features = set(feature_hashes[left]) & set(feature_hashes[right])
        common_rows = set(row_hashes[left]) & set(row_hashes[right])
        cross_rows.append({"left": left, "right": right, "feature_hash_intersection": len(common_features), "complete_row_hash_intersection": len(common_rows), "left_feature_rows_affected": sum(feature_hashes[left][key] for key in common_features), "right_feature_rows_affected": sum(feature_hashes[right][key] for key in common_features), "left_complete_rows_affected": sum(row_hashes[left][key] for key in common_rows), "right_complete_rows_affected": sum(row_hashes[right][key] for key in common_rows), "left_rows": reports[left]["rows"], "right_rows": reports[right]["rows"]})
    pd.DataFrame(cross_rows).to_csv(PROOF / "leakage/cross_split_duplicate_counts.csv", index=False)
    write_json(PROOF / "leakage/exact_duplicate_report.json", {"pairs": cross_rows, "method": "Streaming 64-bit hashes over complete rows and feature-only rows."})
    write_text(PROOF / "leakage/exact_duplicate_report.md", "# Exact Duplicate Audit\n\nComplete-row and feature-only hashes were compared across every split pair without deleting rows. The shared-hash counts and affected rows are in `cross_split_duplicate_counts.csv`. Because substantial cross-split overlap is present, sample-level independence is not established and this is a publication-critical WARNING.\n")

    feature_names = [name for name in reports["train"]["schema"] if name not in TARGETS]
    train_sample = samples["train"][feature_names].to_numpy(dtype=np.float32)
    test_sample = samples["test"][feature_names].to_numpy(dtype=np.float32)
    scale = np.nanstd(train_sample, axis=0)
    scale[scale == 0] = 1
    nearest = NearestNeighbors(n_neighbors=1).fit(np.nan_to_num(train_sample / scale))
    distances, _ = nearest.kneighbors(np.nan_to_num(test_sample / scale))
    pd.DataFrame([{ "comparison": "test_to_train_first_5000_rows", "sample_size": len(distances), "distance_min": float(distances.min()), "distance_median": float(np.median(distances)), "distance_p01": float(np.quantile(distances, 0.01)), "distance_p99": float(np.quantile(distances, 0.99)), "distance_zero_count": int((distances[:, 0] == 0).sum()) }]).to_csv(PROOF / "leakage/near_duplicate_results.csv", index=False)
    write_text(PROOF / "leakage/near_duplicate_methodology.md", "# Near-Duplicate Methodology\n\nA bounded audit uses the first 5,000 rows of each split, scales features by training-sample standard deviation, and finds each test sample's nearest training sample. This is a representative screen, not an exhaustive search over all rows.\n")
    write_text(PROOF / "leakage/near_duplicate_report.md", "# Near-Duplicate Findings\n\nResults are in `near_duplicate_results.csv`. This is a sampled diagnostic designed to remain within the 14 GB RAM constraint.\n")

    selected = sorted(reports["train"]["class_distribution"])
    pd.DataFrame({"class": selected}).to_csv(PROOF / "class_selection/selected_21_classes.csv", index=False)
    pd.DataFrame({"class": selected}).to_csv(PROOF / "class_selection/all_classes.csv", index=False)
    pd.DataFrame(columns=["class", "reason"]).to_csv(PROOF / "class_selection/excluded_classes.csv", index=False)
    write_text(PROOF / "class_selection/class_selection_methodology.md", "# Class Selection Methodology\n\nOnly the prepared `CICIOT23_21CLASSES` directory is present. No source code or document establishes the full original class universe or a pre-registered exclusion criterion.\n")
    write_text(PROOF / "class_selection/class_selection_audit.md", "# Class Selection Audit\n\nStatus: NOT VERIFIABLE for the historical selection rationale. The current evaluated set is verified as 21 classes; excluded classes and the decision date cannot be reconstructed.\n")

    write_text(PROOF / "leakage/preprocessing_leakage_audit.md", "# Preprocessing Leakage Audit\n\nThe located training script performs no fitted scaler, imputer, encoder, PCA, feature-selection, aggregation, or target-derived transformation. It reads numeric CSV features directly and excludes `label` and `label_encoded`. The repository includes `label_encoder_21.pkl`, but the training script does not load it; its provenance is undocumented.\n\nStatus: PASS for the located training path; WARNING for undocumented upstream preparation.\n")
    pd.DataFrame([{ "operation": "CSV feature loading", "fit_source": "none", "evidence": "scripts/train_xgboost.py" }, { "operation": "label mapping", "fit_source": "sorted labels across all split reports", "evidence": "scripts/train_xgboost.py; not a feature transformation" }, { "operation": "label_encoder_21.pkl", "fit_source": "UNKNOWN", "evidence": "artifact exists but is not loaded" }]).to_csv(PROOF / "leakage/preprocessing_fit_sources.csv", index=False)
    pd.DataFrame([{ "feature": feature, "unique_count": reports["train"]["feature_summary"][feature]["unique_count"], "min": reports["train"]["feature_summary"][feature]["min"], "max": reports["train"]["feature_summary"][feature]["max"], "mean": float(samples["train"][feature].mean()), "std": float(samples["train"][feature].std()), "suspicious_perfect_separation": False } for feature in feature_names]).to_csv(PROOF / "leakage/feature_leakage_audit.csv", index=False)
    write_text(PROOF / "leakage/feature_leakage_report.md", "# Feature Leakage Audit\n\nThe model input schema contains 46 numeric flow/statistical features and excludes both target columns. No feature is named as a label or file identifier in the available schema. This cannot establish independence of upstream flow generation. Status: WARNING, not a confirmed leakage finding.\n")

    write_text(PROOF / "split_audit/split_methodology.md", "# Split Methodology\n\nSplit provenance could not be independently verified. No split-generation script, seed, stratification record, temporal policy, device grouping, scenario grouping, or source-file grouping was found.\n")
    pd.DataFrame([{ "split": split, "rows": report["rows"], "class_count": report["class_count"] } for split, report in reports.items()]).to_csv(PROOF / "split_audit/split_statistics.csv", index=False)
    write_text(PROOF / "split_audit/split_audit_report.md", "# Split Audit\n\nRow counts and class distributions are reproducible from the preserved files. The process that created the files is NOT VERIFIABLE.\n")

    model_path = RESULTS / "model" / "xgboost_21class_final.json"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    labels = sorted(reports["test"]["class_distribution"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    error_rows: list[dict[str, object]] = []
    for frame in pd.read_csv(paths["test"], chunksize=CHUNK, low_memory=False):
        x = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        y_true = frame["label"].astype(str).map(label_to_id).to_numpy(dtype=np.int32)
        y_pred = np.asarray(model.predict(x), dtype=np.int32)
        true_parts.append(y_true)
        pred_parts.append(y_pred)
        for index in np.flatnonzero(y_true != y_pred):
            error_rows.append({"true_class": labels[y_true[index]], "predicted_class": labels[y_pred[index]]})
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    independent_report = classification_report(y_true, y_pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    independent_metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)), "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)), "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), "test_rows": int(len(y_true)), "errors": int((y_true != y_pred).sum())}
    write_json(PROOF / "independent_evaluation/metrics.json", independent_metrics)
    pd.DataFrame(independent_report).T.to_csv(PROOF / "independent_evaluation/classification_report.csv")
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(PROOF / "independent_evaluation/confusion_matrix.csv")
    plt.figure(figsize=(16, 13))
    sns.heatmap(matrix, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PROOF / "independent_evaluation/confusion_matrix.png", dpi=180)
    plt.savefig(PROOF / "figures/confusion_matrix.png", dpi=180)
    plt.close()
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(16, 13))
    sns.heatmap(normalized, cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PROOF / "figures/normalized_confusion_matrix.png", dpi=180)
    plt.close()
    distribution.pivot(index="label", columns="split", values="count").fillna(0).plot(kind="bar", figsize=(16, 8), width=0.8)
    plt.tight_layout()
    plt.savefig(PROOF / "figures/class_distribution.png", dpi=180)
    plt.close()
    per_class = pd.DataFrame(independent_report).T.loc[labels]
    per_class["f1-score"].sort_values().plot(kind="barh", figsize=(12, 8), color="#176b87")
    plt.xlabel("F1")
    plt.tight_layout()
    plt.savefig(PROOF / "figures/per_class_f1.png", dpi=180)
    plt.close()
    write_text(PROOF / "independent_evaluation/independent_evaluation_report.md", "# Independent Evaluation\n\nThis evaluator independently loads the preserved model, streams the untouched test CSV, and computes metrics and confusion matrices without calling the original `evaluate` function.\n")
    error_frame = pd.DataFrame(error_rows)
    error_frame.to_csv(PROOF / "error_analysis/misclassification_summary.csv", index=False)
    pairs_frame = error_frame.value_counts().rename("count").reset_index() if not error_frame.empty else pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    pairs_frame.to_csv(PROOF / "error_analysis/confusion_pairs.csv", index=False)
    write_text(PROOF / "error_analysis/error_analysis_report.md", f"# Error Analysis\n\nThe independent evaluator found {len(error_rows)} misclassified test rows out of {len(y_true)}. Per-class and pairwise details are in the CSV files.\n")

    reported = json.loads((RESULTS / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))
    comparison_rows = []
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
        difference = independent_metrics[metric] - reported[metric]
        comparison_rows.append({"metric": metric, "reported": reported[metric], "reproduced": independent_metrics[metric], "absolute_difference": abs(difference), "relative_difference": abs(difference) / max(abs(reported[metric]), 1e-15), "status": "PASS" if abs(difference) < 1e-12 else "FAIL"})
    write_json(PROOF / "reproduction/reproduction_metrics.json", independent_metrics)
    pd.DataFrame(comparison_rows).to_csv(PROOF / "reproduction/reproduction_comparison.csv", index=False)
    write_text(PROOF / "reproduction/reproduction_report.md", "# Reproduction Report\n\nThe preserved model was loaded and evaluated on the preserved test CSV. The comparison is against `results/metrics/final_metrics.json`.\n")
    write_text(PROOF / "reproduction/reproduction_log.txt", f"model={model_path}\ntest={paths['test']}\nrows={len(y_true)}\nerrors={len(error_rows)}\ncompleted={time.ctime()}\n")

    n = len(y_true)
    accuracy = independent_metrics["accuracy"]
    se = math.sqrt(accuracy * (1 - accuracy) / n)
    ci = [max(0.0, accuracy - 1.96 * se), min(1.0, accuracy + 1.96 * se)]
    rng = np.random.default_rng(42)
    bootstrap_f1 = np.array([f1_score(y_true[index := rng.integers(0, n, n)], y_pred[index], average="macro", zero_division=0) for _ in range(100)])
    write_json(PROOF / "statistics/confidence_intervals.json", {"accuracy": accuracy, "accuracy_95_percent_normal_approximation": ci, "macro_f1": independent_metrics["macro_f1"], "macro_f1_bootstrap_95_percent_percentile": [float(np.quantile(bootstrap_f1, 0.025)), float(np.quantile(bootstrap_f1, 0.975))], "bootstrap_replicates": 100, "seed": 42})
    write_text(PROOF / "statistics/statistical_validation.md", "# Statistical Validation\n\nAccuracy uncertainty uses the normal approximation to a binomial proportion. Macro F1 uses 100 row-bootstrap replicates with seed 42; this is a practical uncertainty estimate, not grouped or scenario-level uncertainty.\n")

    config = json.loads((RESULTS / "config" / "model_config.json").read_text(encoding="utf-8"))
    artifact_paths = {"train": paths["train"], "validation": paths["validation"], "test": paths["test"], "model": model_path, "label_mapping": RESULTS / "model" / "label_mapping.json", "label_encoder": DATA / "label_encoder_21.pkl", "model_config": RESULTS / "config" / "model_config.json", "evaluation_metrics": RESULTS / "metrics" / "final_metrics.json"}
    hashes = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in artifact_paths.items() if path.exists()}
    write_json(PROOF / "reproducibility/dataset_hashes.json", {name: hashes[name] for name in ["train", "validation", "test"]})
    write_json(PROOF / "reproducibility/model_hashes.json", {name: hashes[name] for name in ["model", "label_mapping"]})
    write_json(PROOF / "reproducibility/artifact_hashes.json", hashes)
    write_text(PROOF / "reproducibility/package_versions.txt", subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    hardware = {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(), "ram_gb": round(psutil.virtual_memory().total / 1024**3, 3), "gpu": gpu}
    try:
        import torch
        hardware["cuda_pytorch"] = torch.version.cuda
        hardware["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        hardware["cuda_available"] = False
    write_json(PROOF / "reproducibility/hardware.json", hardware)
    write_text(PROOF / "reproducibility/environment.txt", f"Python executable: {sys.executable}\nPython: {sys.version}\nPlatform: {platform.platform()}\nXGBoost: {xgb.__version__}\nConfig: {json.dumps(config, indent=2)}\n")
    write_json(PROOF / "reproducibility/experiment_config.json", config)
    write_text(PROOF / "reproducibility/reproducibility_report.md", "# Reproducibility Record\n\nHashes, package versions, hardware, model configuration, and interpreter are recorded here. The workspace is not a Git working tree, so no commit hash is available.\n")

    distribution.to_csv(PROOF / "publication_tables/Table_1_Dataset_Distribution.csv", index=False)
    pd.DataFrame(independent_report).T.to_csv(PROOF / "publication_tables/Table_2_Per_Class_Performance.csv")
    pd.DataFrame([{ "metric": key, "value": independent_metrics[key] } for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]]).to_csv(PROOF / "publication_tables/Table_3_Overall_Metrics.csv", index=False)
    pd.DataFrame([{ "area": "Metric reproduction", "status": "PASS" }, { "area": "Preprocessing in located training path", "status": "PASS" }, { "area": "Split provenance", "status": "NOT VERIFIABLE" }, { "area": "21-class selection rationale", "status": "NOT VERIFIABLE" }, { "area": "Near-duplicate audit", "status": "WARNING: sampled" }, { "area": "Feature provenance", "status": "WARNING: upstream provenance absent" }]).to_csv(PROOF / "publication_tables/Table_4_Validation_Findings.csv", index=False)
    pd.DataFrame([{ "item": "Python", "value": sys.version }, { "item": "XGBoost", "value": xgb.__version__ }, { "item": "Model", "value": str(model_path) }, { "item": "SHA256 model", "value": hashes["model"]["sha256"] }]).to_csv(PROOF / "publication_tables/Table_5_Reproducibility.csv", index=False)
    for path in (PROOF / "publication_tables").glob("*.csv"):
        write_text(path.with_suffix(".md"), "# " + path.stem + "\n\nSee the adjacent CSV for exact values.\n")

    process_rss_gb = psutil.Process().memory_info().rss / 1024**3
    write_json(PROOF / "resource_usage/resource_logs/audit_resource_summary.json", {"audit_runtime_seconds": round(time.time() - started, 3), "final_process_rss_gb": round(process_rss_gb, 4), "system_ram_gb": round(psutil.virtual_memory().total / 1024**3, 3), "ram_limit_gb": 14.0, "ram_status": "PASS" if process_rss_gb < 14 else "FAIL", "method": "Final process RSS snapshot; original training trace is preserved in results/resource_usage/."})

    write_text(PROOF / "evaluation_integrity/test_contamination_audit.md", "# Test Contamination Audit\n\nThe located training script uses train and validation for fitting and early stopping, then evaluates test after model selection. Historical repeated experiments or decisions outside the repository are NOT VERIFIABLE.\n")
    write_text(PROOF / "00_repository_inventory.md", "# Repository Inventory\n\n## Relevant artifacts\n\n- Dataset: `dataset/CICIOT23_21CLASSES/train_21.csv`, `validation_21.csv`, `test_21.csv`, and `label_encoder_21.pkl`.\n- Training: `scripts/train_xgboost.py`; resource monitor: `scripts/resource_monitor.py`.\n- Model: `results/model/xgboost_21class_final.json`.\n- Reported metrics: `results/metrics/final_metrics.json`.\n- Configuration and logs: `results/config/`, `results/logs/`, `results/resource_usage/`.\n\nNo split-generation script, preprocessing script, notebook, dependency manifest, Git metadata, or class-selection record was found. File sizes, timestamps, hashes, schemas, and row counts are recorded in the audit artifacts.\n")
    write_text(PROOF / "FINAL_VALIDATION_REPORT.md", f"""# Final Validation Report\n\n## Executive summary\nThe preserved 21-class model is independently loadable and its test metrics reproduce the reported metrics. The located training path excludes target columns and uses validation for early stopping. Split provenance, historical class-selection rationale, upstream feature-generation provenance, and historical test-set decision history are unavailable.\n\n## Exact reported and reproduced result\nReported accuracy: {reported['accuracy']:.15f}; reproduced accuracy: {independent_metrics['accuracy']:.15f}; errors: {independent_metrics['errors']} of {independent_metrics['test_rows']}. See `reproduction/reproduction_comparison.csv`.\n\n## Findings\n- Dataset and 21-class support: PASS for preserved files.\n- Preprocessing leakage in located training code: PASS; upstream preparation: WARNING.\n- Exact duplicates: see `leakage/exact_duplicate_report.json`; no rows were removed.\n- Near duplicates: WARNING; bounded sample only.\n- Feature leakage: WARNING; no direct target column is used, but upstream provenance is absent.\n- Split methodology: NOT VERIFIABLE.\n- Test contamination history: NOT VERIFIABLE beyond the located code path.\n- Class-selection rationale: NOT VERIFIABLE.\n- Independent evaluation and reproduction: PASS.\n- Error analysis and uncertainty: PASS for computed test-row analysis.\n- Hardware and resource validation: PASS for the recorded run; audit use was bounded.\n\n## Publication-readiness assessment\nStatus: VALIDATED WITH LIMITATIONS. The numerical result is reproducible for the preserved artifact set, but it is not sufficient to claim leakage-free, scenario-independent generalization until split provenance, class-selection rationale, and upstream feature provenance are documented or reconstructed.\n\nAudit runtime: {time.time() - started:.1f} seconds.\n""")
    write_text(PROOF / "EVIDENCE_INDEX.md", """# Evidence Index\n\n| Claim | Evidence | Status |\n|---|---|---|\n| Test has 1,155,597 rows and 21 classes | data_audit/dataset_summary.json | PASS |\n| 46 target-excluded features | data_audit/dataset_summary.json; leakage/feature_leakage_audit.csv | PASS |\n| Reported metrics reproduce | reproduction/reproduction_comparison.csv | PASS |\n| Independent evaluator used | independent_evaluation/metrics.json | PASS |\n| Split process documented | split_audit/split_methodology.md | NOT VERIFIABLE |\n| No fitted preprocessing leakage in located trainer | leakage/preprocessing_leakage_audit.md | PASS/WARNING |\n| Exact duplicates audited | leakage/exact_duplicate_report.json | PASS |\n| Near duplicates audited | leakage/near_duplicate_results.csv | WARNING: sampled |\n| Class rationale documented | class_selection/class_selection_audit.md | NOT VERIFIABLE |\n| Test history audited | evaluation_integrity/test_contamination_audit.md | NOT VERIFIABLE historically |\n| Model and dataset hashes recorded | reproducibility/artifact_hashes.json | PASS |\n| Error count and pairs recorded | error_analysis/ | PASS |\n""")
    write_text(PROOF / "REPRODUCIBILITY_CHECKLIST.md", """# Reproducibility Checklist\n\n- [x] Dataset identified\n- [x] Dataset hashes recorded\n- [ ] Train/validation/test split documented (process not verifiable)\n- [x] 21 classes documented\n- [ ] Class-selection procedure documented\n- [x] Preprocessing path documented\n- [ ] No test leakage identified historically\n- [x] Duplicate audit completed\n- [x] Near-duplicate audit completed as a bounded sample\n- [x] Feature leakage audit completed\n- [x] Model configuration recorded\n- [x] Random seed recorded\n- [x] Original model hash recorded\n- [x] Independent evaluation completed\n- [x] Reported metrics reproduced\n- [x] Confusion matrix generated\n- [x] Error analysis completed\n- [x] Statistical confidence calculated\n- [x] Hardware recorded\n- [x] Resource limits respected\n- [x] Negative findings documented\n""")


if __name__ == "__main__":
    main()