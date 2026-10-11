"""Centralized CUDA XGBoost experiment for the prepared CICIoT23 dataset."""

from __future__ import annotations

import gc
import json
import logging
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

from resource_monitor import ResourceMonitor

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULTS = ROOT / "results"
SEED = 42
FEATURES: list[str] | None = None
TARGET_COLUMNS = {"label", "label_encoded"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def scan_split(path: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    rows = 0
    missing = 0
    infinite = 0
    schema: list[str] | None = None
    dtypes: dict[str, str] = {}
    non_numeric: set[str] = set()
    for frame in pd.read_csv(path, chunksize=250_000, low_memory=False):
        if schema is None:
            schema = list(frame.columns)
            dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        rows += len(frame)
        missing += int(frame.isna().sum().sum())
        features = frame.drop(columns=TARGET_COLUMNS, errors="ignore")
        numeric = features.select_dtypes(include=[np.number])
        non_numeric.update(features.select_dtypes(exclude=[np.number]).columns)
        infinite += int(np.isinf(numeric.to_numpy(copy=False)).sum())
        for label, count in frame["label"].value_counts().items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    return {
        "path": str(path), "file_size_bytes": path.stat().st_size, "rows": rows,
        "columns": len(schema or []), "feature_count": len([column for column in schema or [] if column not in TARGET_COLUMNS]),
        "schema": schema, "dtypes": dtypes, "class_count": len(counts),
        "class_distribution": dict(sorted(counts.items())), "missing_values": missing,
        "infinite_values": infinite, "non_numeric_features": sorted(non_numeric),
    }


def load_split(path: Path, label_mapping: dict[str, int] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    global FEATURES
    frame = pd.read_csv(path, low_memory=False)
    if "label" not in frame.columns:
        raise ValueError(f"Missing label column in {path}")
    current_features = [column for column in frame.columns if column not in TARGET_COLUMNS]
    if FEATURES is None:
        FEATURES = current_features
    elif current_features != FEATURES:
        raise ValueError(f"Feature schema mismatch in {path}")
    if frame[FEATURES].select_dtypes(exclude=[np.number]).shape[1] != 0:
        raise ValueError(f"Non-numeric feature found in {path}")
    if frame[FEATURES].isna().any().any() or np.isinf(frame[FEATURES].to_numpy()).any():
        raise ValueError(f"Missing or infinite feature value found in {path}")
    labels = frame["label"].astype(str)
    if label_mapping is None:
        label_mapping = {label: index for index, label in enumerate(sorted(labels.unique()))}
    unknown = sorted(set(labels) - set(label_mapping))
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {unknown}")
    X = frame[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = labels.map(label_mapping).to_numpy(dtype=np.int32, copy=True)
    del frame, labels
    gc.collect()
    return X, y, FEATURES


def save_distribution(reports: dict[str, dict[str, object]]) -> None:
    rows = [{"split": split, "label": label, "count": count} for split, report in reports.items() for label, count in report["class_distribution"].items()]
    distribution = pd.DataFrame(rows)
    distribution.to_csv(RESULTS / "class_distribution.csv", index=False)
    pivot = distribution.pivot(index="label", columns="split", values="count").fillna(0)
    plt.figure(figsize=(16, 9), dpi=180)
    pivot.plot(kind="bar", ax=plt.gca(), width=0.82)
    plt.title("CICIoT23 Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "plots" / "class_distribution.png")
    plt.close()


def save_bar(values: pd.Series, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(15, 7), dpi=180)
    values.sort_values().plot(kind="barh", color="#176b87")
    plt.title(title)
    plt.xlabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, object]:
    predicted = np.argmax(probabilities, axis=1)
    report = classification_report(y_true, predicted, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, predicted, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, predicted, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "class_count": len(class_names), "classification_report": report,
    }
    pd.DataFrame(report).T.to_csv(RESULTS / "metrics" / "classification_report.csv")
    write_json(RESULTS / "metrics" / "classification_report.json", report)
    write_json(RESULTS / "metrics" / "final_metrics.json", metrics)
    return metrics


def save_evaluation_plots(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], report: dict[str, object]) -> None:
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=np.arange(len(class_names)))
    for normalized, filename, title in [(False, "confusion_matrix.png", "Confusion Matrix"), (True, "confusion_matrix_normalized.png", "Normalized Confusion Matrix")]:
        values = matrix.astype(float)
        if normalized:
            values = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(18, 15), dpi=180)
        sns.heatmap(values, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS / "plots" / filename)
        plt.close()
    class_frame = pd.DataFrame(report).T.loc[class_names]
    save_bar(class_frame["f1-score"], "Per-Class F1 Score", "F1", RESULTS / "plots" / "per_class_f1.png")
    save_bar(class_frame["precision"], "Per-Class Precision", "Precision", RESULTS / "plots" / "per_class_precision.png")
    save_bar(class_frame["recall"], "Per-Class Recall", "Recall", RESULTS / "plots" / "per_class_recall.png")


def save_curves(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> None:
    y_one_hot = np.eye(len(class_names), dtype=np.int8)[y_true]
    roc_rows, pr_rows = [], []
    plt.figure(figsize=(10, 8), dpi=180)
    for index, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_one_hot[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(y_one_hot[:, index], probabilities[:, index])
        roc_rows.append({"label": name, "roc_auc": float(roc_auc_score(y_one_hot[:, index], probabilities[:, index]))})
        pr_rows.append({"label": name, "average_precision": float(average_precision_score(y_one_hot[:, index], probabilities[:, index]))})
        plt.plot(fpr, tpr, alpha=0.18, linewidth=0.7)
    macro_fpr = np.unique(np.concatenate([roc_curve(y_one_hot[:, i], probabilities[:, i])[0] for i in range(len(class_names))]))
    macro_tpr = np.mean([np.interp(macro_fpr, roc_curve(y_one_hot[:, i], probabilities[:, i])[0], roc_curve(y_one_hot[:, i], probabilities[:, i])[1]) for i in range(len(class_names))], axis=0)
    plt.plot(macro_fpr, macro_tpr, color="black", linewidth=2, label="Macro average")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.title("Multiclass ROC-AUC (one-vs-rest)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "plots" / "multiclass_roc_auc.png")
    plt.close()
    pd.DataFrame(roc_rows).to_csv(RESULTS / "metrics" / "per_class_roc_auc.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(RESULTS / "metrics" / "per_class_pr_metrics.csv", index=False)
    plt.figure(figsize=(10, 8), dpi=180)
    for index in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_one_hot[:, index], probabilities[:, index])
        plt.plot(recall, precision, alpha=0.18, linewidth=0.7)
    recall_grid = np.linspace(0, 1, 500)
    macro_precision = np.mean([np.interp(recall_grid, precision_recall_curve(y_one_hot[:, i], probabilities[:, i])[1][::-1], precision_recall_curve(y_one_hot[:, i], probabilities[:, i])[0][::-1]) for i in range(len(class_names))], axis=0)
    plt.plot(recall_grid, macro_precision, color="black", linewidth=2, label="Macro average")
    plt.title("Multiclass Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "plots" / "multiclass_precision_recall.png")
    plt.close()
    write_json(RESULTS / "metrics" / "roc_pr_summary.json", {"macro_roc_auc": float(roc_auc_score(y_one_hot, probabilities, average="macro", multi_class="ovr")), "weighted_roc_auc": float(roc_auc_score(y_one_hot, probabilities, average="weighted", multi_class="ovr"))})


def main() -> None:
    start = time.perf_counter()
    for folder in ["model", "metrics", "plots", "config", "resource_usage", "logs"]:
        (RESULTS / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESULTS / "logs" / "training.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Experiment started")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    gpu_name = torch.cuda.get_device_name(0)
    build_info = xgb.build_info()
    if not build_info.get("USE_CUDA"):
        raise RuntimeError("Installed XGBoost was not built with CUDA")
    paths = {"train": DATA_ROOT / "train_21.csv", "validation": DATA_ROOT / "validation_21.csv", "test": DATA_ROOT / "test_21.csv"}
    reports = {split: scan_split(path) for split, path in paths.items()}
    class_names = sorted(set(reports["train"]["class_distribution"]) | set(reports["validation"]["class_distribution"]) | set(reports["test"]["class_distribution"]))
    class_count = len(class_names)
    if class_count != 21:
        raise RuntimeError(f"Expected the selected dataset to contain 21 classes; found {class_count}")
    label_mapping = {name: index for index, name in enumerate(class_names)}
    validation_report = {"experiment_class_count": class_count, "dataset_class_count": class_count, "class_names": class_names, "label_mapping": label_mapping, "splits": reports, "test_used_for_tuning": False, "note": "Training uses the prepared CICIOT23 21-class dataset."}
    write_json(RESULTS / "dataset_validation_report.json", validation_report)
    save_distribution(reports)
    X_train, y_train, features = load_split(paths["train"], label_mapping)
    X_validation, y_validation, _ = load_split(paths["validation"], label_mapping)
    monitor = ResourceMonitor(RESULTS / "resource_usage" / "resource_usage.csv")
    monitor.start()
    params = {"objective": "multi:softprob", "num_class": class_count, "n_estimators": 1500, "max_depth": 8, "learning_rate": 0.08, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "max_bin": 256, "tree_method": "hist", "device": "cuda", "eval_metric": "mlogloss", "random_state": SEED, "n_jobs": 4, "early_stopping_rounds": 80}
    logger.info("GPU=%s XGBoost=%s build=%s params=%s", gpu_name, xgb.__version__, build_info, params)
    model = xgb.XGBClassifier(**params)
    train_start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_validation, y_validation)], verbose=50)
    training_seconds = time.perf_counter() - train_start
    history = pd.DataFrame(model.evals_result().get("validation_0", {})).join(pd.DataFrame(model.evals_result().get("validation_1", {})), lsuffix="_train", rsuffix="_validation")
    history.index.name = "iteration"
    history.to_csv(RESULTS / "metrics" / "training_history.csv")
    history.plot(figsize=(11, 6), title="Training and Validation Metric")
    plt.xlabel("Boosting round")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(RESULTS / "plots" / "training_validation_curve.png", dpi=180)
    plt.close()
    del X_train, y_train, X_validation, y_validation
    gc.collect()
    X_test, y_test, _ = load_split(paths["test"], label_mapping)
    eval_start = time.perf_counter()
    probabilities = model.predict_proba(X_test)
    metrics = evaluate(y_test, probabilities, class_names)
    evaluation_seconds = time.perf_counter() - eval_start
    save_evaluation_plots(y_test, probabilities, class_names, metrics["classification_report"])
    save_curves(y_test, probabilities, class_names)
    booster = model.get_booster()
    importance = pd.DataFrame({"feature": features, "gain": [booster.get_score(importance_type="gain").get(feature, 0.0) for feature in features], "weight": [booster.get_score(importance_type="weight").get(feature, 0.0) for feature in features], "cover": [booster.get_score(importance_type="cover").get(feature, 0.0) for feature in features]})
    importance.to_csv(RESULTS / "metrics" / "feature_importance.csv", index=False)
    for kind in ["gain", "weight", "cover"]:
        save_bar(importance.set_index("feature")[kind].nlargest(30), f"Top 30 Feature Importance ({kind})", kind.title(), RESULTS / "plots" / f"feature_importance_{kind}.png")
    model_path = RESULTS / "model" / f"xgboost_{class_count}class_final.json"
    model.save_model(model_path)
    reload_model = xgb.XGBClassifier()
    reload_model.load_model(model_path)
    subset = min(1000, len(X_test))
    reload_match = bool(np.array_equal(np.argmax(model.predict_proba(X_test[:subset]), axis=1), np.argmax(reload_model.predict_proba(X_test[:subset]), axis=1)))
    write_json(RESULTS / "model" / "label_mapping.json", label_mapping)
    write_json(RESULTS / "model" / "reload_verification.json", {"passed": reload_match, "rows_checked": subset})
    (RESULTS / "config" / "environment.txt").write_text(f"Python: {sys.version}\nPlatform: {platform.platform()}\nXGBoost: {xgb.__version__}\nCUDA runtime (PyTorch): {torch.version.cuda}\nGPU: {gpu_name}\nXGBoost build: {json.dumps(build_info)}\n", encoding="utf-8")
    config = {"xgboost_version": xgb.__version__, "python_version": sys.version, "platform": platform.platform(), "cuda_version": torch.version.cuda, "gpu": gpu_name, "random_seed": SEED, "feature_count": len(features), "class_count": class_count, "class_mapping": label_mapping, "dataset_paths": {key: str(value) for key, value in paths.items()}, "dataset_sizes": {key: report["rows"] for key, report in reports.items()}, "final_model_parameters": params, "best_iteration": int(model.best_iteration), "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds}
    write_json(RESULTS / "config" / "model_config.json", config)
    monitor.stop()
    resource_summary = monitor.summary() | {"training_seconds": round(training_seconds, 3), "evaluation_seconds": round(evaluation_seconds, 3), "total_execution_seconds": round(time.perf_counter() - start, 3), "gpu_training_confirmed": True}
    write_json(RESULTS / "resource_usage" / "resource_summary.json", resource_summary)
    model_size_mb = model_path.stat().st_size / 1024**2
    final_report = f"""# Final Results\n\n## Dataset\n- Classes: {class_count}\n- Features: {len(features)}\n- Train samples: {reports['train']['rows']}\n- Validation samples: {reports['validation']['rows']}\n- Test samples: {reports['test']['rows']}\n\n## Model\n- Algorithm: XGBoost CUDA classifier\n- Objective: multi:softprob\n- Best iteration: {model.best_iteration}\n- Model size: {model_size_mb:.2f} MB\n\n## Performance\n- Accuracy: {metrics['accuracy']:.6f}\n- Macro precision: {metrics['macro_precision']:.6f}\n- Macro recall: {metrics['macro_recall']:.6f}\n- Macro F1: {metrics['macro_f1']:.6f}\n- Weighted F1: {metrics['weighted_f1']:.6f}\n- Training time: {training_seconds:.2f} seconds\n- Inference/evaluation time: {evaluation_seconds:.2f} seconds\n\n## Resources\n- Peak RAM: {resource_summary['peak_ram_gb']:.2f} GB\n- RAM limit: 14 GB\n- RAM status: {resource_summary['ram_status']}\n- Peak GPU memory: {resource_summary['peak_gpu_memory_gb']:.2f} GB\n- GPU: {gpu_name}\n- GPU training: CONFIRMED\n\n## Protocol\n- Test data was not used for tuning or early stopping.\n- Train data was used for fitting; validation data was used for early stopping; test data was evaluated once after model selection.\n\nModel: `results/model/xgboost_{class_count}class_final.json`\nResults directory: `results/`\n"""
    (RESULTS / "FINAL_RESULTS.md").write_text(final_report, encoding="utf-8")
    logger.info("Completed successfully: metrics=%s resources=%s model=%s", metrics, resource_summary, model_path)
    if resource_summary["ram_status"] != "PASS":
        raise RuntimeError("Peak RAM exceeded the 14 GB hard limit")
    if not reload_match:
        raise RuntimeError("Reloaded model predictions did not match original model")


if __name__ == "__main__":
    main()