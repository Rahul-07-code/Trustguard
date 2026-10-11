"""TrustGuard-FL federated histogram-XGBoost experiment orchestrator.

Server process
  * owns the global model and the aggregated histogram statistics;
  * never reads the client training shards.

Client processes (spawned separately)
  * each reads *only* its own shard of `train_21.csv` (raw rows never leave);
  * compute local gradient/Hessian histograms and apply the global tree.

All federated traffic uses the real secure channel: TLS transport plus
AES-256-GCM payload encryption and a separate HMAC-SHA-256 key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import py_compile
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.federated.histogram_booster import (  # noqa: E402
    BoosterParams,
    ClientSession,
    bin_edges_from_ranges,
    bin_features,
    bundle_predict_proba,
    initial_margins,
)
from fl_xgboost.federated.secure_transport import (  # noqa: E402
    SecureServerChannel,
    export_bundle_from_trees,
    run_federated_training,
    serve_client,
)
from fl_xgboost.security.secure_channel import (  # noqa: E402
    SecureChannelError,
    SecureConnection,
    connect_tls_client,
    generate_demo_certificate,
    open_sealed_frame,
    seal_frame,
    start_tls_server,
)

DATA_ROOT = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULT_ROOT = ROOT / "results" / "fl_xgboost"
TARGET_COLUMNS = {"label", "label_encoded"}
SEED = 42


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": RESULT_ROOT,
        "models": RESULT_ROOT / "models",
        "rounds": RESULT_ROOT / "rounds",
        "logs": RESULT_ROOT / "logs",
        "security": RESULT_ROOT / "security",
        "unknown_attack": RESULT_ROOT / "unknown_attack",
        "resources": RESULT_ROOT / "resources",
        "reports": RESULT_ROOT / "reports",
        "metrics": RESULT_ROOT / "metrics",
        "plots": RESULT_ROOT / "plots",
        "validation": RESULT_ROOT / "validation",
        "certs": RESULT_ROOT / "security" / "demo_certs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_feature_names() -> list[str]:
    header = pd.read_csv(DATA_ROOT / "train_21.csv", nrows=1, low_memory=False)
    return [column for column in header.columns if column not in TARGET_COLUMNS]


def label_mapping() -> dict[str, int]:
    return json.loads((ROOT / "results" / "model" / "label_mapping.json").read_text(encoding="utf-8"))


def csv_row_count(path: Path) -> int:
    total = 0
    for frame in pd.read_csv(path, usecols=["label"], chunksize=500_000, low_memory=False):
        total += len(frame)
    return total


def load_shard(path: Path, start_row: int, nrows: int, features: list[str], mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one contiguous shard of the training CSV (client-local)."""
    frame = pd.read_csv(path, skiprows=range(1, start_row + 1), nrows=nrows, low_memory=False)
    matrix = frame[features].to_numpy(dtype=np.float64, copy=True)
    labels = frame["label"].map(mapping)
    if labels.isna().any():
        raise ValueError("shard contains a label outside the 21-class mapping")
    label_ids = labels.to_numpy(dtype=np.int64, copy=True)
    if not np.isfinite(matrix).all():
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, label_ids


def bounded_range_sample(matrix_source: Path, features: list[str], rows: int) -> np.ndarray:
    frame = pd.read_csv(matrix_source, nrows=rows, usecols=features, low_memory=False)
    block = frame.to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Client worker (spawned process; raw shard never leaves this process)
# ---------------------------------------------------------------------------
def client_worker(
    client_id: str,
    host: str,
    port: int,
    cafile: str,
    aes_key: bytes,
    hmac_key: bytes,
    shard_path: str,
    start_row: int,
    nrows: int,
    features: list[str],
    mapping: dict[str, int],
    summary_path: str,
) -> None:
    matrix, labels = load_shard(Path(shard_path), start_row, nrows, features, mapping)
    raw_nbytes = int(matrix.nbytes + labels.nbytes)
    socket_ = connect_tls_client(host, port, Path(cafile))
    secure = SecureConnection(socket_, aes_key, hmac_key, label=client_id)
    init = secure.recv(expect="init")
    edges = [np.asarray(edge, dtype=np.float64) for edge in init["edges"]]
    params = BoosterParams(**init["params"])
    base_margins = np.asarray(init["base_margins"], dtype=np.float64)
    binned = bin_features(matrix, edges)
    session = ClientSession(binned, labels, params, base_margins, client_id=client_id)
    del matrix
    counts = np.bincount(labels, minlength=params.num_class)
    secure.send(
        {
            "type": "client_ready",
            "client_id": client_id,
            "rows": int(labels.shape[0]),
            "feature_count": int(binned.shape[1]),
            "binned_bytes": int(binned.nbytes),
            "local_raw_bytes": raw_nbytes,
            "class_distribution": counts.tolist(),
            "tls": secure.tls_metadata(),
        }
    )
    summary = serve_client(secure, session)
    summary["client_id"] = client_id
    summary["rows"] = int(labels.shape[0])
    summary["local_raw_bytes"] = raw_nbytes
    summary["rows_sent_to_server"] = 0
    summary["raw_rows_transmitted"] = 0
    secure.close()
    write_json(Path(summary_path), summary)