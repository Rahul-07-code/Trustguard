"""Bounded native XGBoost federated collective smoke test."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "dataset" / "CICIOT23_21CLASSES" / "train_21.csv"
RESULTS = ROOT / "results" / "fl_xgboost" / "rounds"
FEATURES: list[str] | None = None
LABELS = json.loads((ROOT / "results" / "model" / "label_mapping.json").read_text(encoding="utf-8"))
LABEL_TO_ID = {name: int(value) for name, value in LABELS.items()}


def load_slice(start: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    global FEATURES
    frame = pd.read_csv(DATA, skiprows=range(1, start + 1), nrows=rows, low_memory=False)
    if FEATURES is None:
        FEATURES = [column for column in frame.columns if column not in {"label", "label_encoded"}]
    values = frame[FEATURES].to_numpy(dtype=np.float32, copy=True)
    labels = frame["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int32, copy=True)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("smoke-test slice contains missing or infinite features")
    return values, labels


def run_server(port: int, workers: int) -> None:
    import xgboost.federated

    xgboost.federated.run_federated_server(workers, port, timeout=120)


def run_worker(port: int, workers: int, rank: int, rows: int, rounds: int) -> None:
    import xgboost as xgb
    from xgboost import collective

    start = 1 + rank * rows
    features, labels = load_slice(start, rows)
    environment = {
        "dmlc_communicator": "federated",
        "federated_server_address": f"127.0.0.1:{port}",
        "federated_world_size": workers,
        "federated_rank": rank,
    }
    with collective.CommunicatorContext(**environment):
        train = xgb.DMatrix(features, label=labels)
        parameters = {
            "max_depth": 4,
            "eta": 0.1,
            "objective": "multi:softprob",
            "num_class": 21,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": 1,
            "seed": 42,
        }
        booster = xgb.train(parameters, train, num_boost_round=rounds)
        if collective.get_rank() == 0:
            RESULTS.mkdir(parents=True, exist_ok=True)
            model_path = RESULTS / "global_model_smoke.json"
            booster.save_model(model_path)
            (RESULTS / "smoke_metadata.json").write_text(json.dumps({"workers": workers, "rows_per_client": rows, "rounds": rounds, "feature_count": features.shape[1], "model_path": str(model_path)}, indent=2), encoding="utf-8")


def main() -> None:
    workers = 3
    rows = 2000
    rounds = 2
    port = 19093
    context = mp.get_context("spawn")
    server = context.Process(target=run_server, args=(port, workers), name="fl-server")
    server.start()
    deadline = time.monotonic() + 10
    while server.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
        if server.exitcode is not None:
            break
    if not server.is_alive():
        raise RuntimeError(f"federated server exited before workers: exitcode={server.exitcode}")
    clients = [context.Process(target=run_worker, args=(port, workers, rank, rows, rounds), name=f"fl-client-{rank + 1}") for rank in range(workers)]
    for client in clients:
        client.start()
    for client in clients:
        client.join(180)
    failures = [client.name for client in clients if client.exitcode != 0]
    server.terminate()
    server.join(10)
    if failures:
        raise RuntimeError(f"federated smoke clients failed: {failures}")
    model_path = RESULTS / "global_model_smoke.json"
    if not model_path.exists():
        raise RuntimeError("rank 0 did not write the global smoke model")
    print("native_federated_smoke=passed")
    print(f"clients={workers} rows_per_client={rows} rounds={rounds}")
    print(f"model={model_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()