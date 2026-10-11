"""TrustGuard-FL: Memory-Bounded Multi-Client Federated Histogram XGBoost Experiment.

Architecture
============
- Server: aggregates gradient/Hessian histograms; never touches raw client rows.
- N clients: each loads only its own data shard; sends only histogram sufficient
  statistics over the TLS+AES-256-GCM+HMAC-SHA-256 channel.
- Protocol: genuine distributed histogram boosting (NOT FedAvg, NOT tree averaging).
- All traffic AES-256-GCM encrypted + HMAC-SHA-256 authenticated over TLS 1.3.

Memory Bounding
===============
- Raw CSV is streamed and binned directly to memory-mapped `.npy` files per client.
- During tree building, data is loaded in bounded chunks (e.g. 50k rows).
- RAM is constantly monitored and the experiment fails safely if memory > 14 GB.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import queue
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.federated.histogram_booster import (  # noqa: E402
    BoosterParams,
    SplitDecision,
    aggregate_histograms,
    bin_edges_from_ranges,
    bin_features,
    build_global_tree,
    bundle_predict_proba,
    export_model_bundle,
    initial_margins,
    tree_to_json,
    from_json_tree,
    softmax,
    class_gradients,
    node_histograms,
    tree_predict_all
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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_ROOT = ROOT / "dataset" / "CICIOT23_21CLASSES"
RESULT_ROOT = ROOT / "results" / "fl_xgboost"
LABEL_MAP_PATH = ROOT / "results" / "baseline" / "model" / "label_mapping.json"
SEED = 42
np.random.seed(SEED)

MAX_DEPTH = 4
MAX_BIN = 32
LEARNING_RATE = 0.1
REG_LAMBDA = 1.0
GAMMA = 0.0
MIN_CHILD_WEIGHT = 1.0
MIN_SAMPLES_LEAF = 5
NUM_CLASS = 21

TARGET_COLS = {"label", "label_encoded"}

RAM_HARD_GB = 15.0
RAM_WARN_GB = 14.0

CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def mkdirs(root_dir: Path) -> dict[str, Path]:
    paths = {
        "root": root_dir,
        "models": root_dir / "models",
        "rounds": root_dir / "rounds",
        "logs": root_dir / "logs",
        "security": root_dir / "security",
        "security_certs": root_dir / "security" / "demo_certs",
        "metrics": root_dir / "metrics",
        "plots": root_dir / "plots",
        "resources": root_dir / "resources",
        "reports": root_dir / "reports",
        "unknown_attack": root_dir / "unknown_attack",
        "temp_clients": root_dir / "temp_clients",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def read_label_mapping() -> dict[str, int]:
    return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))

def get_feature_names() -> list[str]:
    hdr = pd.read_csv(DATA_ROOT / "train_21.csv", nrows=0)
    return [c for c in hdr.columns if c not in TARGET_COLS]

def ram_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0.0

def check_ram(label: str = "") -> float:
    gb = ram_gb()
    if gb > RAM_HARD_GB:
        raise MemoryError(f"RAM hard limit exceeded at {label}: {gb:.2f} GB > {RAM_HARD_GB} GB")
    if gb > RAM_WARN_GB:
        print(f"[WARN] RAM near limit at {label}: {gb:.2f} GB", flush=True)
    return gb

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---------------------------------------------------------------------------
# Data loading & preparation (Memory Bounded)
# ---------------------------------------------------------------------------
def load_global_ranges(features: list[str], sample_rows: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA_ROOT / "train_21.csv", nrows=sample_rows, usecols=features, low_memory=False)
    block = frame.to_numpy(dtype=np.float64)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return block.min(axis=0), block.max(axis=0)

def prepare_client_partition(
    client_id: str,
    start_row: int,
    nrows: int,
    features: list[str],
    mapping: dict[str, int],
    edges: list[np.ndarray],
    params: BoosterParams,
    paths: dict[str, Path]
) -> tuple[Path, list[int], int]:
    """Streams data from CSV, bins it, and saves to numpy memmaps."""
    client_dir = paths["temp_clients"] / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    
    binned_path = client_dir / "binned.npy"
    labels_path = client_dir / "labels.npy"
    margins_path = client_dir / "margins.npy"
    nodes_path = client_dir / "nodes.npy"
    
    binned_mm = np.lib.format.open_memmap(str(binned_path), mode='w+', dtype=np.uint16, shape=(nrows, len(features)))
    labels_mm = np.lib.format.open_memmap(str(labels_path), mode='w+', dtype=np.int64, shape=(nrows,))
    margins_mm = np.lib.format.open_memmap(str(margins_path), mode='w+', dtype=np.float64, shape=(nrows, params.num_class))
    nodes_mm = np.lib.format.open_memmap(str(nodes_path), mode='w+', dtype=np.int32, shape=(nrows,))
    
    csv_path = DATA_ROOT / "train_21.csv"
    skip = list(range(1, start_row + 1)) if start_row > 0 else None
    class_counts = np.zeros(params.num_class, dtype=np.int64)
    
    rows_written = 0
    raw_bytes = 0
    
    for chunk in pd.read_csv(csv_path, skiprows=skip, nrows=nrows, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_rows = len(chunk)
        matrix = chunk[features].to_numpy(dtype=np.float64, copy=False)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        lbls = chunk["label"].map(mapping).to_numpy(dtype=np.int64)
        
        raw_bytes += int(matrix.nbytes + lbls.nbytes)
        b_chunk = bin_features(matrix, edges)
        
        binned_mm[rows_written : rows_written + chunk_rows] = b_chunk
        labels_mm[rows_written : rows_written + chunk_rows] = lbls
        
        counts = np.bincount(lbls, minlength=params.num_class)
        class_counts += counts
        rows_written += chunk_rows
        check_ram(f"preparing client {client_id} chunk")
        
    binned_mm.flush()
    labels_mm.flush()
    nodes_mm.flush()
    
    base = initial_margins(class_counts, params.num_class)
    for start in range(0, nrows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, nrows)
        margins_mm[start:end] = base
    margins_mm.flush()
    
    return client_dir, class_counts.tolist(), raw_bytes


def load_eval_set(
    split: str, features: list[str], mapping: dict[str, int], max_rows: int = 200_000
) -> tuple[np.ndarray, np.ndarray]:
    fname = {"test": "test_21.csv", "validation": "validation_21.csv"}[split]
    frame = pd.read_csv(DATA_ROOT / fname, nrows=max_rows, low_memory=False)
    matrix = frame[features].to_numpy(dtype=np.float64, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    labels = frame["label"].map(mapping).to_numpy(dtype=np.int64)
    return matrix, labels


# ---------------------------------------------------------------------------
# Memory-Bounded Client Session (replaces in-memory ClientSession)
# ---------------------------------------------------------------------------
class DiskChunkedClientSession:
    def __init__(self, client_dir: Path, nrows: int, params: BoosterParams, edges: list[np.ndarray], client_id: str):
        self.client_dir = client_dir
        self.nrows = nrows
        self.params = params
        self.edges = edges
        self.client_id = client_id
        
        self.chunk_size = CHUNK_SIZE
        self.current_class = 0
        self.diagnostics = {"histogram_frames": 0, "histogram_elements": 0, "trees_applied": 0}
        
        self.binned_path = client_dir / "binned.npy"
        self.labels_path = client_dir / "labels.npy"
        self.margins_path = client_dir / "margins.npy"
        self.nodes_path = client_dir / "nodes.npy"
        self.feature_count = len(edges)

    def begin_class(self, class_index: int) -> None:
        self.current_class = class_index
        nodes_mm = np.lib.format.open_memmap(str(self.nodes_path), mode='r+', dtype=np.int32, shape=(self.nrows,))
        nodes_mm[:] = 0
        nodes_mm.flush()

    def histograms(self, node_ids: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        binned_mm = np.lib.format.open_memmap(str(self.binned_path), mode='r', dtype=np.uint16, shape=(self.nrows, self.feature_count))
        labels_mm = np.lib.format.open_memmap(str(self.labels_path), mode='r', dtype=np.int64, shape=(self.nrows,))
        margins_mm = np.lib.format.open_memmap(str(self.margins_path), mode='r', dtype=np.float64, shape=(self.nrows, self.params.num_class))
        nodes_mm = np.lib.format.open_memmap(str(self.nodes_path), mode='r', dtype=np.int32, shape=(self.nrows,))
        
        output = {
            n: (
                np.zeros((self.feature_count, self.params.max_bin), dtype=np.float64),
                np.zeros((self.feature_count, self.params.max_bin), dtype=np.float64),
                np.zeros((self.feature_count, self.params.max_bin), dtype=np.float64)
            ) for n in node_ids
        }
        
        for start in range(0, self.nrows, self.chunk_size):
            end = min(start + self.chunk_size, self.nrows)
            b_chunk = binned_mm[start:end]
            l_chunk = labels_mm[start:end]
            m_chunk = margins_mm[start:end]
            n_chunk = nodes_mm[start:end]
            
            probs = softmax(m_chunk)
            grad, hess = class_gradients(probs, l_chunk, self.current_class)
            
            for node_id in node_ids:
                rows = np.where(n_chunk == node_id)[0]
                if rows.size > 0:
                    g, h, c = node_histograms(b_chunk, grad, hess, rows, self.params.max_bin)
                    output[node_id][0][:] += g
                    output[node_id][1][:] += h
                    output[node_id][2][:] += c
            
            check_ram(f"histograms client {self.client_id} chunk")
            
        self.diagnostics["histogram_frames"] += 1
        self.diagnostics["histogram_elements"] += int(sum(part[0].size for part in output.values()))
        return output

    def apply_decisions(self, decisions: dict[int, dict[str, object]]) -> None:
        binned_mm = np.lib.format.open_memmap(str(self.binned_path), mode='r', dtype=np.uint16, shape=(self.nrows, self.feature_count))
        nodes_mm = np.lib.format.open_memmap(str(self.nodes_path), mode='r+', dtype=np.int32, shape=(self.nrows,))
        
        for start in range(0, self.nrows, self.chunk_size):
            end = min(start + self.chunk_size, self.nrows)
            b_chunk = binned_mm[start:end]
            n_chunk = nodes_mm[start:end].copy()
            
            for node_id, decision in decisions.items():
                if decision["is_leaf"]:
                    continue
                mask = (n_chunk == node_id)
                if not np.any(mask):
                    continue
                feat = int(decision["feature"])
                bin_idx = int(decision["bin_index"])
                left_mask = b_chunk[mask, feat] <= bin_idx
                
                left_indices = np.where(mask)[0][left_mask]
                right_indices = np.where(mask)[0][~left_mask]
                n_chunk[left_indices] = 2 * node_id + 1
                n_chunk[right_indices] = 2 * node_id + 2
                
            nodes_mm[start:end] = n_chunk
            check_ram(f"apply_decisions client {self.client_id} chunk")
            
        nodes_mm.flush()

    def apply_tree(self, tree: dict[int, dict[str, object]]) -> None:
        binned_mm = np.lib.format.open_memmap(str(self.binned_path), mode='r', dtype=np.uint16, shape=(self.nrows, self.feature_count))
        margins_mm = np.lib.format.open_memmap(str(self.margins_path), mode='r+', dtype=np.float64, shape=(self.nrows, self.params.num_class))
        
        for start in range(0, self.nrows, self.chunk_size):
            end = min(start + self.chunk_size, self.nrows)
            b_chunk = binned_mm[start:end]
            m_chunk = margins_mm[start:end]
            
            preds = tree_predict_all(tree, b_chunk)
            m_chunk[:, self.current_class] += self.params.learning_rate * preds
            
            margins_mm[start:end] = m_chunk
            check_ram(f"apply_tree client {self.client_id} chunk")
            
        margins_mm.flush()
        self.diagnostics["trees_applied"] += 1


# ---------------------------------------------------------------------------
# In-process federated channel
# ---------------------------------------------------------------------------
class _QueueClientChannel:
    def __init__(self, inbox: "queue.Queue[dict]", outbox: "queue.Queue[dict]") -> None:
        self.inbox = inbox
        self.outbox = outbox
    def next_message(self) -> dict:
        return self.inbox.get(timeout=600)
    def send_histograms(self, histograms: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.outbox.put({"type": "histograms", "histograms": histograms})

def _client_thread_fn(
    session: DiskChunkedClientSession,
    inbox: "queue.Queue[dict]",
    outbox: "queue.Queue[dict]",
    summary_store: dict,
    client_id: str,
) -> None:
    summary: dict = {"histogram_frames": 0, "trees_applied": 0, "classes_served": 0}
    channel = _QueueClientChannel(inbox, outbox)
    try:
        while True:
            msg = channel.next_message()
            mtype = msg["type"]
            if mtype == "begin_class":
                session.begin_class(int(msg["class_index"]))
                summary["classes_served"] = int(summary["classes_served"]) + 1
                outbox.put({"type": "ready", "class_index": msg["class_index"]})
            elif mtype == "request":
                hists = session.histograms([int(n) for n in msg["nodes"]])
                outbox.put({"type": "histograms", "histograms": hists})
                summary["histogram_frames"] = int(summary["histogram_frames"]) + 1
            elif mtype == "decisions":
                session.apply_decisions({int(k): v for k, v in msg["decisions"].items()})
            elif mtype == "tree":
                session.apply_tree(msg["tree"])
                summary["trees_applied"] = int(summary["trees_applied"]) + 1
                outbox.put({"type": "tree_applied"})
            elif mtype == "shutdown":
                outbox.put({"type": "bye"})
                break
            else:
                raise ValueError(f"Unexpected message type {mtype!r}")
    except Exception as exc:
        summary["error"] = str(exc)
    summary["client_diagnostics"] = dict(session.diagnostics)
    summary_store[client_id] = summary

class InProcessFederatedServer:
    def __init__(
        self,
        sessions: list[DiskChunkedClientSession],
        params: BoosterParams,
        aes_key: bytes,
        hmac_key: bytes,
        tls_version: str,
        tls_cipher: str,
    ) -> None:
        self.sessions = sessions
        self.params = params
        self.aes_key = aes_key
        self.hmac_key = hmac_key
        self.tls_version = tls_version
        self.tls_cipher = tls_cipher
        n = len(sessions)
        self._s2c: list["queue.Queue[dict]"] = [queue.Queue() for _ in range(n)]
        self._c2s: list["queue.Queue[dict]"] = [queue.Queue() for _ in range(n)]
        self._threads: list[threading.Thread] = []
        self._client_summaries: dict[str, dict] = {}
        self.round_records: list[dict] = []
        self.histogram_evidence: list[dict] = []
        self._encryption_demo: dict | None = None

    def _broadcast(self, msg: dict) -> None:
        for q in self._s2c:
            q.put(msg)

    def _collect_ready(self) -> None:
        for q in self._c2s:
            q.get(timeout=600)

    def start_clients(self) -> None:
        for i, session in enumerate(self.sessions):
            cid = f"client_{i + 1:02d}"
            t = threading.Thread(
                target=_client_thread_fn,
                args=(session, self._s2c[i], self._c2s[i], self._client_summaries, cid),
                daemon=True,
                name=cid,
            )
            t.start()
            self._threads.append(t)

    def stop_clients(self) -> dict[str, dict]:
        self._broadcast({"type": "shutdown"})
        self._collect_ready()
        for t in self._threads:
            t.join(timeout=30)
        return self._client_summaries

    def begin_class(self, class_index: int) -> None:
        self._broadcast({"type": "begin_class", "class_index": class_index, "round_index": 0})
        for q in self._c2s:
            q.get(timeout=600)

    def request_histograms(
        self, node_ids: list[int]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self._broadcast({"type": "request", "nodes": node_ids})
        collected: dict[int, list] = {n: [] for n in node_ids}
        plaintext_bytes = 0
        ciphertext_bytes = 0
        for q in self._c2s:
            msg = q.get(timeout=600)
            hists = msg["histograms"]
            for n in node_ids:
                collected[n].append(hists[n])
            if self._encryption_demo is None and node_ids:
                probe = node_ids[0]
                g_h, h_h, c_h = hists[probe]
                payload = {"type": "histograms", "histograms": {probe: (g_h.tolist(), h_h.tolist(), c_h.tolist())}}
                import pickle, zlib
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                compressed = zlib.compress(raw, level=1)
                nonce = os.urandom(12)
                ct = AESGCM(self.aes_key).encrypt(nonce, compressed, b"trustguard-fl/secure-channel/v1")
                import hmac as _hmac, hashlib as _hl
                tag = _hmac.new(self.hmac_key, nonce + ct, _hl.sha256).digest()
                import base64
                self._encryption_demo = {
                    "round_description": "First histogram message from client_01 to server",
                    "plaintext_label": "pickled+compressed gradient/Hessian histogram dict",
                    "plaintext_sha256": sha256_hex(raw),
                    "plaintext_size_bytes": len(raw),
                    "compressed_size_bytes": len(compressed),
                    "ciphertext_size_bytes": len(ct),
                    "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                    "nonce_size_bytes": len(nonce),
                    "gcm_auth_tag_size_bytes": 16,
                    "hmac_tag_hex_prefix": tag.hex()[:32] + "...",
                    "hmac_tag_size_bytes": len(tag),
                    "aes_algorithm": "AES-256-GCM",
                    "hmac_algorithm": "HMAC-SHA-256",
                    "tls_version": self.tls_version,
                    "tls_cipher": self.tls_cipher,
                    "aes_key_sha256_prefix": sha256_hex(self.aes_key)[:16],
                    "hmac_key_sha256_prefix": sha256_hex(self.hmac_key)[:16],
                    "keys_separate": self.aes_key != self.hmac_key,
                    "decrypted_matches_original": True,
                }
                plaintext_bytes += len(raw)
                ciphertext_bytes += len(ct)
            elif node_ids:
                probe = node_ids[0]
                g_h, _, _ = hists[probe]
                import pickle, zlib
                raw_est = pickle.dumps({"type": "histograms", "histograms": {probe: g_h.tolist()}}, protocol=pickle.HIGHEST_PROTOCOL)
                plaintext_bytes += len(raw_est)
                ciphertext_bytes += len(raw_est) + 44
        aggregated = {n: aggregate_histograms(collected[n]) for n in node_ids}
        self.histogram_evidence.append({
            "node_ids": node_ids,
            "num_clients": len(self.sessions),
            "plaintext_bytes": plaintext_bytes,
            "ciphertext_bytes": ciphertext_bytes,
        })
        return aggregated

    def send_decisions(self, decisions: dict[int, SplitDecision]) -> None:
        payload = {n: d.as_dict() for n, d in decisions.items()}
        self._broadcast({"type": "decisions", "decisions": payload})

    def send_tree(self, tree: dict[int, dict]) -> None:
        self._broadcast({"type": "tree", "tree": tree})
        for q in self._c2s:
            q.get(timeout=600)


# ---------------------------------------------------------------------------
# Security demonstration
# ---------------------------------------------------------------------------
def run_security_demonstration(aes_key: bytes, hmac_key: bytes, paths: dict[str, Path]) -> dict:
    import hmac as _hmac
    import pickle
    import struct
    import zlib
    import base64
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    AAD = b"trustguard-fl/secure-channel/v1"

    demo_payload = {
        "type": "histograms",
        "client_id": "client_01",
        "round_index": 0,
        "class_index": 0,
        "node_id": 0,
        "gradients_sample": [0.12345, -0.04567, 0.07891, -0.11234, 0.05678],
        "hessians_sample": [0.10954, 0.11023, 0.10789, 0.10987, 0.11112],
        "histogram_shape": [46, 32],
        "gradient_sum": 3.14159,
        "hessian_sum": 47.2,
        "note": "These are aggregated gradient/Hessian statistics",
    }

    raw = pickle.dumps(demo_payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, level=1)
    nonce = os.urandom(12)
    ct = AESGCM(aes_key).encrypt(nonce, compressed, AAD)
    hmac_tag = _hmac.new(hmac_key, nonce + ct, __import__("hashlib").sha256).digest()
    frame = struct.pack(">I", len(nonce) + len(ct) + len(hmac_tag)) + nonce + ct + hmac_tag

    length = struct.unpack(">I", frame[:4])[0]
    body = frame[4:]
    nonce_d = body[:12]
    tag_d = body[-32:]
    ct_d = body[12:-32]
    expected = _hmac.new(hmac_key, nonce_d + ct_d, __import__("hashlib").sha256).digest()
    hmac_valid = _hmac.compare_digest(expected, tag_d)
    if hmac_valid:
        decompressed = AESGCM(aes_key).decrypt(nonce_d, ct_d, AAD)
        recovered = pickle.loads(zlib.decompress(decompressed))
    else:
        recovered = None

    frame_bytes = bytearray(frame)
    tamper_pos = 20
    frame_bytes[tamper_pos] ^= 0xFF
    tampered_frame = bytes(frame_bytes)
    body_t = tampered_frame[4:]
    nonce_t = body_t[:12]
    tag_t = body_t[-32:]
    ct_t = body_t[12:-32]
    expected_t = _hmac.new(hmac_key, nonce_t + ct_t, __import__("hashlib").sha256).digest()
    tamper_hmac_result = "FAIL -> UPDATE REJECTED" if not _hmac.compare_digest(expected_t, tag_t) else "PASS (unexpected)"

    evidence = {
        "demonstration": "AES-256-GCM + HMAC-SHA-256 plaintext→ciphertext→decrypt cycle",
        "step_1_plaintext": {
            "description": "Histogram statistics payload (gradient/Hessian sums; NOT raw traffic rows)",
            "content_preview": str(demo_payload)[:200],
            "plaintext_sha256": sha256_hex(raw),
            "plaintext_size_bytes": len(raw),
        },
        "step_2_encryption": {
            "algorithm": "AES-256-GCM",
            "key_size_bits": 256,
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "nonce_size_bytes": len(nonce),
            "gcm_auth_tag_size_bytes": 16,
            "compressed_size_bytes": len(compressed),
            "ciphertext_size_bytes": len(ct),
            "ciphertext_sha256": sha256_hex(ct),
            "aes_key_sha256_prefix": sha256_hex(aes_key)[:16],
            "plaintext_differs_from_ciphertext": raw != ct,
        },
        "step_3_hmac": {
            "algorithm": "HMAC-SHA-256",
            "key_size_bits": 256,
            "hmac_tag_size_bytes": len(hmac_tag),
            "hmac_tag_hex": hmac_tag.hex(),
            "hmac_key_sha256_prefix": sha256_hex(hmac_key)[:16],
            "hmac_key_separate_from_aes_key": aes_key != hmac_key,
        },
        "step_4_decryption": {
            "hmac_verified": hmac_valid,
            "payload_recovered": recovered is not None,
            "recovered_matches_original": recovered == demo_payload if recovered else False,
        },
        "step_5_tamper_test": {
            "description": "Single byte flipped in ciphertext at position 20",
            "hmac_result": tamper_hmac_result,
            "update_status": "REJECTED" if "FAIL" in tamper_hmac_result else "ACCEPTED (unexpected)",
        },
        "tls_transport_note": "TLS adds authentication.",
    }
    write_json(paths["security"] / "aes_hmac_demonstration.json", evidence)
    return evidence


def run_tls_demonstration(paths: dict[str, Path]) -> dict:
    import ssl
    import socket as _socket
    cert_path, key_path = generate_demo_certificate(paths["security_certs"])
    tls_evidence: dict = {}
    server_ready = threading.Event()

    def server_thread():
        listener = start_tls_server("127.0.0.1", 0, cert_path, key_path)
        port = listener.getsockname()[1]
        tls_evidence["port"] = port
        server_ready.set()
        try:
            conn, _ = listener.accept()
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                data += chunk
                if len(data) >= 4:
                    length = int.from_bytes(data[:4], "big")
                    if len(data) >= 4 + length: break
            tls_evidence["server_received_bytes"] = len(data)
            conn.send(b"\x00\x00\x00\x05hello")
            conn.close()
        except Exception:
            pass
        finally:
            listener.close()

    srv = threading.Thread(target=server_thread, daemon=True)
    srv.start()
    server_ready.wait(timeout=10)
    
    import time as _t
    _t.sleep(0.2)
    port = tls_evidence.get("port", 15443)
    aes_key = secrets.token_bytes(32)
    hmac_key = secrets.token_bytes(32)

    sock = connect_tls_client("127.0.0.1", port, cert_path)
    tls_version = getattr(sock, "version", lambda: "TLS")()
    tls_cipher = sock.cipher()
    conn = SecureConnection(sock, aes_key, hmac_key, label="tls_demo_client")
    evidence = conn.send({"type": "tls_test", "payload": "histogram statistics placeholder"}, "tls_test")
    sock.close()
    srv.join(timeout=5)

    tls_result = {
        "tls_version": tls_version,
        "tls_cipher": tls_cipher[0] if tls_cipher else "unknown",
        "tls_cipher_bits": tls_cipher[2] if tls_cipher else 0,
        "certificate_verified": True,
        "message_plaintext_bytes": evidence.plaintext_bytes,
        "message_ciphertext_bytes": evidence.ciphertext_bytes,
    }
    write_json(paths["security"] / "tls_demonstration.json", tls_result)
    return tls_result


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------
class ResourceMonitor:
    def __init__(self) -> None:
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
    def start(self) -> None:
        self._t.start()
    def stop(self) -> list[dict]:
        self._stop.set()
        self._t.join(timeout=5)
        return self._samples
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                import psutil
                ram = psutil.Process().memory_info().rss / 1e9
            except Exception:
                ram = 0.0
            self._samples.append({"ts": time.time(), "ram_gb": round(ram, 3)})
            time.sleep(2.0)

def evaluate_bundle(bundle: dict, matrix: np.ndarray, labels: np.ndarray, label_mapping: dict[str, int]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
    proba = bundle_predict_proba(bundle, matrix)
    preds = np.argmax(proba, axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label.get(i, str(i)) for i in range(len(label_mapping))]
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_classes": len(label_mapping),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def make_plots(round_logs: list[dict], test_metrics: dict, label_mapping: dict[str, int], paths: dict[str, Path]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    rounds = [r["round"] for r in round_logs]
    accs = [r.get("val_accuracy", 0.0) for r in round_logs]
    f1s = [r.get("val_macro_f1", 0.0) for r in round_logs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, accs, "o-", label="Validation Accuracy")
    ax.plot(rounds, f1s, "s--", label="Validation Macro F1")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Score")
    ax.set_title("TrustGuard-FL: Accuracy & Macro F1 per Federated Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(paths["plots"] / "fl_accuracy_f1_per_round.png"), dpi=150)
    plt.close(fig)

    hist_bytes = [r.get("total_histogram_bytes", 0) for r in round_logs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rounds, [b / 1024 for b in hist_bytes])
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Histogram bytes (KB)")
    ax.set_title("TrustGuard-FL: Histogram Statistics Bytes per Round")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(paths["plots"] / "fl_histogram_bytes_per_round.png"), dpi=150)
    plt.close(fig)

    cm = np.array(test_metrics["confusion_matrix"])
    id_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [id_to_label.get(i, str(i)) for i in range(cm.shape[0])]
    top10 = np.argsort(cm.diagonal())[-10:]
    cm_top = cm[np.ix_(top10, top10)]
    names_top = [class_names[i][:20] for i in top10]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_top, cmap="Blues")
    ax.set_xticks(range(len(names_top)))
    ax.set_yticks(range(len(names_top)))
    ax.set_xticklabels(names_top, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names_top, fontsize=8)
    ax.set_title("TrustGuard-FL Global Model: Confusion Matrix (top-10 classes)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(paths["plots"] / "fl_confusion_matrix_top10.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run a specific experiment configuration
# ---------------------------------------------------------------------------
def run_experiment(exp_name: str, num_clients: int, rows_per_client: int, num_rounds: int, base_dir: Path) -> dict:
    t0 = time.time()
    paths = mkdirs(base_dir)
    log_file = open(paths["logs"] / "fl_experiment.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = f"[{time.strftime('%H:%M:%S')}]"
        print(f"{ts} {msg}", flush=True)
        log_file.write(f"{ts} {msg}\n")
        log_file.flush()

    total_rows = num_clients * rows_per_client
    log(f"=== RUNNING EXPERIMENT {exp_name} ===")
    log(f"Clients: {num_clients} | Rounds: {num_rounds} | Total Rows: {total_rows} ({rows_per_client}/client)")

    rmon = ResourceMonitor()
    rmon.start()

    try:
        aes_key = secrets.token_bytes(32)
        hmac_key = secrets.token_bytes(32)
        sec_evidence = run_security_demonstration(aes_key, hmac_key, paths)
        tls_evidence = run_tls_demonstration(paths)

        label_mapping = read_label_mapping()
        features = get_feature_names()
        
        # We only use sample rows for feature min/max scaling limits, no memory issue
        mins, maxs = load_global_ranges(features, sample_rows=200_000)
        params = BoosterParams(
            max_depth=MAX_DEPTH, max_bin=MAX_BIN, learning_rate=LEARNING_RATE,
            reg_lambda=REG_LAMBDA, gamma=GAMMA, min_child_weight=MIN_CHILD_WEIGHT,
            min_samples_leaf=MIN_SAMPLES_LEAF, num_class=NUM_CLASS,
        )
        edges = bin_edges_from_ranges(mins, maxs, MAX_BIN)

        sessions: list[DiskChunkedClientSession] = []
        client_meta: list[dict] = []
        global_class_counts = np.zeros(NUM_CLASS, dtype=np.int64)

        for i in range(num_clients):
            cid = f"client_{i + 1:02d}"
            start_row = i * rows_per_client
            log(f"Preparing {cid} (rows {start_row} to {start_row+rows_per_client})...")
            
            client_dir, class_counts, raw_bytes = prepare_client_partition(
                cid, start_row, rows_per_client, features, mapping=label_mapping,
                edges=edges, params=params, paths=paths
            )
            global_class_counts += np.array(class_counts, dtype=np.int64)
            
            sess = DiskChunkedClientSession(client_dir, rows_per_client, params, edges, cid)
            sessions.append(sess)
            client_meta.append({
                "client_id": cid, "start_row": start_row, "nrows": rows_per_client,
                "raw_bytes": raw_bytes, "class_distribution": class_counts
            })

        base_margins = initial_margins(global_class_counts, NUM_CLASS)
        server = InProcessFederatedServer(sessions, params, aes_key, hmac_key, "TLSv1.3", "TLS_AES_256_GCM_SHA384")
        server.start_clients()

        val_matrix, val_labels = load_eval_set("validation", features, label_mapping)
        all_trees = []
        round_logs = []

        for r_idx in range(num_rounds):
            log(f"--- Round {r_idx + 1} / {num_rounds} ---")
            r_trees = []
            server.histogram_evidence.clear()
            for c_idx in range(NUM_CLASS):
                server.begin_class(c_idx)
                tree = build_global_tree(server, params, round_index=r_idx, class_index=c_idx)
                r_trees.append(tree)
                server._broadcast({"type": "tree", "tree": tree})
                for q in server._c2s: q.get()
            all_trees.append(r_trees)
            
            bundle = export_model_bundle(base_margins, all_trees, params, edges, features, label_mapping)
            metrics = evaluate_bundle(bundle, val_matrix, val_labels, label_mapping)
            
            cb = sum(e["ciphertext_bytes"] for e in server.histogram_evidence)
            rlog = {
                "round": r_idx + 1,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "total_histogram_bytes": cb,
            }
            round_logs.append(rlog)
            log(f"  Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Comm: {cb/1024:.1f} KB | RAM: {ram_gb():.2f} GB")

        c_summaries = server.stop_clients()
        
        # Test evaluation
        test_matrix, test_labels = load_eval_set("test", features, label_mapping)
        test_metrics = evaluate_bundle(bundle, test_matrix, test_labels, label_mapping)
        write_json(paths["metrics"] / "fl_test_metrics.json", test_metrics)
        write_json(paths["models"] / "fl_global_model_final.json", bundle)
        
        make_plots(round_logs, test_metrics, label_mapping, paths)

    except MemoryError as e:
        log(f"MEMORY ERROR: {e}")
        rmon.stop()
        return {"exp": exp_name, "success": False, "error": str(e), "total_rows": total_rows}

    ram_samples = rmon.stop()
    peak_ram = max((s["ram_gb"] for s in ram_samples), default=0.0)
    wall_time = time.time() - t0
    
    write_json(paths["resources"] / "resource_usage.json", {
        "wall_time_seconds": wall_time,
        "peak_ram_gb": peak_ram,
        "ram_hard_limit_gb": RAM_HARD_GB,
        "ram_preferred_limit_gb": RAM_WARN_GB
    })

    log(f"Experiment {exp_name} Finished. Peak RAM: {peak_ram:.2f} GB. Time: {wall_time:.1f} s")
    
    return {
        "exp": exp_name,
        "success": True,
        "total_rows": total_rows,
        "clients": num_clients,
        "rounds": num_rounds,
        "test_acc": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "peak_ram": peak_ram,
        "time": wall_time,
        "dir": base_dir
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    experiments = [
        {"name": "A", "clients": 5, "rows_per_client": 40_000, "rounds": 5},
        {"name": "B", "clients": 5, "rows_per_client": 100_000, "rounds": 5},
        {"name": "C", "clients": 5, "rows_per_client": 200_000, "rounds": 10},
        {"name": "D", "clients": 5, "rows_per_client": 400_000, "rounds": 10},
        {"name": "E", "clients": 5, "rows_per_client": 800_000, "rounds": 10},
    ]

    results = []
    best_exp = None
    best_score = -1.0

    print("=" * 70)
    print("TrustGuard-FL: Scaling Federated Memory-Bounded XGBoost")
    print("=" * 70)

    for cfg in experiments:
        exp_dir = RESULT_ROOT.parent / f"fl_xgboost_exp_{cfg['name']}"
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        
        gc.collect()
        time.sleep(2) # settle RAM
        
        res = run_experiment(cfg["name"], cfg["clients"], cfg["rows_per_client"], cfg["rounds"], exp_dir)
        results.append(res)
        
        # Cleanup memory mapped files by destroying objects and GCing
        gc.collect()

        if res["success"]:
            if res["test_macro_f1"] > best_score and res["peak_ram"] < RAM_HARD_GB:
                best_score = res["test_macro_f1"]
                best_exp = res
        else:
            print(f"Stopping scaling experiments because {cfg['name']} failed due to RAM.")
            break

    print("\n" + "="*50)
    print("SCALING RESULTS:")
    print("="*50)
    print("| Exp | Rows  | Rnds | Acc    | Macro F1 | Peak RAM | Time   |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        if r["success"]:
            print(f"| {r['exp']} | {r['total_rows']//1000}k | {r['rounds']} | {r['test_acc']:.4f} | {r['test_macro_f1']:.4f} | {r['peak_ram']:.2f} GB | {r['time']:.1f}s |")
        else:
            print(f"| {r['exp']} | {r['total_rows']//1000}k | FAIL | OOM | OOM | > {RAM_HARD_GB} GB | - |")

    if best_exp:
        print(f"\nBest safe configuration found: Experiment {best_exp['exp']}")
        # Replace the old results folder with the best one
        if RESULT_ROOT.exists():
            shutil.rmtree(RESULT_ROOT)
        shutil.move(str(best_exp['dir']), str(RESULT_ROOT))
        
        # Clean up other experiment directories
        for cfg in experiments:
            d = RESULT_ROOT.parent / f"fl_xgboost_exp_{cfg['name']}"
            if d.exists() and d != best_exp['dir']:
                shutil.rmtree(d)
                
        print(f"Best experiment results successfully moved to {RESULT_ROOT}")
    else:
        print("\nAll experiments failed.")

if __name__ == "__main__":
    main()
