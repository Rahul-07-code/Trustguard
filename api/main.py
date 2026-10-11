import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import psutil

# Add TrustGuard-FL root to path to import backend logic
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.security.aes import AESGCMEncryption
from src.security.hmac_auth import HMACAuth
from src.anomaly.isolation_forest import UpdateAnomalyDetector
from run_fl_xgboost import predict_global, get_trees_margin
from src.fl.protocol import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    EncryptedUpdateEnvelope,
    SubmissionOutcome,
)
from src.fl.state import state_manager
from src.fl.audit import audit_logger
from src.fl.server import fl_coordinator
from fastapi import Header

app = FastAPI(title="TrustGuard-FL API", description="SOC Backend Service for TrustGuard-FL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = ROOT / "results" / "fl_xgboost"

# Seed baseline clients if state is empty so inventory is immediately functional
if not state_manager.clients:
    t1, _ = state_manager.register_client("Client-01", "Gateway Node Alpha", "IoT Linux")
    t2, _ = state_manager.register_client("Client-02", "Sensor Hub Beta", "Windows Edge")
    t3, _ = state_manager.register_client("Client-03", "Telemetry Aggregator", "Embedded Linux")
    audit_logger.record_event(
        security_layer="SYSTEM",
        event_type="soc_service_initialized",
        outcome="PASS",
        severity="INFO",
        details={"status": "Coordinator initialized with persistent audit ledger"},
    )

# -----------------
# DATA LOADERS
# -----------------
def load_json(path):
    if not path.exists(): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_records(path):
    if not path.exists(): return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/status")
def get_system_status():
    manifest = load_json(RESULTS_DIR / "manifests" / "dataset_manifest.json")
    metrics = load_json(RESULTS_DIR / "metrics" / "fl_xgboost_metrics.json")
    sec = load_json(RESULTS_DIR / "security" / "security_validation.json")
    mem = load_json(RESULTS_DIR / "monitoring" / "fl_xgboost_memory_report.json")
    
    clients = state_manager.get_all_clients()
    connected_count = sum(1 for c in clients if c["status"] == "CONNECTED")
    active_round = state_manager.get_active_round()
    
    return {
        "status": "OPERATIONAL" if manifest else "OFFLINE",
        "fl_engine": "READY" if (RESULTS_DIR / "models" / "fl_xgboost_final_model.pkl").exists() else "UNAVAILABLE",
        "global_model": "READY",
        "gpu": "ACTIVE",
        "tls": "ACTIVE" if sec.get("TLS_E2E_PASS") else "INACTIVE",
        "aes": "ACTIVE" if sec.get("AES_PASS") else "INACTIVE",
        "hmac": "ACTIVE" if sec.get("HMAC_PASS") else "INACTIVE",
        "isolation_forest": "ACTIVE",
        "dataset_rows": manifest.get("total_rows", 0),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "peak_ram_gb": mem.get("peak_ram_gb", 0.0),
        "ram_limit_gb": mem.get("limit_gb", 14.0),
        "registered_clients_count": len(clients),
        "connected_clients_count": connected_count,
        "active_round_id": active_round.get("round_id", 1),
        "active_round_status": active_round.get("status", "TRAINING"),
    }

@app.get("/api/fl/rounds")
def get_fl_rounds():
    # Return combination of historical round metrics and active round state
    csv_records = load_csv_records(RESULTS_DIR / "metrics" / "fl_xgboost_round_metrics.csv")
    live_rounds = state_manager.get_all_rounds()
    return csv_records if csv_records else live_rounds

@app.get("/api/fl/rounds/active")
def get_active_round():
    return state_manager.get_active_round()

@app.post("/api/fl/rounds/start")
def start_new_round(target_clients: int = 2):
    r = state_manager.start_new_round(target_clients=target_clients)
    audit_logger.record_event(
        security_layer="AGGREGATION",
        event_type="federated_round_started",
        outcome="PASS",
        severity="INFO",
        round_id=r.round_id,
        details={"target_clients": target_clients},
    )
    return r.to_dict()

@app.get("/api/clients")
def get_clients():
    return state_manager.get_all_clients()

@app.post("/api/fl/register", response_model=ClientRegistrationResponse)
def register_client(req: ClientRegistrationRequest):
    token, reg_iso = state_manager.register_client(
        client_id=req.client_id,
        client_name=req.client_name,
        platform=req.platform,
    )
    audit_logger.record_event(
        security_layer="AUTH",
        event_type="client_registered",
        outcome="PASS",
        severity="INFO",
        client_id=req.client_id,
        details={"platform": req.platform, "samples": req.dataset_samples},
    )
    return ClientRegistrationResponse(
        client_id=req.client_id,
        auth_token=token,
        registered_at=reg_iso,
    )

@app.get("/api/fl/keys")
def get_coordination_keys(authorization: Optional[str] = Header(None), x_client_id: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        if x_client_id and not state_manager.authenticate_client(x_client_id, token):
            raise HTTPException(401, "Invalid client credentials")
            
    return fl_coordinator.get_keys_hex()

@app.post("/api/fl/rounds/{round_id}/submit", response_model=SubmissionOutcome)
def submit_update(round_id: int, envelope: EncryptedUpdateEnvelope):
    if envelope.round_id != round_id:
        raise HTTPException(400, "Round ID mismatch between URL and payload envelope")
    return fl_coordinator.process_submission(envelope)

@app.get("/api/security/events")
def get_security_events(
    client_id: Optional[str] = None,
    severity: Optional[str] = None,
    layer: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    events, total = audit_logger.query_events(
        client_id=client_id,
        severity=severity,
        layer=layer,
        outcome=outcome,
        limit=limit,
        offset=offset,
    )
    # Format for UI compatibility with both new and legacy schema
    formatted = []
    for e in events:
        formatted.append({
            "id": e["event_id"],
            "event_id": e["event_id"],
            "timestamp": e["timestamp"],
            "severity": e["severity"],
            "component": e["security_layer"],
            "security_layer": e["security_layer"],
            "event": e["event_type"],
            "event_type": e["event_type"],
            "outcome": e["outcome"],
            "client_id": e.get("client_id", "SYSTEM"),
            "round_id": e.get("round_id"),
            "details": e.get("details", {}),
        })
    return formatted

@app.get("/api/system/health")
def get_system_health():
    mem = load_json(RESULTS_DIR / "monitoring" / "fl_xgboost_memory_report.json")
    return {
        "cpu_usage_pct": psutil.cpu_percent(),
        "ram_usage_gb": psutil.Process().memory_info().rss / 1e9,
        "peak_ram_gb": mem.get("peak_ram_gb", 0),
        "ram_limit_gb": mem.get("limit_gb", 14.0),
        "ram_status": mem.get("status", "UNKNOWN"),
        "gpu_status": "ACTIVE",
        "tls_status": "ACTIVE",
        "backend_status": "OPERATIONAL"
    }

# -----------------
# SECURITY DEMO ENDPOINTS
# -----------------
class DemoRequest(BaseModel):
    payload: str

@app.post("/api/demo/aes")
def demo_aes(req: DemoRequest):
    aes_key = os.urandom(32)
    aes = AESGCMEncryption(aes_key)
    
    nonce, ct = aes.encrypt(req.payload.encode(), b"trustguard-fl")
    dec = aes.decrypt(nonce, ct, b"trustguard-fl")
    
    return {
        "original": req.payload,
        "ciphertext_hex": ct.hex(),
        "nonce_hex": nonce.hex(),
        "decrypted": dec.decode(),
        "status": "PASS"
    }

@app.post("/api/demo/hmac")
def demo_hmac(req: DemoRequest):
    hmac_key = os.urandom(32)
    hmac = HMACAuth(hmac_key)
    
    payload_bytes = req.payload.encode()
    sig = hmac.sign(payload_bytes)
    valid = hmac.verify(payload_bytes, sig)
    
    # Tamper test
    tampered = bytearray(payload_bytes)
    if len(tampered) > 0:
        tampered[0] ^= 0xFF
    tamper_valid = hmac.verify(bytes(tampered), sig)
    
    return {
        "payload": req.payload,
        "signature_hex": sig.hex(),
        "verify_original": valid,
        "verify_tampered": tamper_valid,
        "tamper_detected": not tamper_valid
    }

@app.post("/api/demo/isolation-forest")
def demo_isolation_forest():
    detector = UpdateAnomalyDetector(contamination=0.1)
    from sklearn.datasets import make_classification
    
    # Generate a single distribution for normal clients
    X, y = make_classification(n_samples=250, n_features=20, n_informative=20, n_redundant=0, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    
    # Train enough normal updates to fit the Isolation Forest
    for i in range(4):
        b_normal = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "subsample": 0.8, "seed": i}, dtrain, num_boost_round=2)
        detector.fit_or_evaluate(b_normal, num_features=20)
        
    b_normal = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "subsample": 0.8, "seed": 42}, dtrain, num_boost_round=2)
    is_normal, normal_score = detector.fit_or_evaluate(b_normal, num_features=20)
    
    X_poison, y_poison = make_classification(n_samples=50, n_features=20, random_state=99)
    X_poison[:, 19] = y_poison * 9999 # Poisoning
    dtrain_poison = xgb.DMatrix(X_poison, label=y_poison)
    b_poison = xgb.train({"objective": "binary:logistic", "tree_method": "hist"}, dtrain_poison, num_boost_round=2)
    
    is_poison_normal, poison_score = detector.fit_or_evaluate(b_poison, num_features=20)
    
    return {
        "normal_client_score": float(normal_score),
        "normal_client_accepted": bool(is_normal),
        "poisoned_client_score": float(poison_score),
        "poisoned_client_accepted": bool(is_poison_normal)
    }

# -----------------
# PREDICTION INFERENCE
# -----------------
# Global model cache to save RAM on repeated requests
_global_model_cache = None

@app.post("/api/predict")
async def predict_csv(file: UploadFile = File(...)):
    global _global_model_cache
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
        
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"Error parsing CSV: {e}")
        
    if len(df) == 0:
        raise HTTPException(400, "CSV is empty")
        
    # Drop labels if they exist
    features = [c for c in df.columns if c not in ["label", "label_encoded"]]
    X = df[features].to_numpy(dtype=np.float32)
    
    # Load model
    if _global_model_cache is None:
        model_path = RESULTS_DIR / "models" / "fl_xgboost_final_model.pkl"
        if not model_path.exists():
            raise HTTPException(500, "Global federated model not found")
        with open(model_path, "rb") as f:
            raw_model = pickle.load(f)
            
        # Reconstruct boosters
        _global_model_cache = []
        for round_data in raw_model:
            boosters = []
            for b_bytes in round_data:
                b = xgb.Booster()
                b.load_model(b_bytes)
                boosters.append(b)
            _global_model_cache.append(boosters)
            
    # Load reverse label mapping
    mapping_path = ROOT / "results" / "baseline" / "model" / "label_mapping.json"
    with open(mapping_path, "r") as f:
        fwd_map = json.load(f)
        rev_map = {v: k for k, v in fwd_map.items()}
        
    num_classes = 21
    preds_proba = predict_global(_global_model_cache, X, num_classes)
    pred_idx = np.argmax(preds_proba, axis=1)
    confidences = np.max(preds_proba, axis=1)
    
    labels = [rev_map[i] for i in pred_idx]
    
    # Prepare limited display results (max 100 rows to prevent browser crash)
    display_rows = []
    for i in range(min(100, len(labels))):
        display_rows.append({
            "index": i,
            "prediction": labels[i],
            "confidence": float(confidences[i])
        })
        
    # Count distribution
    distribution = pd.Series(labels).value_counts().to_dict()
    
    return {
        "filename": file.filename,
        "total_rows_processed": len(labels),
        "features_detected": len(features),
        "distribution": distribution,
        "sample_predictions": display_rows
    }
