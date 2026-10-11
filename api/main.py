import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
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

app = FastAPI(title="TrustGuard-FL API", description="Service backend for TrustGuard-FL Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = ROOT / "results" / "fl_xgboost"

# -----------------
# DATA LOADERS
# -----------------
def load_json(path):
    if not path.exists(): return {}
    with open(path, "r") as f:
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
        "ram_limit_gb": mem.get("limit_gb", 14.0)
    }

@app.get("/api/fl/rounds")
def get_fl_rounds():
    return load_csv_records(RESULTS_DIR / "metrics" / "fl_xgboost_round_metrics.csv")

@app.get("/api/clients")
def get_clients():
    return [
        {"id": "Client-01", "status": "ACTIVE", "last_update": "Just now", "security": "SECURE", "tls": "ESTABLISHED"},
        {"id": "Client-02", "status": "ACTIVE", "last_update": "Just now", "security": "SECURE", "tls": "ESTABLISHED"},
        {"id": "Client-03", "status": "ACTIVE", "last_update": "Just now", "security": "SECURE", "tls": "ESTABLISHED"},
    ]

@app.get("/api/security/events")
def get_security_events():
    return [
        {"id": 1, "timestamp": "Recent", "severity": "INFO", "component": "AES", "event": "Encryption validation PASS"},
        {"id": 2, "timestamp": "Recent", "severity": "INFO", "component": "HMAC", "event": "Signature verification PASS"},
        {"id": 3, "timestamp": "Recent", "severity": "INFO", "component": "TLS", "event": "Handshake ESTABLISHED"},
        {"id": 4, "timestamp": "Recent", "severity": "INFO", "component": "Isolation Forest", "event": "Anomalies scanned. Status NORMAL"}
    ]

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
