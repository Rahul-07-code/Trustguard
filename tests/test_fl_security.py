"""TrustGuard-FL Security & Networking Regression Test Suite.

Tests:
1. AES-256-GCM encryption/decryption round trip
2. HMAC-SHA-256 integrity and bit-flip tamper detection
3. Nonce-based replay protection and duplicate submission rejection
4. Token-based client authentication and invalid credential rejection
5. Structured JSONL audit ledger persistence and query filtering
6. End-to-end client submission through coordinator pipeline
"""

import time
import os
import pickle
import pytest
from fastapi.testclient import TestClient
import xgboost as xgb
import numpy as np

from api.main import app
from src.security.aes import AESGCMEncryption
from src.security.hmac_auth import HMACAuth
from src.fl.state import state_manager
from src.fl.audit import audit_logger
from src.fl.server import fl_coordinator
from src.fl.protocol import EncryptedUpdateEnvelope

client = TestClient(app)


def test_aes_encryption_roundtrip():
    key = os.urandom(32)
    aes = AESGCMEncryption(key)
    plaintext = b"Test booster byte stream 12345"
    
    nonce, ct = aes.encrypt(plaintext, b"trustguard-fl")
    assert len(nonce) == 12
    assert ct != plaintext
    
    decrypted = aes.decrypt(nonce, ct, b"trustguard-fl")
    assert decrypted == plaintext


def test_hmac_tamper_rejection():
    key = os.urandom(32)
    hmac = HMACAuth(key)
    payload = b"Sample update payload"
    
    sig = hmac.sign(payload)
    assert hmac.verify(payload, sig) is True
    
    # Tamper with single byte
    tampered = bytearray(payload)
    tampered[0] ^= 0xFF
    assert hmac.verify(bytes(tampered), sig) is False


def test_client_registration_and_auth():
    cid = f"test-client-{int(time.time())}"
    res = client.post("/api/fl/register", json={
        "client_id": cid,
        "client_name": "Test Node",
        "platform": "pytest",
        "dataset_samples": 100
    })
    assert res.status_code == 200
    data = res.json()
    token = data["auth_token"]
    assert token and len(token) > 20
    
    # Authenticate valid
    assert state_manager.authenticate_client(cid, token) is True
    # Authenticate invalid
    assert state_manager.authenticate_client(cid, "wrong-token") is False
    assert state_manager.authenticate_client("unknown-client", token) is False


def test_replay_protection():
    cid = f"client-replay-{int(time.time())}"
    token, _ = state_manager.register_client(cid)
    nonce = os.urandom(12).hex()
    ts = time.time()
    
    # First time nonce is used -> PASS
    ok, msg = state_manager.check_and_record_nonce(1, cid, nonce, ts)
    assert ok is True
    
    # Second time identical nonce is used in round 1 -> REJECTED
    ok2, msg2 = state_manager.check_and_record_nonce(1, cid, nonce, ts)
    assert ok2 is False
    assert "Replay attack detected" in msg2

    # Old timestamp drift (> 300s) -> REJECTED
    old_ts = time.time() - 400
    fresh_nonce = os.urandom(12).hex()
    ok3, msg3 = state_manager.check_and_record_nonce(1, cid, fresh_nonce, old_ts)
    assert ok3 is False
    assert "Timestamp drift exceeds window" in msg3


def test_audit_logger_persistence():
    evt = audit_logger.record_event(
        security_layer="TEST",
        event_type="unit_test_event",
        outcome="PASS",
        severity="INFO",
        client_id="pytest-client",
        details={"test_metric": 42}
    )
    assert evt["event_id"].startswith("EVT-")
    
    events, total = audit_logger.query_events(client_id="pytest-client")
    assert total >= 1
    assert any(e["event_id"] == evt["event_id"] for e in events)


def test_end_to_end_submission_pipeline():
    cid = f"client-e2e-{int(time.time())}"
    reg = client.post("/api/fl/register", json={
        "client_id": cid,
        "client_name": "E2E Node",
        "platform": "pytest"
    }).json()
    token = reg["auth_token"]
    
    # Train dummy tiny booster
    X = np.random.randn(50, 46).astype(np.float32)
    y = np.random.randint(0, 21, size=50).astype(np.int32)
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"objective": "multi:softprob", "num_class": 21, "tree_method": "hist", "max_depth": 2}, dtrain, num_boost_round=1)
    b_bytes = booster.save_raw()
    
    # Encrypt with server's active keys
    plaintext = pickle.dumps({"booster": b_bytes, "client_id": cid})
    nonce_bytes, ct_bytes = fl_coordinator.aes.encrypt(plaintext, b"trustguard-fl")
    sig_bytes = fl_coordinator.hmac.sign(nonce_bytes + ct_bytes)
    
    # Valid submission
    envelope = {
        "round_id": 1,
        "client_id": cid,
        "auth_token": token,
        "timestamp": time.time(),
        "nonce_hex": nonce_bytes.hex(),
        "ciphertext_hex": ct_bytes.hex(),
        "signature_hex": sig_bytes.hex()
    }
    
    sub_res = client.post("/api/fl/rounds/1/submit", json=envelope)
    assert sub_res.status_code == 200
    outcome = sub_res.json()
    assert outcome["success"] is True
    assert outcome["status"] == "ACCEPTED"
    
    # Tampered submission with a second client (to test HMAC rejection specifically)
    cid2 = f"client-tampered-{int(time.time())}"
    reg2 = client.post("/api/fl/register", json={"client_id": cid2, "client_name": "Tampered Node"}).json()
    token2 = reg2["auth_token"]
    
    tampered_ct = bytearray(ct_bytes)
    tampered_ct[2] ^= 0xFF
    envelope_tampered = {
        "round_id": 1,
        "client_id": cid2,
        "auth_token": token2,
        "timestamp": time.time(),
        "nonce_hex": os.urandom(12).hex(),
        "ciphertext_hex": bytes(tampered_ct).hex(),
        "signature_hex": sig_bytes.hex(), # signature of untampered bytes
    }
    
    res_tampered = client.post("/api/fl/rounds/1/submit", json=envelope_tampered)
    assert res_tampered.status_code == 200
    outcome_tampered = res_tampered.json()
    assert outcome_tampered["success"] is False
    assert outcome_tampered["status"] == "REJECTED_HMAC"
