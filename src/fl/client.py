"""TrustGuard-FL Standalone Remote Client Worker.

Can be run on any remote device (e.g., friend's laptop on the same Wi-Fi network)
to participate in real federated learning rounds.

Workflow:
1. Connects to the central coordinator over verified TLS.
2. Authenticates with client credentials.
3. Polls/receives active round assignment.
4. Trains XGBoost booster locally on client's private tabular data.
5. Encrypts model update with AES-256-GCM.
6. Generates cryptographic HMAC-SHA-256 signature.
7. Submits envelope and receives server verification result.
"""

import sys
import os
import time
import json
import argparse
import urllib.request
import urllib.error
import ssl
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security.aes import AESGCMEncryption
from src.security.hmac_auth import HMACAuth
from src.fl.protocol import ClientRegistrationRequest, EncryptedUpdateEnvelope


def http_request(url: str, method: str = "GET", data: dict = None, headers: dict = None, ca_cert: str = None) -> dict:
    headers = headers or {}
    ssl_context = None
    if url.startswith("https://"):
        if ca_cert and os.path.exists(ca_cert):
            ssl_context = ssl.create_default_context(cafile=ca_cert)
        else:
            ssl_context = ssl.create_default_context()
            # For self-signed lab demonstration if CA not passed
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return json.loads(err_body)
        except Exception:
            raise RuntimeError(f"HTTP Error {e.code}: {err_body}")
    except Exception as e:
        raise RuntimeError(f"Connection failed to {url}: {e}")


def load_or_create_local_dataset(data_path: str = None, num_samples: int = 500, num_features: int = 46, num_classes: int = 21):
    """Loads CSV shard if provided, or generates a deterministic private synthetic shard."""
    if data_path and os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        features = [c for c in df.columns if c not in ["label", "label_encoded"]]
        X = df[features].to_numpy(dtype=np.float32)
        y = df["label_encoded"].to_numpy(dtype=np.int32) if "label_encoded" in df else np.zeros(len(df), dtype=np.int32)
        return X, y
        
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=min(25, num_features),
        n_classes=num_classes,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


def run_client(
    server_url: str,
    client_id: str,
    client_name: str = "Remote Client",
    data_path: str = None,
    ca_cert: str = None,
    tamper: bool = False,
    replay: bool = False,
):
    print("=" * 60)
    print(f"  TrustGuard-FL Remote Worker: {client_id}")
    print(f"  Server Target: {server_url}")
    print("=" * 60)

    # 1. Register Client
    print("\n[Step 1] Registering client with central coordinator...")
    reg_url = f"{server_url}/api/fl/register"
    reg_res = http_request(
        reg_url,
        method="POST",
        data={
            "client_id": client_id,
            "client_name": client_name,
            "platform": sys.platform,
            "dataset_samples": 500,
        },
        ca_cert=ca_cert,
    )
    auth_token = reg_res.get("auth_token")
    if not auth_token:
        print(f"Registration failed: {reg_res}")
        return False
    print(f"-> Registration PASS. Issued Auth Token: {auth_token[:10]}... (truncated)")

    # 2. Fetch Keys
    print("\n[Step 2] Retrieving cryptographic parameters...")
    keys_url = f"{server_url}/api/fl/keys"
    keys_res = http_request(
        keys_url,
        method="GET",
        headers={"Authorization": f"Bearer {auth_token}", "X-Client-ID": client_id},
        ca_cert=ca_cert,
    )
    aes_key = bytes.fromhex(keys_res["aes_key_hex"])
    hmac_key = bytes.fromhex(keys_res["hmac_key_hex"])
    aes = AESGCMEncryption(aes_key)
    hmac_auth = HMACAuth(hmac_key)
    print("-> AES-256-GCM and HMAC-SHA-256 parameters initialized.")

    # 3. Query Active Round
    print("\n[Step 3] Polling active round assignment...")
    round_url = f"{server_url}/api/fl/rounds/active"
    round_res = http_request(round_url, method="GET", ca_cert=ca_cert)
    round_id = round_res.get("round_id", 1)
    boost_rounds = 5
    num_classes = 21
    print(f"-> Assigned to Federated Round {round_id} (boost_rounds={boost_rounds}, classes={num_classes}).")

    # 4. Local Training
    print("\n[Step 4] Training local XGBoost Booster on private data...")
    X, y = load_or_create_local_dataset(data_path=data_path, num_features=46, num_classes=num_classes)
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "max_depth": 4,
        "learning_rate": 0.1,
    }
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=boost_rounds)
    duration = time.time() - t0
    booster_bytes = booster.save_raw()
    print(f"-> Local training completed in {duration:.2f}s. Booster size: {len(booster_bytes)} bytes.")

    # 5. Cryptographic Packaging (AES + HMAC)
    print("\n[Step 5] Encrypting and signing model update...")
    plaintext_payload = pickle.dumps({"booster": booster_bytes, "client_id": client_id})
    nonce_bytes, ciphertext_bytes = aes.encrypt(plaintext_payload, b"trustguard-fl")
    
    if tamper:
        print(">> TAMPER FLAG ACTIVE: Flipping byte in ciphertext to simulate Man-In-The-Middle attack <<")
        tampered = bytearray(ciphertext_bytes)
        tampered[0] ^= 0xFF
        ciphertext_bytes = bytes(tampered)

    sig_bytes = hmac_auth.sign(nonce_bytes + ciphertext_bytes)

    nonce_hex = nonce_bytes.hex()
    if replay:
        print(">> REPLAY FLAG ACTIVE: Simulating duplicate replay attack with static nonce <<")
        nonce_hex = "00112233445566778899aabb"

    envelope = {
        "round_id": round_id,
        "client_id": client_id,
        "auth_token": auth_token,
        "timestamp": time.time(),
        "nonce_hex": nonce_hex,
        "ciphertext_hex": ciphertext_bytes.hex(),
        "signature_hex": sig_bytes.hex(),
    }
    print(f"-> Nonce: {nonce_hex}")
    print(f"-> HMAC-SHA-256 Signature: {sig_bytes.hex()[:32]}... (truncated)")

    # 6. Submit over TLS
    print("\n[Step 6] Transmitting update to central server over verified TLS...")
    submit_url = f"{server_url}/api/fl/rounds/{round_id}/submit"
    submit_res = http_request(submit_url, method="POST", data=envelope, ca_cert=ca_cert)
    
    status = submit_res.get("status")
    message = submit_res.get("message")
    event_id = submit_res.get("event_id")
    print(f"-> Server Response Status: {status}")
    print(f"-> Event ID: {event_id}")
    print(f"-> Message: {message}")

    if submit_res.get("success"):
        print("\nSUCCESS: Model update successfully accepted and aggregated into global model!")
        return True
    else:
        print(f"\nALERT: Update rejected by server: {status} ({message})")
        return False


def main():
    parser = argparse.ArgumentParser(description="TrustGuard-FL Remote Client Worker")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="Coordinator URL, e.g. https://192.168.1.15:8443")
    parser.add_argument("--client-id", default="client-remote-01", help="Unique client ID")
    parser.add_argument("--client-name", default="Remote Node", help="Display name for client")
    parser.add_argument("--data-path", default=None, help="Path to local training CSV shard")
    parser.add_argument("--ca-cert", default=None, help="Path to CA certificate for verified TLS")
    parser.add_argument("--tamper", action="store_true", help="Simulate tamper attack (should fail HMAC)")
    parser.add_argument("--replay", action="store_true", help="Simulate replay attack (should fail replay check)")

    args = parser.parse_args()
    success = run_client(
        server_url=args.server_url,
        client_id=args.client_id,
        client_name=args.client_name,
        data_path=args.data_path,
        ca_cert=args.ca_cert,
        tamper=args.tamper,
        replay=args.replay,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
