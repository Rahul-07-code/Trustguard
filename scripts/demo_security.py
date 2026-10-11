import sys
from pathlib import Path
import os
import time
import xgboost as xgb
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.security.aes import AESGCMEncryption
from src.security.hmac_auth import HMACAuth
from src.anomaly.isolation_forest import UpdateAnomalyDetector

def demo_aes_hmac():
    print("========================================")
    print("      AES & HMAC SECURITY DEMO")
    print("========================================")
    
    aes_key = os.urandom(32)
    hmac_key = os.urandom(32)
    
    aes = AESGCMEncryption(aes_key)
    hmac_auth = HMACAuth(hmac_key)
    
    original_payload = b"Federated XGBoost Tree Update: {leaf_weights: [0.1, -0.2, 0.5]}"
    print(f"\n[1] ORIGINAL PLAINTEXT PAYLOAD:\n{original_payload}")
    
    # 1. Encrypt
    nonce, ciphertext = aes.encrypt(original_payload, b"trustguard-fl")
    print(f"\n[2] AES ENCRYPTED CIPHERTEXT (hex):\n{ciphertext.hex()[:60]}... (truncated)")
    
    # 2. Sign
    signature = hmac_auth.sign(nonce + ciphertext)
    print(f"\n[3] HMAC-SHA-256 SIGNATURE (hex):\n{signature.hex()}")
    
    # 3. Verify & Decrypt (Normal)
    is_valid = hmac_auth.verify(nonce + ciphertext, signature)
    if is_valid:
        decrypted = aes.decrypt(nonce, ciphertext, b"trustguard-fl")
        print(f"\n[4] VERIFICATION PASS! DECRYPTED:\n{decrypted}")
    
    # 4. Tamper Test
    print("\n--- INITIATING TAMPER ATTACK ---")
    tampered_ct = bytearray(ciphertext)
    tampered_ct[5] ^= 0xFF  # Flip bits in ciphertext
    
    print("[5] Tampered ciphertext received by Server.")
    tamper_valid = hmac_auth.verify(nonce + bytes(tampered_ct), signature)
    if not tamper_valid:
        print("[6] HMAC VERIFICATION FAILED! Payload was modified in transit. Update Rejected.")
    else:
        print("[6] HMAC VERIFICATION PASSED! (This should not happen)")


def demo_isolation_forest():
    print("\n========================================")
    print("     ISOLATION FOREST ANOMALY DEMO")
    print("========================================")
    
    detector = UpdateAnomalyDetector(contamination=0.1)
    
    # Create a normal booster (simulating regular client training)
    print("Training 5 NORMAL client updates to fit Isolation Forest...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    for i in range(5):
        dtrain = xgb.DMatrix(X, label=y)
        b = xgb.train({"objective": "binary:logistic", "tree_method": "hist"}, dtrain, num_boost_round=2)
        is_normal, score = detector.fit_or_evaluate(b, num_features=20)
        print(f"Normal Update {i+1} -> Accepted: {is_normal} (Score: {score:.3f})")
        
    # Create an ANOMALOUS booster (simulating data poisoning attack)
    print("\nSimulating Data Poisoning Attack (Client sends bizarre update)...")
    X_poison, y_poison = make_classification(n_samples=100, n_features=20, random_state=999)
    # Give extreme weight to a usually useless feature
    X_poison[:, 19] = y_poison * 1000 
    
    dtrain_poison = xgb.DMatrix(X_poison, label=y_poison)
    b_poison = xgb.train({"objective": "binary:logistic", "tree_method": "hist"}, dtrain_poison, num_boost_round=2)
    
    is_normal, score = detector.fit_or_evaluate(b_poison, num_features=20)
    print(f"Malicious Update -> Accepted: {is_normal} (Score: {score:.3f})")
    
    if not is_normal:
        print(">> ISOLATION FOREST SUCCESSFULLY BLOCKED THE POISONED UPDATE! <<")

if __name__ == "__main__":
    demo_aes_hmac()
    demo_isolation_forest()
