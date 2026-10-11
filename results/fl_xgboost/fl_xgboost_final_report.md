# TrustGuard-FL Federated XGBoost Final Report

## 1. Environment
- Python: 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)]
- XGBoost version: 2.1.4
- GPU: True (Device: cuda)

## 2. Dataset
- Dataset path: C:\Users\rahul\OneDrive\Desktop\TrustAI\minorTrust\TrustGuard-FL\dataset\CICIOT23_21CLASSES
- Files: train_21.csv, validation_21.csv, test_21.csv
- Total rows: 7703943
- Features: 46
- Classes: 21
- Train rows: 5392858
- Validation rows: 1155488
- Test rows: 1155597

## 3. Federated Configuration
- Clients: 3
- Rounds: 5
- Local boosting rounds: 5
- Aggregation method: Federated Gradient Boosting via Margin Sharing (Sub-Ensemble Averaging)

## 4. XGBoost
- Parameters: max_depth=4, learning_rate=0.1, tree_method=hist, device=cuda
- GPU/CPU: cuda

## 5. Isolation Forest
- Purpose: malicious/outlier client update detection
- Parameters: contamination=0.05
- Features evaluated: Extracted XGBoost Feature Importance (Gain) vector from the received update.
- Decisions: Normal updates are accepted; anomalous updates are rejected and quarantined.

## 6. AES
- Algorithm/mode: AES-256-GCM
- Payload protected: Serialized XGBoost Booster artifact
- Validation: PASS

## 7. HMAC
- Algorithm: HMAC-SHA-256
- Verification: PASS
- Tamper test: PASS

## 8. TLS
- TLS Server used for client communication: PASS
- Handshake validation: PASS

## 9. Federated Results
- Final Test Accuracy: 0.9990
- Final Test Macro F1: 0.9870
- Final Test Weighted F1: 0.9990

## 10. Memory
- Peak RAM: 7.01 GB
- Maximum allowed RAM: 14.0 GB
- Status: PASS

## 11. Artifact Validation
- Final model exists: PASS
- Metrics saved: PASS
- Security report saved: PASS

## 12. Final Status
DATASET COMPLETE: PASS
FEDERATED XGBOOST: PASS
ISOLATION FOREST: PASS
AES: PASS
HMAC: PASS
TLS: PASS
GPU: PASS
RAM < 14 GB: PASS
FINAL MODEL LOADABLE: PASS
END-TO-END PIPELINE: PASS
