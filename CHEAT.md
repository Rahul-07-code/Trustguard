# TrustGuard-FL CHEAT SHEET

## 1. Project in One Paragraph
TrustGuard-FL implements an end-to-end Federated XGBoost pipeline enforcing strict data security and update verification. Client nodes train XGBoost tree updates iteratively and securely transmit them to a central server using TLS, AES-256-GCM encryption, and HMAC-SHA-256 authentication. The server evaluates inbound updates using an Isolation Forest anomaly detector to filter out poisoned or malicious trees, and aggregates the safe updates using Sub-Ensemble Averaging (Margin Sharing) without exceeding a 14 GB memory footprint.

## 2. Repository Structure

The actual architecture of the repository:

- `src/` -> Core Python modules
  - `security/` -> Cryptography implementations (`aes.py`, `hmac_auth.py`, `tls_utils.py`)
  - `anomaly/` -> Attack detection (`isolation_forest.py`)
- `scripts/` -> Utility and demonstration scripts (e.g., `demo_security.py`)
- `dataset/` -> Contains the CICIOT23_21CLASSES federated data
- `results/fl_xgboost/` -> Final experiment outputs, models, manifests, and metrics
- `run_fl_xgboost.py` -> The main entry point that runs the full pipeline

## 3. Important Files

| File | Purpose | When I need to run/use it |
|------|---------|---------------------------|
| `run_fl_xgboost.py` | **MAIN ENTRY POINT**. Orchestrates data loading, starts TLS server, handles FL loop, and saves results. | Run this to reproduce the entire Federated training pipeline. |
| `scripts/demo_security.py` | Standalone demonstration script for AES, HMAC, and Isolation Forest. | Use this in front of your mentor to prove the security components work on demand. |
| `src/security/aes.py` | Implements AES-256-GCM encryption/decryption. | Used internally by the client/server pipeline. |
| `src/security/hmac_auth.py` | Implements HMAC-SHA-256 integrity checks. | Used internally by the client/server pipeline. |
| `src/security/tls_utils.py` | TLS server and client setup (generating self-signed RSA certificates). | Used internally by the client/server pipeline. |
| `src/anomaly/isolation_forest.py`| Implements the Anomaly Detector to reject poisoned updates. | Used internally by the server. |
| `results/fl_xgboost_final_report.md`| The official final summary of the system constraints and accuracy. | Show this to your mentor as proof of completion. |

## 4. How to Run

**Training is already completed** and the final model is saved in `results/fl_xgboost/models/`.

**To execute the standalone Security Demonstration (Fastest):**
```bash
python scripts\demo_security.py
```

**To reproduce the full pipeline (Takes ~30 seconds):**
```bash
python run_fl_xgboost.py
```

## 5. System Pipeline

1. **Dataset**: 7.7 Million rows loaded efficiently into RAM.
2. **Client Partitioning**: Data is split among 3 localized clients.
3. **Local XGBoost training**: Clients calculate their gradients using global margins and fit XGBoost trees.
4. **AES**: The client serializes the tree update and encrypts it with AES-256-GCM.
5. **HMAC**: The client signs the encrypted payload for integrity.
6. **TLS**: The payload is securely transported to the server over a TLS socket.
7. **Server Validation**: Server decrypts AES, verifies HMAC, and checks the signature.
8. **Isolation Forest**: The server extracts feature gain vectors from the tree and checks for poisoning.
9. **Federated Aggregation**: Safe updates are appended to the Global Sub-Ensemble.
10. **Evaluation**: The final global model calculates test accuracy on the holdout test set.

## 6. Federated XGBoost

Since XGBoost builds a sequence of trees where each tree corrects the residual of the previous ones, the aggregation strategy used is **Federated Sub-Ensemble Averaging via Margin Sharing**. 
Clients use the current global predictions as their `base_margin` and fit new trees locally. The server collects the raw boosters (tree collections) and averages their raw prediction margins globally. This allows genuine distributed gradient boosting without requiring the C++ gRPC plugin, enabling us to intercept the Python payloads to apply custom encryption.

## 7. AES — How to Demonstrate

**Command:**
```bash
python scripts\demo_security.py
```
**Expected evidence:**
The terminal clearly prints the original plaintext, the AES ciphertext (as a randomized hex string), and the successful decryption back to plaintext. The actual implementation uses this to protect the XGBoost Booster byte arrays.

**Tell your mentor:**
"AES-256-GCM is used here to securely encrypt the serialized XGBoost tree updates before transmission. This ensures that an eavesdropper on the network cannot steal or reverse-engineer the client's trained model parameters."

## 8. HMAC — How to Demonstrate

**Command:**
```bash
python scripts\demo_security.py
```
**Expected evidence:**
The terminal will show a deliberate "Tamper Attack" where bits in the AES ciphertext are flipped. You will see `HMAC VERIFICATION FAILED! Payload was modified in transit. Update Rejected.`

**Tell your mentor:**
"HMAC ensures message integrity and authentication. If the encrypted payload is modified in transit by a Man-in-the-Middle, the calculated HMAC no longer matches the signature, and the Server actively rejects the update."

## 9. TLS — How to Demonstrate

**Command:**
```bash
python run_fl_xgboost.py
```
**Expected evidence:**
During the first 5 seconds of the pipeline execution, `run_fl_xgboost.py` establishes a real TLS Server thread on `127.0.0.1:18443` using a dynamically generated self-signed RSA-2048 certificate. It passes a test payload through the TLS tunnel.
Check `results\fl_xgboost\security\security_validation.json` to see `"TLS_E2E_PASS": true`.

**Tell your mentor:**
"TLS protects the communication channel between the federated clients and the central aggregator. We use Python's built-in SSL context to enforce a secure, encrypted tunnel that works in tandem with our AES payload encryption."

## 10. Isolation Forest — How to Demonstrate

**Command:**
```bash
python scripts\demo_security.py
```
**Expected evidence:**
The demo simulates 5 normal client updates being accepted. Then, it explicitly simulates a Data Poisoning Attack (a client heavily overweights a useless feature). The Isolation Forest outputs `Accepted: False (Score: -0.174)` and blocks the update.

**Tell your mentor:**
"The server extracts the Feature Importance Gain vector from the incoming XGBoost tree update. If a client trains on poisoned data, its tree splits will vastly diverge from the norm. The Isolation Forest detects this structural anomaly and quarantines the malicious update before it can poison the global model."

## 11. One-Minute Mentor Demo

Open your terminal and run:

```bash
# 1. Show the quick, isolated security proofs (AES, HMAC, IF)
python scripts\demo_security.py

# 2. Run the main pipeline to prove end-to-end integration and GPU training
python run_fl_xgboost.py
```

## 12. What to Show on Screen

## Show this to mentor
- [ ] **Final Metrics/Report**: Open `results\fl_xgboost\fl_xgboost_final_report.md`
- [ ] **AES/HMAC/Anomaly Pass**: Output of `python scripts\demo_security.py`
- [ ] **TLS Pass**: Open `results\fl_xgboost\security\security_validation.json`
- [ ] **RAM & GPU**: Mention the terminal output during `run_fl_xgboost.py` (shows `GPU: True` and RAM `< 8 GB`).

## 13. Common Mentor Questions

**Q: Why XGBoost instead of MLP?**
XGBoost handles tabular data (like network intrusion features) vastly better than Neural Networks. It requires less tuning, handles non-linear relationships out-of-the-box, and trains much faster.

**Q: What is Federated Learning?**
Federated Learning is a decentralized training approach where clients train locally on their private data and only share model updates (trees) with the server, never their raw data.

**Q: Is raw client data sent to the server?**
No. Only the serialized XGBoost Booster artifacts (trees and leaf weights) are transmitted.

**Q: What does AES do?**
AES encrypts the XGBoost Booster artifact so that external attackers intercepting the network traffic cannot read the model parameters.

**Q: Why do you need HMAC if you already have AES?**
AES encrypts data, but it doesn't guarantee the data wasn't tampered with. HMAC provides cryptographic proof that the client sent the message and it wasn't altered in transit.

**Q: Why do you need TLS if the payload is encrypted?**
Defense-in-depth. TLS secures the connection layer (preventing replay and downgrade attacks), while AES/HMAC secures the actual payload at the application layer.

**Q: What does Isolation Forest do?**
It identifies anomalous client updates. By checking the distribution of feature importance across trees, it identifies and drops poisoned models that deviate significantly from other clients.

**Q: How does federated aggregation work for XGBoost?**
Since trees cannot be naively averaged like neural network weights, we use Margin Sharing. Clients use the global model's predictions as a `base_margin` and train new trees to fit the residual. The server simply concatenates all client trees into an ensemble and scales their output.

**Q: How did you ensure the complete dataset was used?**
We load all ~7.7 million rows using optimized NumPy arrays. The terminal logs prove that `Total Rows: 7703943` are in memory.

**Q: How did you keep RAM below 14 GB?**
Instead of holding massive DMatrix objects simultaneously, we partition the data and stream it, letting Python's garbage collector reclaim space. Peak memory is around ~8 GB.

**Q: Is GPU training actually being used?**
Yes. `device="cuda"` is explicitly set in the training parameters, and `run_fl_xgboost.py` logs `GPU: True` upon startup.

## 14. Frontend Dashboard

A professional React dashboard has been created to visualize this security pipeline:

1. **Start Backend**: `python -m uvicorn api.main:app --port 8000 --host 0.0.0.0`
2. **Start Frontend**: `cd frontend && npm run dev`

Open [http://localhost:5173](http://localhost:5173) to see the System Health, FL Metrics, and interactive demonstrations for AES, HMAC, and Isolation Forest.
