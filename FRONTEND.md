# TrustGuard-FL Frontend Dashboard

This directory contains the professional cybersecurity frontend and API service for the TrustGuard-FL Federated XGBoost project.

## 1. Architecture

- **Backend / API**: FastAPI (Python)
- **Frontend**: React + TypeScript (Vite)
- **Styling**: Vanilla CSS (CSS Modules)
- **Data Source**: The backend reads directly from the existing completed `results/fl_xgboost/` experiment artifacts to guarantee 100% fidelity to the actual implementation. No fake or mock data is used for training metrics.

## 2. API Endpoints

The FastAPI backend exposes the following critical endpoints:

- `GET /api/status`: Overall system health and FL status.
- `GET /api/fl/rounds`: Historical round metrics (Accuracy, F1).
- `GET /api/clients`: Registered FL clients.
- `GET /api/security/events`: Immutable audit trail of security operations.
- `POST /api/predict`: Validates and runs inference on an uploaded CSV file using the final `fl_xgboost_final_model.pkl`.

**Demo Endpoints (Non-destructive):**
- `POST /api/demo/aes`: Encrypts/decrypts a payload using AES-256-GCM.
- `POST /api/demo/hmac`: Signs and verifies (with a tamper simulation) a payload using HMAC-SHA-256.
- `POST /api/demo/isolation-forest`: Simulates a normal client and a poisoned client update, returning the IF evaluation.

## 3. How to Run

You must run BOTH the backend API and the frontend dashboard.

**Terminal 1: Start the Backend API**
```bash
# In the TrustGuard-FL root directory
python -m uvicorn api.main:app --port 8000 --host 0.0.0.0
```

**Terminal 2: Start the Frontend Dashboard**
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at: [http://localhost:5173](http://localhost:5173)

## 4. Mentor Demonstration Guide

You can now demonstrate the exact security mechanisms interactively via the UI:

1. **Overview Page**: Prove the RAM limits and GPU usage are accurately tracked from the `fl_xgboost_memory_report.json`.
2. **Federated Learning Page**: Show the actual learning curve over 5 rounds from `fl_xgboost_round_metrics.csv`.
3. **Security & Demo Page (CRITICAL)**:
   - Run the **AES-256-GCM** demo to show ciphertext output.
   - Run the **HMAC-SHA-256** demo to show verification and the automated tampering rejection.
   - Run the **Isolation Forest** demo to simulate a data-poisoning attack and prove it gets dropped.
4. **CSV Validation Page**: 
   - Drag and drop `dataset/CICIOT23_21CLASSES/test_21.csv` (or a small sample).
   - The global model will dynamically load and classify the rows. 
   - **Privacy Notice**: This data is parsed entirely in-memory and discarded. It is not saved to disk or added to the training set.
5. **Architecture Page**: Shows the physical separation of local IoT data vs federated updates.

## 5. Security & Privacy Note
Raw organizational / IoT traffic is never sent to the central server. The pipeline uses Sub-Ensemble Margin Sharing to aggregate safe, AES-encrypted, HMAC-signed XGBoost trees locally. The dashboard visualizes this exact flow securely.
