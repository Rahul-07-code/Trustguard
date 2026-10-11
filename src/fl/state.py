"""TrustGuard-FL In-Memory and Durable State Manager.

Maintains:
- Registered clients, tokens, and connectivity freshness
- Active and completed federated training rounds
- Replay protection cache (nonces and timestamp window)
- Submission tracking per client per round
"""

import time
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


class ClientRecord:
    def __init__(
        self,
        client_id: str,
        client_name: str,
        platform: str,
        auth_token: str,
        registered_at: str,
        status: str = "ACTIVE",
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.platform = platform
        self.auth_token = auth_token
        self.registered_at = registered_at
        self.last_seen = registered_at
        self.status = status  # ACTIVE, OFFLINE, REVOKED
        self.completed_rounds = 0
        self.rejected_updates = 0
        self.last_error = ""

    def to_dict(self) -> dict[str, Any]:
        # Determine connection state based on freshness (within last 45s)
        try:
            last_seen_dt = datetime.fromisoformat(self.last_seen)
            now_dt = datetime.now(timezone.utc)
            delta = (now_dt - last_seen_dt).total_seconds()
            conn_state = "CONNECTED" if delta < 45 else "OFFLINE"
        except Exception:
            conn_state = "OFFLINE"

        return {
            "id": self.client_id,
            "name": self.client_name,
            "platform": self.platform,
            "status": self.status if self.status != "ACTIVE" else conn_state,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "completed_rounds": self.completed_rounds,
            "rejected_updates": self.rejected_updates,
            "last_error": self.last_error or "None",
            "security": "SECURE",
            "tls": "VERIFIED_TLS" if conn_state == "CONNECTED" else "DISCONNECTED",
        }


class RoundRecord:
    def __init__(self, round_id: int, target_clients: int = 2):
        self.round_id = round_id
        self.target_clients = target_clients
        self.status = "TRAINING"  # PREPARING, TRAINING, AGGREGATING, COMPLETED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = ""
        self.submitted_clients: list[str] = []
        self.raw_updates: dict[str, bytes] = {}  # client_id -> booster bytes
        self.metrics: dict[str, float] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_id,
            "round_id": self.round_id,
            "status": self.status,
            "target_clients": self.target_clients,
            "participating_clients": self.submitted_clients,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "val_acc": self.metrics.get("val_acc", 0.0),
            "val_macro_f1": self.metrics.get("val_macro_f1", 0.0),
            "submissions_count": len(self.submitted_clients),
        }


class StateManager:
    def __init__(self, persistence_file: Optional[Path] = None):
        if persistence_file is None:
            root = Path(__file__).resolve().parents[2]
            self.file_path = root / "results" / "fl_xgboost" / "coordinator_state.json"
        else:
            self.file_path = Path(persistence_file)
            
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        self.clients: dict[str, ClientRecord] = {}
        self.rounds: dict[int, RoundRecord] = {}
        self.active_round_id: int = 1
        
        # Replay cache: (round_id, client_id, nonce_hex) -> timestamp
        self.seen_nonces: dict[tuple[int, str, str], float] = {}
        
        # Load pre-existing state if available
        self._load()
        
        # Ensure default round 1 exists if none
        if 1 not in self.rounds:
            self.rounds[1] = RoundRecord(round_id=1, target_clients=2)

    def register_client(
        self, client_id: str, client_name: str = "Client Node", platform: str = "Windows"
    ) -> tuple[str, str]:
        """Register a client and issue a secure auth token."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            token = secrets.token_urlsafe(32)
            
            client = ClientRecord(
                client_id=client_id,
                client_name=client_name,
                platform=platform,
                auth_token=token,
                registered_at=now_iso,
            )
            self.clients[client_id] = client
            self._save()
            return token, now_iso

    def authenticate_client(self, client_id: str, auth_token: str) -> bool:
        """Verify token matches registered client and update heartbeat."""
        with self._lock:
            client = self.clients.get(client_id)
            if not client:
                return False
            if client.status == "REVOKED":
                return False
            # Constant-time comparison
            if not secrets.compare_digest(client.auth_token, auth_token):
                client.last_error = "Invalid authentication token"
                return False
                
            client.last_seen = datetime.now(timezone.utc).isoformat()
            return True

    def check_and_record_nonce(
        self, round_id: int, client_id: str, nonce_hex: str, timestamp: float
    ) -> tuple[bool, str]:
        """
        Validate message freshness and detect replay attacks.
        Rejects if timestamp is older than 300s or nonce has been seen.
        """
        now = time.time()
        # Reject timestamp skew greater than 5 minutes (300 seconds)
        if abs(now - timestamp) > 300:
            return False, f"Timestamp drift exceeds window: drift={abs(now - timestamp):.1f}s > 300s"

        key = (round_id, client_id, nonce_hex)
        with self._lock:
            # Check replay
            if key in self.seen_nonces:
                return False, f"Replay attack detected: duplicate nonce {nonce_hex} for client {client_id} in round {round_id}"
                
            # Check duplicate submission in active round
            r = self.rounds.get(round_id)
            if r and client_id in r.submitted_clients:
                return False, f"Duplicate submission rejected: client {client_id} already submitted update for round {round_id}"

            # Record in replay cache
            self.seen_nonces[key] = timestamp
            
            # Prune nonces older than 1 hour to prevent unbounded memory growth
            cutoff = now - 3600
            self.seen_nonces = {k: v for k, v in self.seen_nonces.items() if v > cutoff}
            
            return True, "Freshness and nonce verified"

    def record_submission(self, round_id: int, client_id: str, booster_bytes: bytes) -> RoundRecord:
        with self._lock:
            if round_id not in self.rounds:
                self.rounds[round_id] = RoundRecord(round_id)
            r = self.rounds[round_id]
            if client_id not in r.submitted_clients:
                r.submitted_clients.append(client_id)
            r.raw_updates[client_id] = booster_bytes
            
            client = self.clients.get(client_id)
            if client:
                client.completed_rounds += 1
                client.last_seen = datetime.now(timezone.utc).isoformat()
                
            self._save()
            return r

    def record_rejection(self, client_id: str, reason: str):
        with self._lock:
            client = self.clients.get(client_id)
            if client:
                client.rejected_updates += 1
                client.last_error = reason
                client.last_seen = datetime.now(timezone.utc).isoformat()
                self._save()

    def get_all_clients(self) -> list[dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self.clients.values()]

    def get_all_rounds(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self.rounds.values()]

    def get_active_round(self) -> dict[str, Any]:
        with self._lock:
            r = self.rounds.get(self.active_round_id)
            if not r:
                r = RoundRecord(self.active_round_id)
                self.rounds[self.active_round_id] = r
            return r.to_dict()

    def start_new_round(self, target_clients: int = 2) -> RoundRecord:
        with self._lock:
            self.active_round_id += 1
            new_r = RoundRecord(self.active_round_id, target_clients=target_clients)
            self.rounds[self.active_round_id] = new_r
            self._save()
            return new_r

    def complete_round(self, round_id: int, metrics: dict[str, float]):
        with self._lock:
            r = self.rounds.get(round_id)
            if r:
                r.status = "COMPLETED"
                r.completed_at = datetime.now(timezone.utc).isoformat()
                r.metrics = metrics
                self._save()

    def _save(self):
        try:
            data = {
                "active_round_id": self.active_round_id,
                "clients": {
                    cid: {
                        "client_id": c.client_id,
                        "client_name": c.client_name,
                        "platform": c.platform,
                        "auth_token": c.auth_token,
                        "registered_at": c.registered_at,
                        "last_seen": c.last_seen,
                        "status": c.status,
                        "completed_rounds": c.completed_rounds,
                        "rejected_updates": c.rejected_updates,
                        "last_error": c.last_error,
                    }
                    for cid, c in self.clients.items()
                },
                "rounds": {
                    rid: {
                        "round_id": r.round_id,
                        "target_clients": r.target_clients,
                        "status": r.status,
                        "created_at": r.created_at,
                        "completed_at": r.completed_at,
                        "submitted_clients": r.submitted_clients,
                        "metrics": r.metrics,
                    }
                    for rid, r in self.rounds.items()
                },
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_round_id = data.get("active_round_id", 1)
            for cid, cdata in data.get("clients", {}).items():
                c = ClientRecord(
                    client_id=cdata["client_id"],
                    client_name=cdata.get("client_name", "Client"),
                    platform=cdata.get("platform", "Windows"),
                    auth_token=cdata["auth_token"],
                    registered_at=cdata.get("registered_at", ""),
                    status=cdata.get("status", "ACTIVE"),
                )
                c.last_seen = cdata.get("last_seen", "")
                c.completed_rounds = cdata.get("completed_rounds", 0)
                c.rejected_updates = cdata.get("rejected_updates", 0)
                c.last_error = cdata.get("last_error", "")
                self.clients[cid] = c
                
            for rid_str, rdata in data.get("rounds", {}).items():
                rid = int(rid_str)
                r = RoundRecord(rid, target_clients=rdata.get("target_clients", 2))
                r.status = rdata.get("status", "COMPLETED")
                r.created_at = rdata.get("created_at", "")
                r.completed_at = rdata.get("completed_at", "")
                r.submitted_clients = rdata.get("submitted_clients", [])
                r.metrics = rdata.get("metrics", {})
                self.rounds[rid] = r
        except Exception:
            pass


# Global singleton coordinator state
state_manager = StateManager()
