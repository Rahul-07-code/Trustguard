"""TrustGuard-FL Structured Persistent Audit Logger.

Appends structured JSONL events to disk (results/fl_xgboost/security/audit_trail.jsonl).
Guarantees that security events from TLS handshakes, client authentication,
AES decryption, HMAC verification, replay detection, anomaly filtering,
and federated aggregation are recorded immutably and queryable.
"""

import os
import json
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


class AuditLogger:
    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            root = Path(__file__).resolve().parents[2]
            self.log_path = root / "results" / "fl_xgboost" / "security" / "audit_trail.jsonl"
        else:
            self.log_path = Path(log_path)
            
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_event(
        self,
        security_layer: str,
        event_type: str,
        outcome: str,  # PASS, FAIL, REJECTED
        severity: str = "INFO",  # INFO, WARNING, CRITICAL
        client_id: Optional[str] = None,
        round_id: Optional[int] = None,
        correlation_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record an immutable security event."""
        event = {
            "event_id": f"EVT-{uuid.uuid4().hex[:12].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id or uuid.uuid4().hex[:8],
            "security_layer": security_layer,
            "event_type": event_type,
            "outcome": outcome,
            "severity": severity,
            "client_id": client_id or "SYSTEM",
            "round_id": round_id,
            "details": details or {},
        }
        
        line = json.dumps(event) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                
        return event

    def query_events(
        self,
        client_id: Optional[str] = None,
        severity: Optional[str] = None,
        layer: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query stored events with pagination and filtering."""
        if not self.log_path.exists():
            return [], 0

        matched = []
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            if client_id and evt.get("client_id") != client_id:
                                continue
                            if severity and evt.get("severity") != severity:
                                continue
                            if layer and evt.get("security_layer") != layer:
                                continue
                            if outcome and evt.get("outcome") != outcome:
                                continue
                            matched.append(evt)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                return [], 0

        # Most recent first
        matched.reverse()
        total = len(matched)
        paginated = matched[offset : offset + limit]
        return paginated, total


# Global singleton instance
audit_logger = AuditLogger()
