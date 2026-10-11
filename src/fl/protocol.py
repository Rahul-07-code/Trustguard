"""TrustGuard-FL Wire Protocol and Schemas.

Defines Pydantic models and dataclasses for:
- Client registration & authentication tokens
- Round coordination and assignments
- Cryptographically sealed update envelopes (AES-GCM + HMAC-SHA-256)
- Submission responses and verification outcomes
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class ClientRegistrationRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier, e.g. client-01")
    client_name: str = Field("Edge Client", description="Human-readable node label")
    platform: str = Field("Windows/Linux", description="Client OS or environment")
    dataset_samples: int = Field(0, description="Number of training rows on client")


class ClientRegistrationResponse(BaseModel):
    client_id: str
    auth_token: str
    status: str = "REGISTERED"
    registered_at: str
    message: str = "Client registered successfully"


class RoundAssignment(BaseModel):
    round_id: int
    client_id: str
    boost_rounds: int = 5
    num_classes: int = 21
    max_depth: int = 4
    learning_rate: float = 0.1
    device: str = "cpu"
    status: str = "ASSIGNED"


class EncryptedUpdateEnvelope(BaseModel):
    round_id: int = Field(..., description="Current federated round ID")
    client_id: str = Field(..., description="Authenticated client identifier")
    auth_token: str = Field(..., description="Client authentication token")
    timestamp: float = Field(..., description="Client UTC timestamp to enforce freshness")
    nonce_hex: str = Field(..., description="12-byte initialization vector (AES-GCM)")
    ciphertext_hex: str = Field(..., description="AES-256-GCM ciphertext containing serialized booster")
    signature_hex: str = Field(..., description="HMAC-SHA-256 signature of (nonce + ciphertext)")


class SubmissionOutcome(BaseModel):
    success: bool
    event_id: str
    round_id: int
    client_id: str
    status: str  # ACCEPTED, REJECTED_AUTH, REJECTED_HMAC, REJECTED_REPLAY, REJECTED_DECRYPT, REJECTED_ANOMALY
    message: str
    anomaly_score: Optional[float] = None
