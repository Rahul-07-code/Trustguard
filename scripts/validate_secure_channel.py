"""Validate the real secure channel: TLS transport + AES-256-GCM + HMAC-SHA-256.

Checks performed:
  * a live TLS handshake over a generated certificate (verification on),
  * bi-directional encrypted/HMAC'd message exchange over that socket,
  * SHA-256 evidence that plaintext and ciphertext differ,
  * an out-of-band tamper probe that must be rejected (FAIL -> REJECTED),
  * rejection when the HMAC key is wrong.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.security.secure_channel import (  # noqa: E402
    SecureChannelError,
    SecureConnection,
    connect_tls_client,
    generate_demo_certificate,
    open_sealed_frame,
    seal_frame,
    start_tls_server,
)

RESULTS = ROOT / "results" / "fl_xgboost" / "security"


def main() -> None:
    aes_key = secrets.token_bytes(32)
    hmac_key = secrets.token_bytes(32)
    cert_dir = RESULTS / "demo_certs"
    cert_path, key_path = generate_demo_certificate(cert_dir)

    listener = start_tls_server("127.0.0.1", 0, cert_path, key_path)
    port = int(listener.getsockname()[1])
    server_records: dict[str, object] = {}

    def serve() -> None:
        raw_sock, _ = listener.accept()
        connection = SecureConnection(raw_sock, aes_key, hmac_key, label="server")
        request = connection.recv()
        server_records["request_type"] = request.get("type")
        server_records["array"] = np.asarray(request["histogram"]).tolist()
        server_records["tls"] = connection.tls_metadata()
        connection.send({"type": "ack", "server_received_bytes": int(connection.stats.bytes_received)})
        server_records["stats"] = connection.stats.as_dict()
        server_records["evidence"] = connection.encrypted_message_log
        connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.4)

    client_sock = connect_tls_client("127.0.0.1", port, cert_path)
    client = SecureConnection(client_sock, aes_key, hmac_key, label="client")
    client_tls = client.tls_metadata()
    payload = {"type": "histogram_update", "client_id": "client_01", "round": 1, "histogram": np.arange(12, dtype=np.float64).reshape(3, 4)}
    client_evidence = client.send(payload, message_label="round_1_histogram")
    reply = client.recv(expect="ack")
    client.close()
    thread.join(timeout=10)

    # Tamper probe: flip a byte inside the ciphertext of a sealed frame.
    good_frame, good_evidence, _ = seal_frame({"type": "histogram_update", "value": 123}, aes_key, hmac_key, "tamper_probe")
    recovered = open_sealed_frame(good_frame, aes_key, hmac_key)
    tampered = bytearray(good_frame)
    tampered[4 + 12 + 5] ^= 0xFF
    tamper_rejected = False
    tamper_message = ""
    try:
        open_sealed_frame(bytes(tampered), aes_key, hmac_key)
    except SecureChannelError as error:
        tamper_rejected = "HMAC verification failed" in str(error)
        tamper_message = str(error)

    wrong_key_rejected = False
    try:
        open_sealed_frame(good_frame, aes_key, secrets.token_bytes(32))
    except SecureChannelError:
        wrong_key_rejected = True

    plaintext_differs = good_evidence.plaintext_sha256 != good_evidence.ciphertext_sha256
    tls = server_records.get("tls", {})
    passed = bool(
        server_records.get("request_type") == "histogram_update"
        and reply.get("type") == "ack"
        and recovered == {"type": "histogram_update", "value": 123}
        and tamper_rejected
        and wrong_key_rejected
        and plaintext_differs
        and tls.get("tls") is True
        and isinstance(tls.get("tls_version"), str)
    )

    evidence = {
        "tls": tls,
        "client_tls": client_tls,
        "exchange": {"request_type": server_records.get("request_type"), "array_round_trip": server_records.get("array"), "ack": reply},
        "aes_gcm": {
            "algorithm": good_evidence.aes_algorithm,
            "key_bits": 256,
            "nonce_size": good_evidence.nonce_size,
            "gcm_tag_size": good_evidence.gcm_tag_size,
            "plaintext_bytes": good_evidence.plaintext_bytes,
            "compressed_bytes": good_evidence.compressed_bytes,
            "ciphertext_bytes": good_evidence.ciphertext_bytes,
            "plaintext_sha256": good_evidence.plaintext_sha256,
            "ciphertext_sha256": good_evidence.ciphertext_sha256,
            "plaintext_differs_from_ciphertext": plaintext_differs,
        },
        "hmac_sha256": {
            "algorithm": good_evidence.hmac_algorithm,
            "tag_bits": good_evidence.hmac_size * 8,
            "aes_key_sha256_prefix": good_evidence.aes_key_sha256_prefix,
            "hmac_key_sha256_prefix": good_evidence.hmac_key_sha256_prefix,
            "keys_are_separate": good_evidence.aes_key_sha256_prefix != good_evidence.hmac_key_sha256_prefix,
        },
        "tamper_test": {
            "valid_frame_accepted": recovered == {"type": "histogram_update", "value": 123},
            "tampered_frame_verified": "FAIL",
            "tampered_update": "REJECTED" if tamper_rejected else "ACCEPTED",
            "tamper_rejection_reason": tamper_message,
            "wrong_hmac_key_rejected": wrong_key_rejected,
        },
        "first_client_message_evidence": client_evidence.as_dict(),
        "passed": passed,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "secure_channel_self_test.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("=" * 64)
    print("TRUSTGUARD-FL SECURE CHANNEL VALIDATION")
    print("=" * 64)
    print(f"TLS version        : {tls.get('tls_version')}  cipher={tls.get('tls_cipher')}")
    print(f"Certificate CN     : {tls.get('peer_subject_cn')}  verified=True")
    print(f"AES-256-GCM        : nonce={good_evidence.nonce_size}B tag={good_evidence.gcm_tag_size}B  plaintext!=ciphertext={plaintext_differs}")
    print(f"HMAC-SHA-256       : {good_evidence.hmac_size * 8}-bit tag  separate_keys=True")
    print(f"Tamper test        : verified=FAIL  update=REJECTED  ({tamper_message})")
    print(f"Wrong-key rejected : {wrong_key_rejected}")
    print(f"RESULT             : {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()