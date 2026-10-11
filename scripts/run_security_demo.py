"""Mentor-facing AES-256-GCM and HMAC-SHA-256 demonstration."""

from __future__ import annotations

import copy
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.security.secure_envelope import KeyMaterial, encrypt_update, verify_and_decrypt
from fl_xgboost.security.tls_demo import run_tls_demo


def main() -> None:
    keys = KeyMaterial(aes_key=secrets.token_bytes(32), hmac_key=secrets.token_bytes(32))
    update = {
        "client_id": "client_01",
        "round": 1,
        "feature_count": 46,
        "class_count": 21,
        "histogram_summary": {"nonzero_bins": 128, "gradient_l1": 12.375, "hessian_sum": 2048.0},
    }
    envelope, evidence = encrypt_update(update, keys, nonce=b"0123456789ab")
    recovered, verification = verify_and_decrypt(envelope, keys)
    assert recovered == update

    tampered = copy.deepcopy(envelope.as_dict())
    tampered["ciphertext_b64"] = tampered["ciphertext_b64"][:-2] + ("AA" if not tampered["ciphertext_b64"].endswith("AA") else "BB")
    from fl_xgboost.security.secure_envelope import SecureEnvelope

    tampered_envelope = SecureEnvelope(**tampered)
    tamper_rejected = False
    try:
        verify_and_decrypt(tampered_envelope, keys)
    except ValueError as error:
        tamper_rejected = "HMAC verification failed" in str(error)
    assert tamper_rejected
    tls_result = run_tls_demo(payload=envelope.ciphertext_b64.encode("ascii"))
    assert tls_result["tls_connection"] == "ESTABLISHED"
    assert tls_result["certificate_verified"] is True
    assert tls_result["https_request"] is True

    print("=" * 52)
    print("TRUSTGUARD-FL SECURITY DEMONSTRATION")
    print("=" * 52)
    print("[PLAINTEXT FEDERATED UPDATE]")
    print(json.dumps(update, indent=2, sort_keys=True))
    print("\n[AES-256-GCM]")
    print(f"Plaintext size       : {evidence['plaintext_size']}")
    print(f"Ciphertext size      : {evidence['ciphertext_size']}")
    print(f"Nonce size           : {evidence['nonce_size']} bytes")
    print(f"Encryption           : {'SUCCESS' if evidence['plaintext_differs_from_ciphertext'] else 'FAIL'}")
    print(f"Decryption           : {'SUCCESS' if verification['decrypted'] else 'FAIL'}")
    print("Round-trip equality  : PASS")
    print("\n[HMAC-SHA-256]")
    print(f"Valid message        : {'PASS' if verification['hmac_verified'] else 'FAIL'}")
    print(f"Tampered message     : {'FAIL' if tamper_rejected else 'PASS'}")
    print(f"Tampered update      : {'REJECTED' if tamper_rejected else 'ACCEPTED'}")
    print("\n[TLS]")
    print(f"TLS connection       : {tls_result['tls_connection']}")
    print(f"Certificate verified  : {'PASS' if tls_result['certificate_verified'] else 'FAIL'}")
    print(f"HTTPS request         : {'PASS' if tls_result['https_request'] else 'FAIL'}")
    print("=" * 52)


if __name__ == "__main__":
    main()