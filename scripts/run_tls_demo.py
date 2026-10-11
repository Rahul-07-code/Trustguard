"""Run the local HTTPS/TLS transport proof."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_xgboost.security.tls_demo import run_tls_demo


def main() -> None:
    result = run_tls_demo()
    assert result["tls_connection"] == "ESTABLISHED"
    assert result["certificate_verified"] is True
    assert result["https_request"] is True
    print("TLS connection       : ESTABLISHED")
    print("Certificate verified  : PASS")
    print("HTTPS request         : PASS")
    print("Secure transport      : PASS")


if __name__ == "__main__":
    main()