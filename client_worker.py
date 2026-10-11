"""TrustGuard-FL Remote Client Worker CLI Entry Point.

Example:
    python client_worker.py --server-url https://192.168.1.15:8443 --client-id client-02 --ca-cert cert.pem
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fl.client import main

if __name__ == "__main__":
    main()
