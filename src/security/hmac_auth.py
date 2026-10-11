import hmac
import hashlib

class HMACAuth:
    def __init__(self, key: bytes = None):
        import os
        self.key = key or os.urandom(32)

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self.key, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)
