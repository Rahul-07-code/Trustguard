import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AESGCMEncryption:
    def __init__(self, key: bytes = None):
        self.key = key or AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, data: bytes, associated_data: bytes = None) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, data, associated_data)
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes, associated_data: bytes = None) -> bytes:
        return self.aesgcm.decrypt(nonce, ciphertext, associated_data)
