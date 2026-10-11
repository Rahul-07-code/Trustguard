import socket
import ssl
import threading
import tempfile
import os
import ipaddress
import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def _build_san_extension(hosts: list[str]) -> x509.SubjectAlternativeName:
    san_list = []
    for h in hosts:
        try:
            ip = ipaddress.ip_address(h)
            san_list.append(x509.IPAddress(ip))
        except ValueError:
            san_list.append(x509.DNSName(h))
    return x509.SubjectAlternativeName(san_list)

def generate_self_signed_cert(cert_path: str, key_path: str, sans: list[str] = None):
    """Generates self-signed RSA-2048 certificate with IP and DNS SANs."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"TrustGuard FL"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])
    
    hosts = sans or ["localhost", "127.0.0.1"]
    san_ext = _build_san_extension(hosts)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        now - datetime.timedelta(minutes=5)
    ).not_valid_after(
        now + datetime.timedelta(days=365)
    ).add_extension(
        san_ext,
        critical=False,
    ).sign(key, hashes.SHA256())

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

def generate_ca_and_server_cert(
    ca_cert_path: str,
    ca_key_path: str,
    server_cert_path: str,
    server_key_path: str,
    hosts: list[str] = None,
):
    """
    Generates a local Certificate Authority (CA) and issues a server certificate
    signed by this CA with verified IP/DNS SANs for strict LAN verification.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # 1. Generate CA Key and Self-Signed CA Cert
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"TrustGuard-FL Root CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"TrustGuard Internal CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365 * 2))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )
    
    with open(ca_key_path, "wb") as f:
        f.write(ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    # 2. Generate Server Key and Certificate Signed by CA
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"TrustGuard FL Node"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"TrustGuard Server"),
    ])
    
    server_hosts = hosts or ["localhost", "127.0.0.1"]
    san_ext = _build_san_extension(server_hosts)
    
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(san_ext, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    with open(server_key_path, "wb") as f:
        f.write(server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(server_cert_path, "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))

def create_tls_server(host, port, cert_path, key_path):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.bind((host, port))
    sock.listen(5)
    ssock = context.wrap_socket(sock, server_side=True)
    return ssock

def create_tls_client(host, port, ca_cert_path, verify_hostname: bool = False):
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert_path)
    if verify_hostname:
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
    else:
        context.check_hostname = False
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    ssock = context.wrap_socket(sock, server_hostname=host)
    ssock.connect((host, port))
    return ssock

