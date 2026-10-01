  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
#!/usr/bin/env bash
# Generate a local CA, server cert and client cert for mTLS testing.
# NOT for production; for production use a real PKI or a managed CA (e.g., ACME, Vault PKI).
set -euo pipefail

OUTDIR="${1:-certs}"
mkdir -p "${OUTDIR}"

echo "Generating CA key and certificate..."
openssl genrsa -out "${OUTDIR}/ca.key.pem" 4096
openssl req -x509 -new -nodes -key "${OUTDIR}/ca.key.pem" -sha256 -days 3650 -subj "/CN=Aegis-Local-CA" -out "${OUTDIR}/ca.cert.pem"

echo "Generating server key and CSR..."
openssl genrsa -out "${OUTDIR}/server.key.pem" 2048
openssl req -new -key "${OUTDIR}/server.key.pem" -subj "/CN=localhost" -out "${OUTDIR}/server.csr.pem"

echo "Signing server cert with CA..."
openssl x509 -req -in "${OUTDIR}/server.csr.pem" -CA "${OUTDIR}/ca.cert.pem" -CAkey "${OUTDIR}/ca.key.pem" -CAcreateserial -out "${OUTDIR}/server.cert.pem" -days 365 -sha256

echo "Generating client key and CSR..."
openssl genrsa -out "${OUTDIR}/client.key.pem" 2048
openssl req -new -key "${OUTDIR}/client.key.pem" -subj "/CN=aegis-client" -out "${OUTDIR}/client.csr.pem"

echo "Signing client cert with CA..."
openssl x509 -req -in "${OUTDIR}/client.csr.pem" -CA "${OUTDIR}/ca.cert.pem" -CAkey "${OUTDIR}/ca.key.pem" -CAcreateserial -out "${OUTDIR}/client.cert.pem" -days 365 -sha256

echo "Done. Files in ${OUTDIR}:"
ls -l "${OUTDIR}"
echo
echo "To test an mTLS server using these certs, run the Python uvicorn wrapper script scripts/run_uvicorn_mtls.py"
