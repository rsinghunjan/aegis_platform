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
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
"""
Run a FastAPI app (import path) with mTLS enforced using ssl.SSLContext.

Example:
  python scripts/run_uvicorn_mtls.py aegis_multimodal_ai_system.inference.server:app \
      --cert certs/server.cert.pem --key certs/server.key.pem --ca certs/ca.cert.pem --port 9000

This script creates an ssl.SSLContext with CERT_REQUIRED and loads the CA bundle to verify client certificates.
Uvicorn supports being passed an SSLContext via programmatic API.
"""
import argparse
import importlib
import ssl
import sys
import logging

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(app_path: str, certfile: str, keyfile: str, cafile: str, host: str = "0.0.0.0", port: int = 9000, workers: int = 1):
    module_name, app_name = app_path.split(":", 1)
    mod = importlib.import_module(module_name)
    app = getattr(mod, app_name)

    context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    context.load_verify_locations(cafile=cafile)
    context.verify_mode = ssl.CERT_REQUIRED
    # optionally limit ciphers/protocols here

    # Uvicorn programmatic run
    uvicorn.run(app, host=host, port=port, ssl=context, workers=workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("app", help="app import path like package.module:app")
    parser.add_argument("--cert", required=True, help="server cert (PEM)")
    parser.add_argument("--key", required=True, help="server key (PEM)")
    parser.add_argument("--ca", required=True, help="CA cert(s) to verify client certs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    run(args.app, args.cert, args.key, args.ca, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
scripts/run_uvicorn_mtls.py
