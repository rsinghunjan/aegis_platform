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
#!/usr/bin/env python
"""
CLI wrapper to import a model from Hugging Face Hub into Aegis.

Example:
  python scripts/hf_import.py --repo-id google/flan-t5-small --model-name flan-t5-small --version hf-20251203 --upload-s3 --sign-key aegis-model-sign --register

Environment:
  - To upload to S3: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_S3_BUCKET (or pass --s3-bucket)
  - To sign: VAULT_ADDR + VAULT_TOKEN (or use OIDC in CI) must be set for Vault Transit usage
"""
import argparse
import os
import logging
from api.hf_importer import import_from_hf
from api import registry as _registry  # adjust import if your registry is in different module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.hf_import_cli")

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True, help="Hugging Face repo id (e.g., 'google/flan-t5-small')")
parser.add_argument("--model-name", required=True, help="Name to register in Aegis registry")
parser.add_argument("--version", default=None, help="Version tag to use in registry (default timestamp)")
parser.add_argument("--sign-key", default=None, help="Vault Transit key name to sign artifact")
parser.add_argument("--upload-s3", action="store_true", help="Upload resulting artifact to S3")
parser.add_argument("--s3-bucket", default=None, help="S3 bucket to upload artifact")
parser.add_argument("--s3-prefix", default=None, help="S3 prefix/folder for artifacts")
parser.add_argument("--register", action="store_true", help="Register artifact in local registry")
parser.add_argument("--convert-torchscript", action="store_true", help="Attempt to convert model to TorchScript (best-effort)")
args = parser.parse_args()

def main():
    res = import_from_hf(
        repo_id=args.repo_id,
        model_name=args.model_name,
        version=args.version,
        sign_key=args.sign_key,
        upload_s3=args.upload_s3,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        registry=_registry if args.register else None,
        register=args.register,
        convert_to_torchscript=args.convert_torchscript,
    )
    logger.info("Import result: %s", res)

if __name__ == "__main__":
    main()
scripts/hf_import.py
