"""Secret management abstraction (Vault, AWS Secrets Manager, GCP Secret Manager, env)."""
from __future__ import annotations

import abc
import os
from typing import Dict, Optional


class SecretManager(abc.ABC):
    name: str = "secret-manager"

    @abc.abstractmethod
    def get_secret(self, key: str) -> str:
        raise NotImplementedError

    def get_secret_or_default(self, key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            return self.get_secret(key)
        except KeyError:
            return default

    def is_available(self) -> bool:
        return True


class EnvSecretManager(SecretManager):
    """Reads secrets from environment variables, with an optional prefix.

    This is the safe default/fallback used when no external secret store
    (Vault, AWS/GCP Secrets Manager) is configured.
    """

    name = "env"

    def __init__(self, prefix: str = "AEGIS_SECRET_"):
        self.prefix = prefix

    def get_secret(self, key: str) -> str:
        env_key = f"{self.prefix}{key.upper()}"
        value = os.environ.get(env_key)
        if value is None:
            raise KeyError(key)
        return value


class VaultSecretManager(SecretManager):
    """HashiCorp Vault-backed secret manager (KV v2 engine)."""

    name = "vault"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, mount_point: str = "secret"):
        self.url = url or os.environ.get("VAULT_ADDR")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.mount_point = mount_point

    def is_available(self) -> bool:
        if not (self.url and self.token):
            return False
        try:
            import hvac  # noqa: F401
        except ImportError:
            return False
        return True

    def get_secret(self, key: str) -> str:
        if not self.is_available():
            raise RuntimeError("vault secret manager unavailable (missing hvac/config)")
        import hvac

        client = hvac.Client(url=self.url, token=self.token)
        response = client.secrets.kv.v2.read_secret_version(path=key, mount_point=self.mount_point)
        data: Dict[str, str] = response["data"]["data"]
        if "value" in data:
            return data["value"]
        return next(iter(data.values()))


class AWSSecretsManager(SecretManager):
    name = "aws-secrets-manager"

    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name or os.environ.get("AWS_REGION")

    def is_available(self) -> bool:
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True

    def get_secret(self, key: str) -> str:
        if not self.is_available():
            raise RuntimeError("aws secrets manager unavailable (missing boto3)")
        import boto3

        client = boto3.client("secretsmanager", region_name=self.region_name)
        response = client.get_secret_value(SecretId=key)
        return response.get("SecretString", "")


class GCPSecretManager(SecretManager):
    name = "gcp-secret-manager"

    def __init__(self, project: str):
        self.project = project

    def is_available(self) -> bool:
        try:
            from google.cloud import secretmanager  # noqa: F401
        except ImportError:
            return False
        return True

    def get_secret(self, key: str) -> str:
        if not self.is_available():
            raise RuntimeError("gcp secret manager unavailable (missing google-cloud-secret-manager)")
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{self.project}/secrets/{key}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")
