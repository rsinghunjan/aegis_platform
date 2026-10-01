"""Execution-boundary policy for trusted, registered tool adapters."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any


class SandboxProfile(str, Enum):
    PURE = "pure"
    NETWORK = "network"
    STORAGE = "storage"
    RESTRICTED_SUBPROCESS = "restricted-subprocess"


class SandboxViolation(ValueError):
    pass


class SandboxBoundary:
    """Apply bounded JSON and profile checks before/after trusted tool handlers."""

    def __init__(
        self,
        max_payload_bytes: int = 65536,
        max_output_bytes: int = 65536,
        allowed_environments: tuple[str, ...] = ("local", "test"),
        network_policy: Any = None,
    ):
        self.max_payload_bytes = max_payload_bytes
        self.max_output_bytes = max_output_bytes
        self.allowed_environments = allowed_environments
        self.network_policy = network_policy

    def check_input(self, payload: Any, profile: str, environment: str) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode()
        if len(encoded) > self.max_payload_bytes:
            raise SandboxViolation("sandbox_input_too_large")
        if profile == SandboxProfile.RESTRICTED_SUBPROCESS.value:
            raise SandboxViolation("subprocess_profile_requires_external_adapter")
        if profile not in {item.value for item in SandboxProfile}:
            raise SandboxViolation("unknown_sandbox_profile")
        if profile == SandboxProfile.NETWORK.value:
            if environment not in self.allowed_environments:
                raise SandboxViolation("sandbox_environment_not_allowed")
            if self.network_policy is None:
                raise SandboxViolation("network_policy_not_configured")
            try:
                decision = self.network_policy(payload, environment)
            except Exception as exc:
                raise SandboxViolation("network_policy_error") from exc
            if decision is not True:
                raise SandboxViolation("network_policy_denied")
        if profile == SandboxProfile.PURE.value and _contains_executable_request(payload):
            raise SandboxViolation("executable_payload_rejected")

    def check_output(self, output: Any) -> None:
        try:
            encoded = json.dumps(output, separators=(",", ":"), default=str).encode()
        except (TypeError, ValueError) as exc:
            raise SandboxViolation("sandbox_output_not_json") from exc
        if len(encoded) > self.max_output_bytes:
            raise SandboxViolation("sandbox_output_too_large")


def _contains_executable_request(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"command", "cmd", "script", "code", "shell"}:
                return True
            if _contains_executable_request(child):
                return True
    elif isinstance(value, list):
        return any(_contains_executable_request(child) for child in value)
    return False
