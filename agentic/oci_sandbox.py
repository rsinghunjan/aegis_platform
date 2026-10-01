"""gVisor-backed OCI execution for high-risk tool handlers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from typing import Any


logger = logging.getLogger(__name__)


class SandboxExecutionError(RuntimeError):
    """Raised when an isolated tool execution cannot complete safely."""


class GVisorSandboxExecutor:
    def __init__(
        self,
        docker_binary: str = "docker",
        runtime: str = "runsc",
        max_output_bytes: int = 65536,
    ):
        self.docker_binary = docker_binary
        self.runtime = runtime
        self.max_output_bytes = max_output_bytes

    async def execute(
        self, spec: Any, payload: dict[str, Any], _handler: Any = None
    ) -> Any:
        image = spec.sandbox_image
        entrypoint = spec.sandbox_entrypoint
        if not image or image.startswith("-") or not entrypoint:
            raise SandboxExecutionError("sandbox_image_and_entrypoint_required")
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", entrypoint
        ):
            raise SandboxExecutionError("invalid_sandbox_entrypoint")
        if (
            not math.isfinite(spec.sandbox_cpu_limit)
            or spec.sandbox_cpu_limit <= 0
            or spec.sandbox_memory_limit_mb <= 0
            or spec.sandbox_disk_limit_mb <= 0
        ):
            raise SandboxExecutionError("invalid_sandbox_resource_limit")

        secrets = {}
        secret_bytes = 0
        for target, source in spec.sandbox_secret_environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
                raise SandboxExecutionError("invalid_sandbox_environment_name")
            value = os.environ.get(source)
            if value is None:
                raise SandboxExecutionError("sandbox_secret_not_configured")
            secrets[target] = value
            secret_bytes += len(target.encode()) + len(value.encode())
        if secret_bytes > 65536:
            raise SandboxExecutionError("sandbox_secrets_too_large")

        request = json.dumps(
            {"payload": payload, "environment": secrets},
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        container_name = f"aegis-tool-{uuid.uuid4().hex}"
        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull=never",
            "--runtime",
            self.runtime,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            f"{spec.sandbox_memory_limit_mb}m",
            "--cpus",
            str(spec.sandbox_cpu_limit),
            "--shm-size",
            "16m",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={spec.sandbox_disk_limit_mb}m",
            "--user",
            "65532:65532",
            "--interactive",
            "--entrypoint",
            "python",
            image,
            "-m",
            "agentic.sandbox_runner",
            entrypoint,
        ]
        started = time.monotonic()
        logger.info(
            "tool_sandbox_starting",
            extra={
                "sandbox": {
                    "container": container_name,
                    "runtime": self.runtime,
                    "image": image,
                    "state": "starting",
                    "cpu_limit": spec.sandbox_cpu_limit,
                    "memory_limit_mb": spec.sandbox_memory_limit_mb,
                    "disk_limit_mb": spec.sandbox_disk_limit_mb,
                }
            },
        )
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            process.stdin.write(request)
            await process.stdin.drain()
            process.stdin.close()
            stdout = await self._read_output(
                process.stdout, min(spec.max_output_bytes, self.max_output_bytes)
            )
            await process.wait()
            if process.returncode:
                if process.returncode == 137:
                    raise SandboxExecutionError("sandbox_oom_or_killed")
                raise SandboxExecutionError(
                    f"sandbox_execution_failed:{process.returncode}"
                )
            try:
                response = json.loads(
                    stdout,
                    parse_constant=lambda _value: (_ for _ in ()).throw(
                        ValueError("non-finite JSON number")
                    ),
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise SandboxExecutionError("sandbox_output_not_json") from exc
            if (
                not isinstance(response, dict)
                or set(response) != {"result", "resource_usage"}
                or not isinstance(response["resource_usage"], dict)
            ):
                raise SandboxExecutionError("sandbox_output_invalid")
            usage = response["resource_usage"]
            if set(usage) != {
                "cpu_seconds",
                "memory_peak_bytes",
                "disk_used_bytes",
            } or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in usage.values()
            ):
                raise SandboxExecutionError("sandbox_resource_usage_invalid")
            logger.info(
                "tool_sandbox_terminated",
                extra={
                    "sandbox": {
                        "container": container_name,
                        "state": "succeeded",
                        "duration_seconds": time.monotonic() - started,
                        "cpu_limit": spec.sandbox_cpu_limit,
                        "memory_limit_mb": spec.sandbox_memory_limit_mb,
                        "disk_limit_mb": spec.sandbox_disk_limit_mb,
                        "resource_usage": usage,
                    }
                },
            )
            return response["result"]
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                self._kill(process)
                await process.wait()
            await self._remove_container(container_name)
            logger.warning(
                "tool_sandbox_terminated",
                extra={
                    "sandbox": {
                        "container": container_name,
                        "state": "cancelled",
                        "duration_seconds": time.monotonic() - started,
                    }
                },
            )
            raise
        except (OSError, SandboxExecutionError):
            if process is not None and process.returncode is None:
                self._kill(process)
                await process.wait()
            logger.exception(
                "tool_sandbox_terminated",
                extra={
                    "sandbox": {
                        "container": container_name,
                        "state": "failed",
                        "duration_seconds": time.monotonic() - started,
                    }
                },
            )
            raise
        finally:
            await self._remove_container(container_name)

    @staticmethod
    async def _read_output(stream: Any, max_bytes: int) -> bytes:
        output = bytearray()
        while True:
            chunk = await stream.read(min(8192, max_bytes + 1 - len(output)))
            if not chunk:
                return bytes(output)
            output.extend(chunk)
            if len(output) > max_bytes:
                raise SandboxExecutionError("sandbox_output_too_large")

    @staticmethod
    def _kill(process: Any) -> None:
        try:
            process.kill()
        except ProcessLookupError:
            pass

    async def _remove_container(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=5)
        except (OSError, asyncio.TimeoutError):
            logger.warning(
                "tool_sandbox_cleanup_failed",
                extra={"sandbox": {"container": container_name}},
            )
