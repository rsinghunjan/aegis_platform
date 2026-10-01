import asyncio
import json

import pytest

from agentic.oci_sandbox import GVisorSandboxExecutor, SandboxExecutionError
from agentic.runtime import ToolSpec


class FakeStdin:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        pass


class FakeStream:
    def __init__(self, data):
        self.data = data

    async def read(self, limit):
        chunk, self.data = self.data[:limit], self.data[limit:]
        return chunk


class FakeProcess:
    def __init__(
        self,
        output=b'{"result":{"ok":true},"resource_usage":{"cpu_seconds":0.1,'
        b'"memory_peak_bytes":1024,"disk_used_bytes":64}}',
    ):
        self.stdin = FakeStdin()
        self.stdout = FakeStream(output)
        self.returncode = None
        self.killed = False

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_gvisor_oci_invocation_is_isolated_and_receives_only_configured_secrets(
    monkeypatch,
):
    monkeypatch.setenv("TOOL_API_TOKEN", "test-secret")
    commands = []
    process = FakeProcess()

    async def create_process(*args, **kwargs):
        commands.append((args, kwargs))
        if args[1] == "run":
            return process
        return FakeProcess()

    monkeypatch.setattr(
        "agentic.oci_sandbox.asyncio.create_subprocess_exec", create_process
    )
    spec = ToolSpec(
        name="publish",
        risk_level="high",
        sandbox_image="registry.example/tools@sha256:abc",
        sandbox_entrypoint="tools.publish:run",
        sandbox_cpu_limit=0.5,
        sandbox_memory_limit_mb=128,
        sandbox_disk_limit_mb=32,
        sandbox_secret_environment={"API_TOKEN": "TOOL_API_TOKEN"},
    )

    result = asyncio.run(GVisorSandboxExecutor().execute(spec, {"id": "one"}))

    assert result == {"ok": True}
    command = commands[0][0]
    assert command[command.index("--runtime") + 1] == "runsc"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--memory") + 1] == "128m"
    assert command[command.index("--cpus") + 1] == "0.5"
    assert command[command.index("--tmpfs") + 1].endswith("size=32m")
    request = json.loads(process.stdin.data)
    assert request == {
        "payload": {"id": "one"},
        "environment": {"API_TOKEN": "test-secret"},
    }
    assert "test-secret" not in " ".join(command)


def test_gvisor_executor_fails_closed_without_image_or_entrypoint(monkeypatch):
    async def unexpected_process(*_args, **_kwargs):
        raise AssertionError("must not start a container")

    monkeypatch.setattr(
        "agentic.oci_sandbox.asyncio.create_subprocess_exec", unexpected_process
    )
    with pytest.raises(SandboxExecutionError, match="sandbox_image_and_entrypoint"):
        asyncio.run(
            GVisorSandboxExecutor().execute(
                ToolSpec(name="high_risk", risk_level="high"), {}
            )
        )


def test_gvisor_executor_bounds_container_output(monkeypatch):
    process = FakeProcess(output=b'{"output_is_too_large":true}')

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        "agentic.oci_sandbox.asyncio.create_subprocess_exec", create_process
    )
    spec = ToolSpec(
        name="publish",
        risk_level="high",
        sandbox_image="tools:latest",
        sandbox_entrypoint="tools.publish:run",
        max_output_bytes=8,
    )

    with pytest.raises(SandboxExecutionError, match="sandbox_output_too_large"):
        asyncio.run(GVisorSandboxExecutor().execute(spec, {}))
    assert process.killed
