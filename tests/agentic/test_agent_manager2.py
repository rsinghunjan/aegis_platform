import asyncio

import pytest

pytest.importorskip("aegis_multimodal_ai_system.agentic.agent_manager")

from aegis_multimodal_ai_system.agentic.agent_manager import (
    AgentManager,
    example_echo_tool,
    async_wrapper_example,
)


def test_sync_tool_invocation():
    am = AgentManager()
    am.register_tool("echo", example_echo_tool)
    action = {
        "tool": "echo",
        "payload": {"text": "hello"},
    }
    res = asyncio.run(am.dispatch_action(action))
    assert res["status"] == "ok"
    assert res["result"]["echo_text"] == "hello"


def test_async_tool_invocation():
    am = AgentManager()
    am.register_tool("async_echo", async_wrapper_example)
    action = {
        "tool": "async_echo",
        "payload": {"text": "async"},
    }
    res = asyncio.run(am.dispatch_action(action))
    assert res["status"] == "ok"
    assert res["result"]["echo_text"] == "async"


def test_missing_tool():
    am = AgentManager()
    res = asyncio.run(am.dispatch_action({"tool": "nope"}))
    assert res["status"] == "error"


def test_timeout_behavior():
    async def slow_tool(payload):
        await asyncio.sleep(1)
        return {"ok": True}

    am = AgentManager()
    am.register_tool("slow", slow_tool)
    res = asyncio.run(
        am.dispatch_action({"tool": "slow", "timeout_seconds": 0.01})
    )
    assert res["status"] == "error"
    assert "timeout" in res.get("error", "")
