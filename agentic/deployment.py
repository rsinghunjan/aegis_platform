"""Shared runtime construction for the production API and Celery worker."""
from __future__ import annotations

import importlib
import os

from agentic.runtime import AgentRuntime


def create_configured_agent_runtime() -> AgentRuntime:
    runtime = AgentRuntime()
    module_name = os.getenv("AEGIS_AGENT_TOOL_MODULE", "").strip()
    if not module_name:
        return runtime

    module = importlib.import_module(module_name)
    configure_runtime = getattr(module, "configure_agent_runtime", None)
    if not callable(configure_runtime):
        raise RuntimeError(
            f"{module_name} must expose configure_agent_runtime(runtime)"
        )
    configure_runtime(runtime)
    return runtime
