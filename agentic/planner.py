"""Provider-neutral planners that return only constrained JSON tool plans."""
from __future__ import annotations

import json
import os
import uuid
from urllib.parse import urlparse
import urllib.request
from typing import Any, Optional

from agentic.runtime import (
    DeterministicJSONPlanner,
    Plan,
    PlanStep,
    Planner,
)


class OpenAICompatiblePlanner(Planner):
    """Optional chat-completions adapter with deterministic, fail-closed fallback."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_steps: int = 16,
        fallback: Optional[Planner] = None,
    ):
        if not endpoint or not model or timeout <= 0 or max_steps <= 0:
            raise ValueError("planner endpoint, model, timeout, and step bound are required")
        parsed_endpoint = urlparse(endpoint)
        local_http = parsed_endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed_endpoint.scheme != "https" and not (
            parsed_endpoint.scheme == "http" and local_http
        ):
            raise ValueError("planner endpoint must use HTTPS or loopback HTTP")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_steps = max_steps
        self.fallback = fallback or DeterministicJSONPlanner()
        self.provider_name = "openai-compatible-chat-completions"
        # "llm" when the last plan came from the model, "fallback" otherwise.
        self.last_source = "llm"

    @classmethod
    def from_environment(cls) -> Planner:
        """LLM planning is the default; set AEGIS_LLM_PLANNER_ENABLED=false to opt out.

        Outside dev/test (AEGIS_ALLOW_DEV_PROVIDERS), a missing API key is a
        startup error rather than a silent downgrade to deterministic planning.
        """
        flag = os.getenv("AEGIS_LLM_PLANNER_ENABLED", "").strip().lower()
        if flag in {"0", "false", "no"}:
            return DeterministicJSONPlanner()
        dev = os.getenv("AEGIS_ALLOW_DEV_PROVIDERS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        endpoint = (
            os.getenv("AEGIS_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        key = os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("AEGIS_LLM_MODEL", "gpt-4o-mini")
        if not key:
            if dev:
                return DeterministicJSONPlanner()
            raise RuntimeError(
                "LLM planner requires AEGIS_LLM_API_KEY; set "
                "AEGIS_LLM_PLANNER_ENABLED=false to use deterministic planning"
            )
        try:
            return cls(
                endpoint=endpoint,
                model=model,
                api_key=key,
                timeout=float(os.getenv("AEGIS_LLM_TIMEOUT_SECONDS", "10")),
                max_steps=int(os.getenv("AEGIS_LLM_MAX_STEPS", "16")),
            )
        except (TypeError, ValueError) as exc:
            if dev:
                return DeterministicJSONPlanner()
            raise RuntimeError(f"invalid LLM planner configuration: {exc}") from exc

    def plan(
        self, goal: str, registered_tools: set[str], recovery: bool = False
    ) -> Plan:
        if not self.api_key:
            self.last_source = "fallback"
            return self.fallback.plan(goal, registered_tools, recovery)
        try:
            candidate = self._request_plan(goal, registered_tools)
            plan = self._validate(candidate, registered_tools)
            self.last_source = "llm"
            return plan
        except Exception:
            self.last_source = "fallback"
            return self.fallback.plan(goal, registered_tools, recovery)

    def plan_with_inference(
        self,
        goal: str,
        registered_tools: set[str],
        generate: Any,
        recovery: bool = False,
    ) -> Plan:
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": sorted(registered_tools)},
                            "input": {"type": "object"},
                            "acceptance": {"type": "object"},
                        },
                        "required": ["tool", "input"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
        prompt = (
            "Return only JSON matching this schema. Never provide code, commands, "
            f"or unregistered tools: {json.dumps(schema)}\n"
            f"Recovery planning: {str(recovery).lower()}\n"
            f"Goal: {goal}"
        )
        try:
            content = generate(prompt)
            if not isinstance(content, str) or len(content) > 256_000:
                raise ValueError("invalid planner response")
            plan = self._validate(
                json.loads(
                    content,
                    parse_constant=lambda _value: (_ for _ in ()).throw(
                        ValueError("non-finite JSON number")
                    ),
                ),
                registered_tools,
            )
            self.last_source = "llm"
            return plan
        except Exception:
            self.last_source = "fallback"
            return self.fallback.plan(goal, registered_tools, recovery)

    def _request_plan(self, goal: str, registered_tools: set[str]) -> Any:
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": sorted(registered_tools)},
                            "input": {"type": "object"},
                            "acceptance": {"type": "object"},
                        },
                        "required": ["tool", "input"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON matching this schema. Never provide code, "
                            f"commands, or unregistered tools: {json.dumps(schema)}"
                        ),
                    },
                    {"role": "user", "content": goal},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request.add_unredirected_header(
            "Authorization", "Bearer " + self.api_key
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            envelope = json.loads(response.read(1_000_000))
        content = envelope["choices"][0]["message"]["content"]
        if not isinstance(content, str) or len(content) > 256_000:
            raise ValueError("invalid planner response")
        return json.loads(
            content,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )

    def _validate(self, value: Any, registered_tools: set[str]) -> Plan:
        if not isinstance(value, dict) or set(value) != {"steps"}:
            raise ValueError("unsupported planner fields")
        steps = value["steps"]
        if not isinstance(steps, list) or not steps or len(steps) > self.max_steps:
            raise ValueError("invalid plan size")
        plan_steps = []
        for ordinal, spec in enumerate(steps):
            if not isinstance(spec, dict) or set(spec) - {"tool", "input", "acceptance"}:
                raise ValueError("unsupported step fields")
            tool = spec.get("tool")
            tool_input = spec.get("input")
            acceptance = spec.get("acceptance", {})
            if tool not in registered_tools or not isinstance(tool_input, dict):
                raise ValueError("unknown tool or invalid input")
            if not isinstance(acceptance, dict):
                raise ValueError("invalid acceptance criteria")
            if set(acceptance) - {"required_keys", "equals"}:
                raise ValueError("unsupported acceptance criteria")
            required = acceptance.get("required_keys", [])
            if not isinstance(required, list) or any(
                not isinstance(key, str) for key in required
            ):
                raise ValueError("invalid required_keys")
            if not isinstance(acceptance.get("equals", {}), dict):
                raise ValueError("invalid equals criteria")
            encoded = json.dumps([tool_input, acceptance], allow_nan=False)
            if any(marker in encoded.lower() for marker in ("```", "subprocess", "shell_exec")):
                raise ValueError("executable content is not accepted")
            plan_steps.append(
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    ordinal=ordinal,
                    tool_name=tool,
                    input=tool_input,
                    acceptance_criteria=acceptance,
                )
            )
        return Plan(plan_id=str(uuid.uuid4()), steps=plan_steps)
