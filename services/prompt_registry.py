"""Versioned RAG prompt templates promoted only after a passing evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

DEFAULT_TEMPLATE = (
    "Answer the question using only the supplied context. If it is "
    "insufficient, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)
MAX_TEMPLATE_CHARS = 4000


def validate_template(template: str) -> None:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("prompt template must not be empty")
    if len(template) > MAX_TEMPLATE_CHARS:
        raise ValueError("prompt template exceeds the maximum length")
    if "{context}" not in template or "{query}" not in template:
        raise ValueError("prompt template must contain {context} and {query}")
    # Only the two placeholders are allowed; reject any other format field.
    try:
        template.format(context="", query="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("prompt template has unsupported placeholders") from exc


def render(template: str, context: str, query: str) -> str:
    # Single pass so placeholders inside retrieved text or the query stay literal.
    values = {"{context}": context, "{query}": query}
    return re.sub(r"\{context\}|\{query\}", lambda m: values[m.group(0)], template)


class PromptRegistry:
    """Active prompt plus candidates; promotion needs a passing eval and an approver."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self._path = Path(path) if path else None
        self._lock = threading.RLock()
        self._versions: list[dict[str, Any]] = [
            {"version": 1, "template": DEFAULT_TEMPLATE, "status": "active"}
        ]
        if self._path and self._path.exists():
            self._versions = json.loads(self._path.read_text(encoding="utf-8"))

    @classmethod
    def from_environment(cls) -> "PromptRegistry":
        return cls(os.getenv("AEGIS_PROMPT_REGISTRY_PATH") or None)

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._versions, indent=2), encoding="utf-8")
            tmp.replace(self._path)

    def active(self) -> dict[str, Any]:
        with self._lock:
            return dict(next(v for v in self._versions if v["status"] == "active"))

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(v) for v in self._versions]

    @staticmethod
    def _public(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            **{k: v for k, v in entry.items() if k != "template"},
            "template_sha256": hashlib.sha256(entry["template"].encode()).hexdigest(),
        }

    def propose(self, template: str, reason: str = "") -> dict[str, Any]:
        validate_template(template)
        with self._lock:
            entry = {
                "version": max(v["version"] for v in self._versions) + 1,
                "template": template,
                "status": "candidate",
                "reason": reason[:500],
            }
            self._versions.append(entry)
            self._save()
            return self._public(entry)

    def promote(
        self,
        version: int,
        approver: str,
        evaluator: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a candidate and activate it only if every gate passes."""
        if not approver:
            raise ValueError("approving actor is required")
        with self._lock:
            entry = next((v for v in self._versions if v["version"] == version), None)
            if entry is None:
                raise KeyError("prompt version not found")
            if entry["status"] != "candidate":
                raise ValueError("prompt version is not a candidate")
            report = evaluator(entry["template"])
            if report.get("status") != "passed":
                entry["status"] = "rejected"
                entry["evaluation"] = {
                    "status": report.get("status"),
                    "regressions": report.get("regressions", {}),
                }
                self._save()
                raise ValueError("prompt candidate failed evaluation")
            for item in self._versions:
                if item["status"] == "active":
                    item["status"] = "retired"
            entry["status"] = "active"
            entry["approved_by"] = approver
            entry["evaluation"] = {
                "status": "passed",
                "metrics": report.get("metrics", {}),
            }
            self._save()
            return self._public(entry)
