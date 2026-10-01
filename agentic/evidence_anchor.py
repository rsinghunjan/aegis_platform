"""Adapters for writing and checking run evidence in an external immutable log."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Protocol
from urllib.error import URLError
from urllib.parse import quote, urlencode, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_without_redirects(request: Request, timeout: float):
    return build_opener(_NoRedirectHandler).open(request, timeout=timeout)


class EvidenceAnchorBackend(Protocol):
    """External append-only log operations required by evidence verification."""

    name: str

    def anchor(
        self, run_id: str, tenant_id: str, head_sha256: str
    ) -> dict[str, Any]: ...

    def list_anchors(self, run_id: str, tenant_id: str) -> list[dict[str, Any]]: ...

    def verify(self, proof: dict[str, Any]) -> bool: ...


class HttpTransparencyLogBackend:
    """Adapter for an HTTPS service implementing the documented anchor API."""

    name = "http-transparency-log"

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 5.0):
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Evidence anchor URL must be an HTTPS origin")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Evidence anchor timeout must be positive")
        self.base_url = urlunparse(parsed._replace(path=parsed.path.rstrip("/")))
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> HttpTransparencyLogBackend | None:
        base_url = os.getenv("AEGIS_EVIDENCE_ANCHOR_URL")
        if not base_url:
            return None
        return cls(
            base_url,
            token=os.getenv("AEGIS_EVIDENCE_ANCHOR_TOKEN"),
            timeout=float(os.getenv("AEGIS_EVIDENCE_ANCHOR_TIMEOUT", "5")),
        )

    def anchor(
        self, run_id: str, tenant_id: str, head_sha256: str
    ) -> dict[str, Any]:
        proof = self._request(
            "POST",
            "/v1/anchors",
            {"run_id": run_id, "tenant_id": tenant_id, "head_sha256": head_sha256},
        )
        if not self._matches(proof, run_id, tenant_id, head_sha256):
            raise ValueError("Evidence anchor service returned a mismatched receipt")
        if not isinstance(proof.get("anchor_id"), str) or not proof["anchor_id"]:
            raise ValueError("Evidence anchor service returned an invalid receipt")
        return proof

    def list_anchors(self, run_id: str, tenant_id: str) -> list[dict[str, Any]]:
        query = urlencode({"run_id": run_id, "tenant_id": tenant_id})
        result = self._request("GET", f"/v1/anchors?{query}")
        anchors = result.get("anchors") if isinstance(result, dict) else None
        if not isinstance(anchors, list) or any(
            not isinstance(item, dict) for item in anchors
        ):
            raise ValueError("Evidence anchor service returned an invalid record list")
        return anchors

    def verify(self, proof: dict[str, Any]) -> bool:
        anchor_id = proof.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id:
            return False
        stored = self._request("GET", f"/v1/anchors/{quote(anchor_id, safe='')}")
        return (
            isinstance(stored, dict)
            and stored.get("anchor_id") == anchor_id
            and self._matches(
                stored,
                str(proof.get("run_id", "")),
                str(proof.get("tenant_id", "")),
                str(proof.get("head_sha256", "")),
            )
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if payload else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with _open_without_redirects(request, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
        except (OSError, URLError) as exc:
            raise RuntimeError("Evidence anchor service request failed") from exc
        if len(raw) > 1_048_576:
            raise ValueError("Evidence anchor service response is too large")
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Evidence anchor service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("Evidence anchor service returned an invalid response")
        return result

    @staticmethod
    def _matches(
        proof: dict[str, Any], run_id: str, tenant_id: str, head_sha256: str
    ) -> bool:
        return (
            proof.get("run_id") == run_id
            and proof.get("tenant_id") == tenant_id
            and proof.get("head_sha256") == head_sha256
        )
