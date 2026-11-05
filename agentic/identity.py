"""OIDC identity and tenant authorization for the production agent API."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import jwt
from fastapi import HTTPException, Request

from agentic.worker import AgentPrincipal


class OIDCTenantIdentity:
    def __init__(
        self,
        jwks_uri: str,
        issuer: str,
        audience: str,
        tenant_claim: str = "tenant_id",
    ):
        self.issuer = issuer
        self.audience = audience
        self.tenant_claim = tenant_claim
        self.jwks = jwt.PyJWKClient(jwks_uri, timeout=5)

    @classmethod
    def from_environment(cls) -> OIDCTenantIdentity | None:
        jwks_uri = os.getenv("OIDC_JWKS_URI", "").strip()
        issuer = os.getenv("OIDC_ISSUER", "").strip()
        audience = os.getenv("OIDC_AUDIENCE", "").strip()
        if not any((jwks_uri, issuer, audience)):
            return None
        if not all((jwks_uri, issuer, audience)):
            raise RuntimeError(
                "OIDC_JWKS_URI, OIDC_ISSUER, and OIDC_AUDIENCE must be configured together"
            )
        return cls(
            jwks_uri,
            issuer,
            audience,
            tenant_claim=os.getenv("AEGIS_OIDC_TENANT_CLAIM", "tenant_id"),
        )

    def _claims(self, request: Request) -> dict[str, Any]:
        claims = getattr(request.state, "oidc_claims", None)
        if claims is not None:
            return claims

        authorization = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            raise HTTPException(status_code=401, detail="****** required")

        try:
            signing_key = self.jwks.get_signing_key_from_jwt(token.strip())
        except jwt.PyJWKClientError as exc:
            raise HTTPException(
                status_code=503, detail="OIDC signing keys are unavailable"
            ) from exc

        algorithms = [
            algorithm.strip()
            for algorithm in os.getenv("OIDC_ALLOWED_ALGORITHMS", "RS256").split(",")
            if algorithm.strip()
        ]
        try:
            claims = jwt.decode(
                token.strip(),
                signing_key.key,
                algorithms=algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
        request.state.oidc_claims = claims
        return claims

    async def resolve_principal(self, request: Request) -> AgentPrincipal:
        claims = await asyncio.to_thread(self._claims, request)
        principal_id = claims.get("sub")
        if not isinstance(principal_id, str) or not principal_id.strip():
            raise HTTPException(status_code=401, detail="Invalid token subject")

        scopes = claims.get("scopes", claims.get("scope", claims.get("scp", [])))
        if isinstance(scopes, str):
            scopes = scopes.split()
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            scopes = []

        role = claims.get("role", "agent")
        environment = claims.get(
            "environment", os.getenv("AEGIS_AGENT_ENVIRONMENT", "production")
        )
        return AgentPrincipal(
            principal_id=principal_id,
            role=role if isinstance(role, str) else "agent",
            environment=environment if isinstance(environment, str) else "production",
            scopes=scopes,
        )

    async def authorize_tenant(
        self,
        request: Request,
        tenant_id: str,
        _action: str,
        _actor: str | None = None,
    ) -> bool:
        claims = await asyncio.to_thread(self._claims, request)
        tenants = claims.get(self.tenant_claim)
        if isinstance(tenants, str):
            return tenants == tenant_id
        if isinstance(tenants, list):
            return tenant_id in tenants
        return False
