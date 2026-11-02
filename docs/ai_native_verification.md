# AI-native verification and release gates

## Automated coverage

The `CI — tests` workflow runs the full repository suite with `pytest -q`. It also
starts PostgreSQL 15 and sets `AEGIS_TEST_POSTGRES_URL`, which runs the
`AICoreStore` retention, tenant-isolation, and deletion test against PostgreSQL.
Without that variable, the PostgreSQL-only test is skipped.

`tests/test_ai_tenant_isolation.py` verifies tenant scoping for persisted memory
and telemetry, knowledge retrieval and deletion, AI API responses, agent run
lookups, and operator insights.

## Threat model

The tenant ID is supplied by each request and is therefore untrusted until the
configured `tenant_authorizer` approves the requested action. The AI and agent
routes must continue to authorize each operation; the storage and retrieval
layers must continue to constrain reads and deletes to the requested tenant.
The tests above cover cross-tenant retrieval, deletion, insights, and run-ID
lookups. AICore's memory events use query hashes and operational metadata;
callers of the generic event store must keep payloads free of raw prompts.
Knowledge documents and inference audit data are tenant-keyed.

**Open security finding:** agent run goals are stored verbatim in the agent
database (`agentic/runtime.py`). There is no application-level retention or
deletion path for a run goal. Do not include secrets or sensitive raw content in
goals. Before production use, establish and enforce a retention/deletion policy
and verify database and backup access controls/encryption. The automated tests
do not establish those operational controls.

## Load verification

The AI-specific k6 smoke profile exercises `/ai/answer` with a bounded 5-to-25
virtual-user ramp and checks for less than 1% failed requests and a p95 below
2 seconds. Run it against a representative non-production deployment with the
test/echo inference provider and a low-cost test tenant:

```sh
k6 run \
  -e API_BASE=https://staging.example.invalid \
  -e TENANT_ID=test-tenant \
  -e AEGIS_AUTH_TOKEN="$AEGIS_AUTH_TOKEN" \
  tests/load/k6/ai_native_load_test.js
```

The token is supplied through the environment and must not be committed or
placed in shell history. This profile is a smoke test, not a production capacity
benchmark. Run sustained, production-shaped load and verify provider spend,
tenant quotas, database saturation, and deployment-specific SLOs before making
production capacity claims.
