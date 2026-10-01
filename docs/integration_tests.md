
```markdown
Integration tests — runnable stack (Postgres, Redis, MinIO, API, Celery worker)

Overview
- docker/docker-compose.integration.yml spins up a minimal integration stack.
- tests/integration/test_integration_stack.py runs end-to-end checks against the running stack.
- scripts/wait_for_services.sh is a small helper used by tests to wait for the API to become healthy.

Run locally
1. Start the integration stack:
   docker compose -f docker/docker-compose.integration.yml up -d --build

2. Wait for API health:
   ./scripts/wait_for_services.sh http://localhost:8081/health 60 2

3. Run migrations & seed inside container (or let tests do it):
   docker compose -f docker/docker-compose.integration.yml exec -T api alembic upgrade head
   docker compose -f docker/docker-compose.integration.yml exec -T api python scripts/seed_db.py

4. Run tests:
   pytest tests/integration -q

CI
- Use the provided GitHub Actions (see .github/workflows/integration-tests.yml) to run integration tests in CI.
- Ensure the runner has Docker installed (ubuntu-latest does).
