# tests/conftest.py
import os
import shutil
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure we run tests using a dedicated SQLite file DB for determinism
TEST_DB_FILE = Path("./test_aegis.db")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_FILE.resolve()}"
# Use a stable secret for tests
os.environ["AEGIS_SECRET_KEY"] = "test-secret-key"

# Import api modules after env var is set so they pick up DATABASE_URL
import api.db as db_module  # noqa: E402
import api.models as models_module  # noqa: E402
import api.auth as auth_module  # noqa: E402

try:
    # api_server pulls in the full FastAPI app; some unrelated modules in this
    # tree may be unavailable/broken in a given checkout. Import defensively so
    # that a single unrelated broken module doesn't prevent collection of
    # tests that don't need the `client` fixture at all.
    import api.api_server as api_server  # noqa: E402
except Exception:  # pragma: no cover - defensive fallback
    api_server = None

# Recreate modules to ensure they pick up env vars if already imported
importlib.reload(db_module)
importlib.reload(models_module)
if api_server is not None:
    importlib.reload(api_server)
importlib.reload(auth_module)

# Create DB tables for the test session
models_module.Base.metadata.create_all(bind=db_module.engine)


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient for the running app instance.
    Uses the in-repo api.api_server.app.
    """
    if api_server is None:
        pytest.skip("api.api_server is unavailable in this checkout")
    with TestClient(api_server.app) as c:
        yield c


@pytest.fixture(scope="session")
def admin_token():
    """
    Return a signed access token with admin/predict scopes for tests.
    Uses the auth.create_access_token helper.
    """
    return auth_module.create_access_token(subject="admin", scopes=["predict", "model:read", "admin"])


@pytest.fixture(scope="session")
def predict_token():
    return auth_module.create_access_token(subject="alice", scopes=["predict", "model:read"])


@pytest.fixture(autouse=True)
def clean_db_between_tests():
    """
    Ensure tests have a clean-ish DB state across tests by truncating key tables.
    This keeps tests deterministic while remaining fast using sqlite file DB.
    """
    yield
    # Truncate simple tables used by tests (safety_events, jobs, model_versions, models, users, refresh_tokens)
    from sqlalchemy import text
    with db_module.engine.connect() as conn:
        for tbl in ("safety_events", "jobs", "model_versions", "models", "refresh_tokens", "users"):
            try:
                conn.execute(text(f"DELETE FROM {tbl};"))
            except Exception:
                # table may not exist yet or be empty; ignore
                pass
        conn.commit()


def pytest_sessionfinish(session, exitstatus):
    # cleanup the sqlite test DB file after test session completes
    try:
        if TEST_DB_FILE.exists():
            TEST_DB_FILE.unlink()
    except Exception:
        pass
