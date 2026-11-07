import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_production_compose_renders_api_and_worker(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is not installed")
    version = subprocess.run(
        ["docker", "compose", "version"], capture_output=True, check=False
    )
    if version.returncode:
        pytest.skip("Docker Compose is not installed")

    shutil.copy(ROOT / "docker-compose.production.yml", tmp_path)
    shutil.copy(ROOT / "docker" / "production.env.example", tmp_path / ".env.production")
    env = os.environ.copy()
    env.pop("AEGIS_IMAGE", None)
    result = subprocess.run(
        [
            "docker", "compose", "--env-file", ".env.production",
            "-f", "docker-compose.production.yml", "config", "--format", "json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    services = json.loads(result.stdout)["services"]
    api = services["api"]
    worker = services["agent-worker"]

    assert api["image"] == worker["image"]
    for service in (api, worker):
        assert service["environment"]["AEGIS_ENV"] == "production"
        assert service["environment"]["AEGIS_ALLOW_DEV_PROVIDERS"] == "false"
        assert service["environment"]["AEGIS_AGENT_QUEUE"] == "aegis-agent"
        assert service["environment"]["AEGIS_AI_DATABASE_URL"].startswith("postgresql")
        assert service["environment"]["CELERY_BROKER_URL"].startswith("rediss://")
    assert api["ports"][0]["host_ip"] == "127.0.0.1"
    assert "--queues=aegis-agent" in worker["command"]
    assert worker["healthcheck"]["disable"] is True
