from __future__ import annotations

import os
import subprocess
from celery import Celery

AUTO_MIGRATE = os.getenv("AEGIS_AUTO_MIGRATE", "false").lower() == "true"

if AUTO_MIGRATE:
    # best-effort; init-job is preferred in K8s
    subprocess.check_call(["python", "-m", "scripts.db_migrate"])

broker = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery("aegis_workers", broker=broker, backend=backend)
celery_app.autodiscover_tasks(["aegis_workers.tasks"])
