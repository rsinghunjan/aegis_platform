"""
Celery app factory for Aegis async tasks.

Usage (dev):
  export AEGIS_BROKER_URL=redis://redis:6379/0
  celery -A api.celery_app.app worker -Q aegis_tasks -l info

The app is intentionally lightweight so the worker process can import the
same application code (api.models, api.db, api.registry) and update DB rows.
"""
import os
import sys

from celery import Celery

# make sure repo root is on PYTHONPATH so tasks can import api.*
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

BROKER_URL = os.environ.get("AEGIS_BROKER_URL", os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"))
BACKEND_URL = os.environ.get("AEGIS_RESULT_BACKEND", os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL))

app = Celery(
    "aegis_tasks",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

# Basic recommended celery config; tune for prod in env or celery config module
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    timezone="UTC",
    enable_utc=True,
)

# Autodiscover tasks module in api.tasks
app.autodiscover_tasks(["api.tasks"])
