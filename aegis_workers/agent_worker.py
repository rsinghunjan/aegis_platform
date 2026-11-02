"""Celery task registration for durable agent-run execution."""
from agentic.deployment import create_configured_agent_runtime
from agentic.worker import AgentWorker, register_celery_agent_worker
from aegis_workers.worker import celery_app

runtime = create_configured_agent_runtime()
runtime.store.initialize()
register_celery_agent_worker(celery_app, AgentWorker(runtime))

