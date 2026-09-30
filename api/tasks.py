"""Celery lifecycle adapter for registered, production job handlers."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from celery import current_app, shared_task
from sqlalchemy import update

from api.db import SessionLocal
from api.models import Job

logger = logging.getLogger("aegis_tasks")

JobHandler = Callable[[dict[str, Any]], Any]
_HANDLERS: dict[str, tuple[JobHandler, Optional[Callable[[dict[str, Any]], None]]]] = {}


def register_job_handler(
    kind: str,
    handler: JobHandler,
    validator: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Register an application-owned handler; there is deliberately no fake default."""
    if not kind or not callable(handler):
        raise ValueError("A job kind and callable handler are required")
    _HANDLERS[kind] = (handler, validator)


def _now() -> datetime:
    return datetime.utcnow()


def cancel_job(request_id: str) -> bool:
    """Mark a job cancelled and ask Celery to revoke its queued/running delivery."""
    session = SessionLocal()
    task_id = request_id
    try:
        job = session.query(Job).filter_by(request_id=request_id).one_or_none()
        if job is None or job.status in {"SUCCESS", "FAILED", "CANCELLED"}:
            return False
        task_id = (job.input_payload or {}).get("_job_meta", {}).get(
            "celery_task_id", request_id
        )
        job.status = "CANCELLED"
        job.updated_at = _now()
        session.commit()
    finally:
        session.close()
    current_app.control.revoke(task_id, terminate=True)
    return True


@shared_task(bind=True, name="aegis.process_job", max_retries=3)
def process_job(self, request_id: str) -> dict[str, Any]:
    """Validate, execute, and persist the lifecycle for a registered job kind."""
    session = SessionLocal()
    try:
        job = session.query(Job).filter_by(request_id=request_id).one_or_none()
        if job is None:
            return {"status": "FAILED", "error": "job not found", "request_id": request_id}
        if job.status in {"SUCCESS", "FAILED", "CANCELLED", "RUNNING"}:
            return {
                "status": job.status,
                "request_id": request_id,
                "output": job.output_payload,
            }

        payload = job.input_payload or {}
        kind = payload.get("kind", "default") if isinstance(payload, dict) else None
        handler_spec = _HANDLERS.get(kind)
        if handler_spec is None:
            job.status = "FAILED"
            job.output_payload = {"error": f"No handler registered for job kind: {kind}"}
            job.updated_at = _now()
            session.commit()
            return {"status": "FAILED", "request_id": request_id, "error": "handler unavailable"}

        handler, validator = handler_spec
        if not isinstance(payload, dict):
            raise ValueError("Job input must be a JSON object")
        handler_payload = {
            key: value for key, value in payload.items() if key not in {"kind", "_job_meta"}
        }
        if validator:
            validator(handler_payload)
        claimed = session.execute(
            update(Job)
            .where(
                Job.request_id == request_id,
                Job.status.in_(["PENDING", "RETRYING"]),
            )
            .values(status="RUNNING", updated_at=_now())
        )
        session.commit()
        if claimed.rowcount != 1:
            session.refresh(job)
            return {"status": job.status, "request_id": request_id}
        session.expire_all()
        job = session.query(Job).filter_by(request_id=request_id).one()
        result = handler(handler_payload)
        if hasattr(result, "__await__"):
            raise TypeError("Celery job handlers must be synchronous")
        session.refresh(job)
        if job.status == "CANCELLED":
            return {"status": "CANCELLED", "request_id": request_id}
        job.output_payload = result if isinstance(result, dict) else {"result": result}
        job.status = "SUCCESS"
        job.updated_at = _now()
        session.commit()
        return {"status": "SUCCESS", "request_id": request_id, "output": job.output_payload}
    except Exception as exc:
        session.rollback()
        job = session.query(Job).filter_by(request_id=request_id).one_or_none()
        if job:
            job.status = "RETRYING" if self.request.retries < self.max_retries else "FAILED"
            job.output_payload = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            job.updated_at = _now()
            session.commit()
        logger.exception("Job %s failed", request_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=min(60, 2**self.request.retries))
        raise
    finally:
        session.close()
