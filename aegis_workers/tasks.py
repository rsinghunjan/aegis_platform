from __future__ import annotations

from datetime import datetime
import uuid

from celery import shared_task
from sqlalchemy import select

from aegis_db.session import get_sessionmaker
from aegis_db.models import Job, Run


@shared_task(name="aegis.job.run")
def run_job(job_id: str) -> dict:
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.id == job_id))
        if not job:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}

        job.status = "running"
        run = Run(
            id=uuid.uuid4().hex,
            tenant_id=job.tenant_id,
            job_id=job.id,
            started_at=datetime.utcnow(),
            finished_at=None,
            metrics_json={},
            artifacts_uri=None,
        )
        session.add(run)
        session.commit()

        # TODO: route by job.kind; call rag indexing/eval/train/deploy etc.
        # For now, mark succeeded.
        job.status = "succeeded"
        run.finished_at = datetime.utcnow()
        run.metrics_json = {"note": "stub execution via celery; implement job.kind dispatch next"}
        session.commit()

        return {"ok": True, "job_id": job_id, "run_id": run.id}
