"""
Asynchronous tasks for Aegis.

- process_job(job_id): long-running job harness that:
    * sets job.status -> RUNNING
    * performs simulated preprocessing/inference (replace with real logic)
    * writes output_payload and sets status -> SUCCESS or FAILED

Important:
- Tasks use SQLAlchemy sessions from api.db.SessionLocal to update DB.
- In production, keep task workers isolated and sized appropriately for models (GPU scheduling etc).
"""
import os
import time
import json
import logging
from datetime import datetime

from celery import shared_task, current_task

# Ensure imports find the api package (celery worker process should run from repo root)
from api.db import SessionLocal
from api.models import Job

logger = logging.getLogger("aegis_tasks")
logging.basicConfig(level=logging.INFO)

# Per-unit simulated work duration (seconds). Kept small by default and
# configurable via env var so this demo/placeholder loop doesn't needlessly
# tie up a Celery worker thread/process for multiple seconds per job; replace
# time.sleep with real (ideally I/O-bound, non-blocking-where-possible) work
# in production.
UNIT_WORK_SLEEP_S = float(os.environ.get("AEGIS_TASK_UNIT_SLEEP_S", "0.05"))


def _now():
    return datetime.utcnow()


@shared_task(bind=True, name="aegis.process_job")
def process_job(self, request_id: str):
    """
    Long-running job driver.
    - request_id: Job.request_id (string) used to find DB row
    """
    session = SessionLocal()
    try:
        job = session.query(Job).filter_by(request_id=request_id).one_or_none()
        if job is None:
            logger.error("process_job: job not found %s", request_id)
            return {"error": "job not found", "request_id": request_id}

        # mark as started
        job.status = "RUNNING"
        job.updated_at = _now()
        session.commit()
        logger.info("Started job %s", request_id)

        # Simulated preprocessing (e.g., download/convert large files, compute embeddings)
        # In your real task:
        #  - stream files from object storage
        #  - call model registry / inference functions
        #  - write outputs to object store and put references in output_payload
        payload = job.input_payload or {}
        # Example: if input contains 'batch' emulate longer processing
        work_units = payload.get("work_units", 1)
        # simulate per-unit work
        results = []
        for i in range(int(work_units)):
            # check for task revoke (soft cancel support)
            if self.request.called_directly is False and self.request.is_revoked():
                job.status = "CANCELLED"
                job.updated_at = _now()
                session.commit()
                logger.info("Job %s revoked/cancelled", request_id)
                return {"status": "cancelled", "request_id": request_id}

            logger.info("Processing unit %d/%d for job %s", i + 1, work_units, request_id)
            # simulate CPU-bound or IO-bound work; replace with real ops.
            # Uses a small, configurable sleep (see UNIT_WORK_SLEEP_S) instead
            # of a hardcoded 1s so this demo loop doesn't waste a worker
            # process/thread for longer than necessary.
            if UNIT_WORK_SLEEP_S > 0:
                time.sleep(UNIT_WORK_SLEEP_S)
            results.append({"unit": i + 1, "label": "demo", "score": 0.9})

        # Simulate postprocessing (aggregating results)
        output = {"request_id": request_id, "items": results, "summary": {"count": len(results)}}

        # Save output into DB (for small outputs). For large outputs, write to object store and reference path.
        job.output_payload = output
        job.status = "SUCCESS"
        job.updated_at = _now()
        session.commit()
        logger.info("Completed job %s", request_id)
        return {"status": "success", "request_id": request_id, "output": output}
    except Exception as exc:
        logger.exception("Job %s failed: %s", request_id, exc)
        # mark as failed
        try:
            job = session.query(Job).filter_by(request_id=request_id).one_or_none()
            if job:
                job.status = "FAILED"
                job.updated_at = _now()
                job.output_payload = {"error": str(exc)}
                session.commit()
        except Exception:
            session.rollback()
        raise
    finally:
        session.close()
