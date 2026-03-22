
# (excerpt) server routes for async job submission & status
# Add the new endpoints to the existing api/api_server.py file you already have.
# I am showing the new route handlers below — drop them into the file after registry exists.

from fastapi import BackgroundTasks
from api.tasks import process_job
from api.db import SessionLocal
from api.models import Job
from sqlalchemy.exc import SQLAlchemyError

# ... existing imports and code above remain unchanged ...

@app.post("/v1/jobs", status_code=202)
async def create_job(payload: dict, current_user = Depends(auth.require_scopes(["predict"]))):
    """
    Enqueue an async job for long-running processing (preprocessing, batch infer, etc).

    Request body: JSON with fields:
      - model_name (optional): string
      - version (optional): string
      - parameters: dict
      - work_units: integer (example to simulate longer work)
    Returns:
      {"request_id": "<id>", "status": "PENDING"}
    """
    session = SessionLocal()
    try:
        # create a Job row
        request_id = str(uuid.uuid4())
        job = Job(
            request_id=request_id,
            user_id=None if not current_user else getattr(current_user, "id", None),
            model_version_id=None,
            status="PENDING",
            input_payload=payload,
            output_payload=None,
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        # enqueue Celery task (use request_id to correlate)
        task = process_job.apply_async(args=[request_id], queue="aegis_tasks")

        # Optionally store celery task id back in job (extend model if desired)
        job_meta = {"celery_task_id": task.id}
        job.input_payload = {**(job.input_payload or {}), "_job_meta": job_meta}
        session.commit()

        return {"request_id": request_id, "status": "PENDING"}
    except SQLAlchemyError:
        session.rollback()
        raise HTTPException(status_code=500, detail="database error")
    finally:
        session.close()


@app.get("/v1/jobs/{request_id}")
async def get_job_status(request_id: str, current_user = Depends(auth.require_scopes(["model:read"]))):
    """
    Get job status and (small) output payload if available.
    """
    session = SessionLocal()
    try:
        job = session.query(Job).filter_by(request_id=request_id).one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        # In multi-tenant setup, enforce ownership here (check job.user_id vs current_user.id)
        return {
            "request_id": job.request_id,
            "status": job.status,
            "input_payload": job.input_payload,
            "output_payload": job.output_payload,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        }
    finally:
        session.close()


@app.get("/v1/jobs")
async def list_jobs(limit: int = 50, offset: int = 0, current_user = Depends(auth.require_scopes(["model:read"]))):
    """
    List jobs (for admin / model:read scope). In production, add tenant filtering.
    """
    session = SessionLocal()
    try:
        q = session.query(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
        items = []
        for job in q:
            items.append({
                "request_id": job.request_id,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            })
        return {"jobs": items}
    finally:
        session.close()
