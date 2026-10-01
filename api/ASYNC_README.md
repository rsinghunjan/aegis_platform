  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
Async jobs & worker (Celery) — Quickstart
---------------------------------------

What was added:
- api/celery_app.py        (Celery factory)
- api/tasks.py             (example long-running job task)
- new async job routes in api/api_server.py: POST /v1/jobs, GET /v1/jobs/{id}, GET /v1/jobs
- docker/docker-compose.celery.yml  (dev compose file with redis + worker)
- requirements for Celery: api/requirements-celery.txt

Run locally (dev)
1. Start dependencies:
   docker compose -f docker/docker-compose.celery.yml up -d

2. Install python deps in a venv:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r api/requirements-celery.txt

3. Apply migrations and seed DB (if not already):
   export DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/aegis
   alembic upgrade head
   python scripts/seed_db.py

4. Start API (if not using docker api service):
   AEGIS_BROKER_URL=redis://localhost:6379/0 python -m api.api_server

5. Start Celery worker (if not using docker worker service):
   export AEGIS_BROKER_URL=redis://localhost:6379/0
   celery -A api.celery_app.app worker -Q aegis_tasks -l info

Example usage
- Enqueue job:
  curl -X POST "http://localhost:8080/v1/jobs" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" \
    -d '{"work_units": 3, "parameters": {"batch": true}}'

- Check job:
  curl -X GET "http://localhost:8080/v1/jobs/<request_id>" -H "Authorization: Bearer <TOKEN>"

Notes & production considerations
- For large outputs, write to object store (S3/GCS) and store reference paths in Job.output_payload.
- Use separate queues for GPU-bound and CPU-bound tasks so workers can be sized appropriately.
- Add task retry/backoff policies and dead-letter handling.
- Consider using Celery beat or an external scheduler for scheduled jobs (periodic embedding refresh).
- Implement job cancellation (store celery task id in DB and call revoke; ensure tasks check for revoke flag).
- Secure broker and backend (use TLS + auth for Redis/RabbitMQ) and isolate networks.
