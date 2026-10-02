Database migrations & seeding
---------------------------------

What I added:
- SQLAlchemy models: api/models.py
- DB setup: api/db.py
- Alembic scaffolding: alembic/ + alembic.ini
- Initial migration: alembic/versions/0001_initial.py
- Local Postgres dev compose: docker-compose.postgres.yml
- Seed script: scripts/seed_db.py
- DB requirements: api/requirements-db.txt

The active Alembic history is one linear chain rooted at `0001_initial`.
`0006_add_billing_accounts` supplies the previously missing parent of the
billing-enforcement revision; the agent and memory tables are added by
`0008_agent_memory_schema`.

Quick local dev steps
1) Start Postgres:
   docker compose -f docker-compose.postgres.yml up -d

2) Install Python deps (use a venv):
   python -m venv .venv
   source .venv/bin/activate
   pip install -r api/requirements-db.txt

3) Point DATABASE_URL (optional) — default is postgres://postgres:password@localhost:5432/aegis
   export DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/aegis

4) Run Alembic migrations:
   alembic upgrade head

   If alembic command not found, run:
   python -m alembic upgrade head

For local SQLite development:
   DATABASE_URL=sqlite:///./aegis.db alembic upgrade head

Inspect the revision chain before deploying:
   alembic history

5) Seed the DB (creates admin/alice and a demo model):
   python scripts/seed_db.py

Agent and persistent-memory stores use the same `alembic upgrade head` flow for
their configured database URLs; run the command before deploying when databases
are managed separately. Add schema changes as Alembic revisions, review them
with `alembic history`, and test both `alembic upgrade head` and
`alembic downgrade base` against SQLite. PostgreSQL deployments should also run
upgrade and downgrade checks against their deployment database configuration.

In production, keep `DATABASE_URL` in a secrets manager and apply migrations
before starting services.
