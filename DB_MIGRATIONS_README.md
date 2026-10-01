DB migrations & seeding (Postgres)
---------------------------------

What I added:
- SQLAlchemy models: api/models.py
- DB setup: api/db.py
- Alembic scaffolding: alembic/ + alembic.ini
- Initial migration: alembic/versions/0001_initial.py
- Local Postgres dev compose: docker-compose.postgres.yml
- Seed script: scripts/seed_db.py
- DB requirements: api/requirements-db.txt

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

5) Seed the DB (creates admin/alice and a demo model):
   python scripts/seed_db.py

Notes & next steps
- In production, keep DATABASE_URL in a secrets manager and do not use plaintext passwords.
- Replace Base.metadata.create_all() in the seed script with alembic migrations; create_all remains for convenience in dev only.
- Add Alembic versions for future schema changes; use `alembic revision --autogenerate -m "msg"`.
- Integrate migration step into your CI/CD deploy pipeline (run alembic upgrade before starting services).
