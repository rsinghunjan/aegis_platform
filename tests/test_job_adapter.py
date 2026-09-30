from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import api.tasks as tasks
from api.models import Base, Job


def make_job_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(tasks, "SessionLocal", sessions)
    return sessions


def test_registered_job_handler_lifecycle_is_idempotent(tmp_path, monkeypatch):
    sessions = make_job_session(tmp_path, monkeypatch)
    with sessions() as session:
        session.add(
            Job(
                request_id="job-1",
                status="PENDING",
                input_payload={"kind": "sum", "numbers": [1, 2]},
            )
        )
        session.commit()

    calls = []
    tasks.register_job_handler(
        "sum",
        lambda payload: calls.append(payload) or {"total": sum(payload["numbers"])},
    )
    first = tasks.process_job.run("job-1")
    second = tasks.process_job.run("job-1")
    assert first["status"] == "SUCCESS"
    assert second["status"] == "SUCCESS"
    assert first["output"] == {"total": 3}
    assert calls == [{"numbers": [1, 2]}]


def test_unregistered_job_does_not_return_demo_output(tmp_path, monkeypatch):
    sessions = make_job_session(tmp_path, monkeypatch)
    with sessions() as session:
        session.add(
            Job(
                request_id="job-2",
                status="PENDING",
                input_payload={"kind": "no_handler"},
            )
        )
        session.commit()

    result = tasks.process_job.run("job-2")
    assert result["status"] == "FAILED"
    with sessions() as session:
        assert session.query(Job).filter_by(request_id="job-2").one().status == "FAILED"


def test_job_cancellation_marks_state_and_revokes_task(tmp_path, monkeypatch):
    sessions = make_job_session(tmp_path, monkeypatch)
    with sessions() as session:
        session.add(
            Job(
                request_id="job-3",
                status="PENDING",
                input_payload={"_job_meta": {"celery_task_id": "celery-3"}},
            )
        )
        session.commit()
    revoked = []
    monkeypatch.setattr(
        tasks,
        "current_app",
        SimpleNamespace(control=SimpleNamespace(revoke=lambda task_id, terminate: revoked.append(task_id))),
    )

    assert tasks.cancel_job("job-3")
    assert revoked == ["celery-3"]
    with sessions() as session:
        assert session.query(Job).filter_by(request_id="job-3").one().status == "CANCELLED"
