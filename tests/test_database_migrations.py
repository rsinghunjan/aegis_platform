from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect


def _config(database_url):
    config = Config("alembic.ini")
    config.attributes["database_url"] = database_url
    return config


def test_migration_history_is_a_single_chain():
    script = ScriptDirectory.from_config(Config("alembic.ini"))

    assert script.get_bases() == ["0001_initial"]
    assert script.get_heads() == ["0008_agent_memory_schema"]


def test_sqlite_agent_and_memory_migrations_upgrade_and_downgrade(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migrations.db'}"
    config = _config(database_url)
    engine = create_engine(database_url)

    command.upgrade(config, "head")

    inspector = inspect(engine)
    assert {
        "agent_runs",
        "agent_plan_steps",
        "agent_tool_calls",
        "agent_policy_decisions",
        "agent_approvals",
        "agent_capabilities",
        "agent_evidence",
        "agent_evidence_anchors",
        "conversation_memory",
        "billing_accounts",
        "model_audit",
        "runs",
        "audit_log",
        "rag_documents",
        "rag_chunks",
    }.issubset(inspector.get_table_names())
    assert {
        "requested_at",
        "expires_at",
        "policy_version",
    }.issubset({column["name"] for column in inspector.get_columns("agent_approvals")})
    assert "previous_sha256" in {
        column["name"] for column in inspector.get_columns("agent_evidence")
    }

    command.downgrade(config, "0002_pgvector_rag")
    assert "agent_runs" not in inspect(engine).get_table_names()
    assert "rag_documents" in inspect(engine).get_table_names()

    command.upgrade(config, "head")
    command.downgrade(config, "base")
    assert "agent_runs" not in inspect(engine).get_table_names()
    assert "conversation_memory" not in inspect(engine).get_table_names()
    engine.dispose()
