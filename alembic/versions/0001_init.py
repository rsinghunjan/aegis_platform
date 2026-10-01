"""init core tables

Revision ID: 0001_init
Revises: 0001_create_model_audit_table
Create Date: 2026-04-22
"""
from __future__ import annotations

from alembic import context
from alembic import op
import sqlalchemy as sa

revision = "0001_init"
down_revision = "0001_create_model_audit_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    offline = context.is_offline_mode()
    tables = {"tenants", "jobs"} if offline else set(sa.inspect(op.get_bind()).get_table_names())
    if "tenants" not in tables:
        op.create_table(
            "tenants",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.current_timestamp(),
            ),
        )
        tables.add("tenants")
    if "jobs" not in tables:
        op.create_table(
            "jobs",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column(
                "tenant_id",
                sa.String(length=64),
                sa.ForeignKey("tenants.id"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.current_timestamp(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index("ix_jobs_tenant_id", "jobs", ["tenant_id"])
        op.create_index("ix_jobs_kind", "jobs", ["kind"])
        op.create_index("ix_jobs_status", "jobs", ["status"])
        tables.add("jobs")
    if "runs" not in tables:
        op.create_table(
            "runs",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.Column("job_id", sa.String(length=64), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("metrics_json", sa.JSON(), nullable=False),
            sa.Column("artifacts_uri", sa.Text(), nullable=True),
        )
        op.create_index("ix_runs_tenant_id", "runs", ["tenant_id"])
        op.create_index("ix_runs_job_id", "runs", ["job_id"])
    if "audit_log" not in tables:
        op.create_table(
            "audit_log",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.Column("actor", sa.String(length=255), nullable=False),
            sa.Column("action", sa.String(length=255), nullable=False),
            sa.Column("resource", sa.String(length=255), nullable=False),
            sa.Column("decision", sa.String(length=32), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column(
                "ts",
                sa.DateTime(timezone=True),
                server_default=sa.func.current_timestamp(),
            ),
        )
        op.create_index("ix_audit_tenant_id", "audit_log", ["tenant_id"])

    tenants = sa.table(
        "tenants",
        sa.column("id", sa.String()),
        sa.column("name", sa.String()),
        sa.column("created_at", sa.DateTime()),
    )
    connection = op.get_bind()
    if offline or connection.execute(
        sa.select(tenants.c.id).where(tenants.c.id == "default")
    ).first() is None:
        op.bulk_insert(tenants, [{"id": "default", "name": "Default Tenant"}])


def downgrade() -> None:
    if context.is_offline_mode():
        op.drop_index("ix_audit_tenant_id", table_name="audit_log")
        op.drop_table("audit_log")
        op.drop_index("ix_runs_job_id", table_name="runs")
        op.drop_index("ix_runs_tenant_id", table_name="runs")
        op.drop_table("runs")
        return

    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "audit_log" in tables:
        op.drop_index("ix_audit_tenant_id", table_name="audit_log")
        op.drop_table("audit_log")
    if "runs" in tables:
        op.drop_index("ix_runs_job_id", table_name="runs")
        op.drop_index("ix_runs_tenant_id", table_name="runs")
        op.drop_table("runs")
    if "jobs" in tables and "kind" in {
        column["name"] for column in inspector.get_columns("jobs")
    }:
        op.drop_index("ix_jobs_status", table_name="jobs")
        op.drop_index("ix_jobs_kind", table_name="jobs")
        op.drop_index("ix_jobs_tenant_id", table_name="jobs")
        op.drop_table("jobs")
        op.drop_table("tenants")
