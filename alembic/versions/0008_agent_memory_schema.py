"""add durable agent and conversation memory tables

Revision ID: 0008_agent_memory_schema
Revises: 0002_pgvector_rag
Create Date: 2026-10-01
"""
from alembic import op
import sqlalchemy as sa


revision = "0008_agent_memory_schema"
down_revision = "0002_pgvector_rag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("run_id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("goal_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("risk", sa.String(length=16), nullable=False),
        sa.Column("budget", sa.Float(), nullable=False),
        sa.Column("spent_cost", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("plan_json", sa.JSON(), nullable=True),
        sa.Column("final_result_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("tenant_id", "idempotency_key"),
    )
    op.create_index("ix_agent_runs_tenant_id", "agent_runs", ["tenant_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "agent_plan_steps",
        sa.Column("step_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("input_json", sa.JSON(), nullable=False),
        sa.Column("acceptance_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("result_hash", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_plan_steps_run_id", "agent_plan_steps", ["run_id"])
    op.create_index("ix_agent_plan_steps_tenant_id", "agent_plan_steps", ["tenant_id"])
    op.create_index("ix_agent_plan_steps_status", "agent_plan_steps", ["status"])

    op.create_table(
        "agent_tool_calls",
        sa.Column("call_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("output_hash", sa.String(length=64), nullable=True),
        sa.Column("result_metadata", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_tool_calls_run_id", "agent_tool_calls", ["run_id"])
    op.create_index("ix_agent_tool_calls_tenant_id", "agent_tool_calls", ["tenant_id"])
    op.create_index("ix_agent_tool_calls_step_id", "agent_tool_calls", ["step_id"])

    op.create_table(
        "agent_policy_decisions",
        sa.Column("decision_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_policy_decisions_run_id", "agent_policy_decisions", ["run_id"])
    op.create_index("ix_agent_policy_decisions_tenant_id", "agent_policy_decisions", ["tenant_id"])

    op.create_table(
        "agent_approvals",
        sa.Column("approval_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("requested_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("sla_seconds", sa.Integer(), nullable=False, server_default="3600"),
        sa.Column("escalation_status", sa.String(length=32), nullable=False, server_default="none"),
        sa.Column("denial_reason", sa.Text(), nullable=True),
        sa.Column("capability_version", sa.String(length=64), nullable=True),
        sa.Column("plan_hash", sa.String(length=64), nullable=True),
        sa.Column("policy_version", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_agent_approvals_run_id", "agent_approvals", ["run_id"])
    op.create_index("ix_agent_approvals_tenant_id", "agent_approvals", ["tenant_id"])
    op.create_index("ix_agent_approvals_step_id", "agent_approvals", ["step_id"])

    op.create_table(
        "agent_capabilities",
        sa.Column("name", sa.String(length=128), primary_key=True),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("spec_hash", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "agent_evidence",
        sa.Column("evidence_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("previous_sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_evidence_run_id", "agent_evidence", ["run_id"])
    op.create_index("ix_agent_evidence_tenant_id", "agent_evidence", ["tenant_id"])

    op.create_table(
        "agent_evidence_anchors",
        sa.Column("evidence_anchor_id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.run_id"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("head_sha256", sa.String(length=64), nullable=False),
        sa.Column("backend", sa.String(length=128), nullable=False),
        sa.Column("proof_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "run_id",
            "tenant_id",
            "head_sha256",
            "backend",
            name="uq_agent_evidence_anchor_head",
        ),
    )
    op.create_index("ix_agent_evidence_anchors_run_id", "agent_evidence_anchors", ["run_id"])
    op.create_index("ix_agent_evidence_anchors_tenant_id", "agent_evidence_anchors", ["tenant_id"])

    op.create_table(
        "conversation_memory",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversation_memory_tenant_id", "conversation_memory", ["tenant_id"])
    op.create_index("ix_conversation_memory_session_id", "conversation_memory", ["session_id"])
    op.create_index("ix_conversation_memory_created_at", "conversation_memory", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_conversation_memory_created_at", table_name="conversation_memory")
    op.drop_index("ix_conversation_memory_session_id", table_name="conversation_memory")
    op.drop_index("ix_conversation_memory_tenant_id", table_name="conversation_memory")
    op.drop_table("conversation_memory")
    op.drop_index("ix_agent_evidence_anchors_tenant_id", table_name="agent_evidence_anchors")
    op.drop_index("ix_agent_evidence_anchors_run_id", table_name="agent_evidence_anchors")
    op.drop_table("agent_evidence_anchors")
    op.drop_index("ix_agent_evidence_tenant_id", table_name="agent_evidence")
    op.drop_index("ix_agent_evidence_run_id", table_name="agent_evidence")
    op.drop_table("agent_evidence")
    op.drop_table("agent_capabilities")
    op.drop_index("ix_agent_approvals_step_id", table_name="agent_approvals")
    op.drop_index("ix_agent_approvals_tenant_id", table_name="agent_approvals")
    op.drop_index("ix_agent_approvals_run_id", table_name="agent_approvals")
    op.drop_table("agent_approvals")
    op.drop_index("ix_agent_policy_decisions_tenant_id", table_name="agent_policy_decisions")
    op.drop_index("ix_agent_policy_decisions_run_id", table_name="agent_policy_decisions")
    op.drop_table("agent_policy_decisions")
    op.drop_index("ix_agent_tool_calls_step_id", table_name="agent_tool_calls")
    op.drop_index("ix_agent_tool_calls_tenant_id", table_name="agent_tool_calls")
    op.drop_index("ix_agent_tool_calls_run_id", table_name="agent_tool_calls")
    op.drop_table("agent_tool_calls")
    op.drop_index("ix_agent_plan_steps_status", table_name="agent_plan_steps")
    op.drop_index("ix_agent_plan_steps_tenant_id", table_name="agent_plan_steps")
    op.drop_index("ix_agent_plan_steps_run_id", table_name="agent_plan_steps")
    op.drop_table("agent_plan_steps")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_id", table_name="agent_runs")
    op.drop_table("agent_runs")
