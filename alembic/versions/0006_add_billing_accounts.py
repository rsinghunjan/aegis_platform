"""add billing_accounts table

Revision ID: 0006_add_billing_accounts
Revises: 0005_add_triage_items
Create Date: 2025-12-03 01:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


revision = "0006_add_billing_accounts"
down_revision = "0005_add_triage_items"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "billing_accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(length=200),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("gateway_customer_id", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index("ix_billing_accounts_tenant_id", "billing_accounts", ["tenant_id"])


def downgrade():
    op.drop_index("ix_billing_accounts_tenant_id", table_name="billing_accounts")
    op.drop_table("billing_accounts")
