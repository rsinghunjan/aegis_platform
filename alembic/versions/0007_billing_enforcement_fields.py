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
 47
 48
 49
 50
 51
 52
 53
 54
 55
"""
add billing_suspended, dunning fields to billing_accounts

Revision ID: 0007_billing_enforcement_fields
Revises: 0006_add_billing_accounts
Create Date: 2025-12-03 01:30:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0007_billing_enforcement_fields"
down_revision = "0006_add_billing_accounts"
branch_labels = None
depends_on = None


def upgrade():
    # Add billing enforcement & dunning columns to billing_accounts
    op.add_column(
        "billing_accounts",
        sa.Column("billing_suspended", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("billing_suspension_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("billing_suspended_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("suspension_expires_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("dunning_level", sa.SmallInteger(), nullable=False, server_default="0"),
    )

    # Ensure indexes remain usable; create index on tenant_id if not present (idempotent-ish)
    try:
        op.create_index("ix_billing_accounts_tenant", "billing_accounts", ["tenant_id"])
    except Exception:
        # ignore if index already exists
        pass


def downgrade():
    op.drop_index("ix_billing_accounts_tenant", table_name="billing_accounts")
    op.drop_column("billing_accounts", "dunning_level")
    op.drop_column("billing_accounts", "suspension_expires_at")
    op.drop_column("billing_accounts", "billing_suspended_at")
    op.drop_column("billing_accounts", "billing_suspension_reason")
    op.drop_column("billing_accounts", "billing_suspended")
alembic/versions/0007_billing_enforcement_fields.py
