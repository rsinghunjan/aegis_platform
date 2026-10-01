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


def downgrade():
    op.drop_column("billing_accounts", "dunning_level")
    op.drop_column("billing_accounts", "suspension_expires_at")
    op.drop_column("billing_accounts", "billing_suspended_at")
    op.drop_column("billing_accounts", "billing_suspension_reason")
    op.drop_column("billing_accounts", "billing_suspended")
