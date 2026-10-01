"""add audit_logs and data_retention_policies

Revision ID: 0003_add_audit_and_retention
Revises: 0002_add_safety_events
Create Date: 2025-12-03 00:20:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0003_add_audit_and_retention'
down_revision = '0002_add_safety_events'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('action', sa.String(length=200), nullable=False),
        sa.Column('actor', sa.String(length=200), nullable=False),
        sa.Column('target_type', sa.String(length=200), nullable=False),
        sa.Column('target_id', sa.Integer(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])

    op.create_table(
        'data_retention_policies',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=200), nullable=False, unique=True),
        sa.Column('table_name', sa.String(length=200), nullable=False),
        sa.Column('timestamp_column', sa.String(length=200), nullable=False, server_default='created_at'),
        sa.Column('retention_days', sa.Integer(), nullable=False, server_default='90'),
        sa.Column('action', sa.String(length=20), nullable=False, server_default='delete'),
        sa.Column('tenant_column', sa.String(length=200), nullable=True),
        sa.Column('filter_sql', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_data_retention_policies_name', 'data_retention_policies', ['name'])


def downgrade():
    op.drop_index('ix_data_retention_policies_name', table_name='data_retention_policies')
    op.drop_table('data_retention_policies')
    op.drop_index('ix_audit_logs_action', table_name='audit_logs')
    op.drop_table('audit_logs')
