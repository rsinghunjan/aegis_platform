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
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
"""add tenants, tenant_quotas, usage_records, invoices

Revision ID: 0004_add_tenant_usage_invoice
Revises: 0003_add_audit_and_retention
Create Date: 2025-12-03 00:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0004_add_tenant_usage_invoice'
down_revision = '0003_add_audit_and_retention'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'tenants',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('name', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )

    op.create_table(
        'tenant_quotas',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', sa.String(length=100), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('rate_per_min', sa.Integer(), nullable=True),
        sa.Column('burst', sa.Integer(), nullable=True),
        sa.Column('daily_quota_units', sa.Integer(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_tenant_quotas_tenant_id', 'tenant_quotas', ['tenant_id'])

    op.create_table(
        'usage_records',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', sa.String(length=100), nullable=True),
        sa.Column('model_name', sa.String(length=200), nullable=False),
        sa.Column('version', sa.String(length=100), nullable=False),
        sa.Column('units', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('inference_ms', sa.Float(), nullable=True),
        sa.Column('cost_estimate', sa.Float(), nullable=True),
        sa.Column('extra', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_usage_records_tenant', 'usage_records', ['tenant_id'])

    op.create_table(
        'invoices',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', sa.String(length=100), nullable=True),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('amount', sa.Float(), nullable=False, server_default='0'),
        sa.Column('currency', sa.String(length=10), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='issued'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_invoices_tenant', 'invoices', ['tenant_id'])


def downgrade():
    op.drop_index('ix_invoices_tenant', table_name='invoices')
    op.drop_table('invoices')
    op.drop_index('ix_usage_records_tenant', table_name='usage_records')
    op.drop_table('usage_records')
    op.drop_index('ix_tenant_quotas_tenant_id', table_name='tenant_quotas')
    op.drop_table('tenant_quotas')
    op.drop_table('tenants')
