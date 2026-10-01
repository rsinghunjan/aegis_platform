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
"""add triage_items table

Revision ID: 0005_add_triage_items
Revises: 0004_add_tenant_usage_invoice
Create Date: 2025-12-03 00:40:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0005_add_triage_items'
down_revision = '0004_add_tenant_usage_invoice'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'triage_items',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('safety_event_id', sa.Integer(), sa.ForeignKey('safety_events.id', ondelete='SET NULL'), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='open'),
        sa.Column('assigned_to', sa.String(length=200), nullable=True),
        sa.Column('reasons', sa.JSON(), nullable=True),
        sa.Column('input_snapshot', sa.Text(), nullable=True),
        sa.Column('review_note', sa.Text(), nullable=True),
        sa.Column('resolution', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_triage_safety_event', 'triage_items', ['safety_event_id'])


def downgrade():
    op.drop_index('ix_triage_safety_event', table_name='triage_items')
    op.drop_table('triage_items')
