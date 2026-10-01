"""add safety_events table

Revision ID: 0002_add_safety_events
Revises: 0001_initial
Create Date: 2025-12-03 00:10:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_add_safety_events'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'safety_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('request_id', sa.String(length=100), nullable=False, index=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('model_version_id', sa.Integer(), nullable=True),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('reasons', sa.JSON(), nullable=True),
        sa.Column('input_snapshot', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_safety_events_request_id', 'safety_events', ['request_id'])
    op.create_index('ix_safety_events_decision', 'safety_events', ['decision'])


def downgrade():
    op.drop_index('ix_safety_events_decision', table_name='safety_events')
    op.drop_index('ix_safety_events_request_id', table_name='safety_events')
    op.drop_table('safety_events')
