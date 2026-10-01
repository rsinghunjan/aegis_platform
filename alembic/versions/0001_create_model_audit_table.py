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
"""create model_audit table

Revision ID: 0001_create_model_audit_table
Revises: 
Create Date: 2025-12-05 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_create_model_audit_table'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'model_audit',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_name', sa.String(256), nullable=False, index=True),
        sa.Column('model_version', sa.String(64), nullable=True),
        sa.Column('actor', sa.String(256), nullable=False),
        sa.Column('action', sa.String(64), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('verification_passed', sa.Boolean, nullable=False, server_default=sa.sql.expression.false()),
        sa.Column('verification_details', sa.JSON, nullable=True),
        sa.Column('signature_issuer', sa.String(256), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
    )

def downgrade():
    op.drop_table('model_audit')
alembic/versions/0001_create_model_audit_table.py
