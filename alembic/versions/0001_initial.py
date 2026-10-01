"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2025-12-03 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(length=150), nullable=False, unique=True),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column('disabled', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_users_username', 'users', ['username'])

    # refresh_tokens
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token', sa.Text(), nullable=False, unique=True),
        sa.Column('revoked', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_refresh_tokens_token', 'refresh_tokens', ['token'])

    # models
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=200), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_models_name', 'models', ['name'])

    # model_versions
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(length=100), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('artifact_path', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_unique_constraint('uq_model_version', 'model_versions', ['model_id', 'version'])

    # jobs
    op.create_table(
        'jobs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('request_id', sa.String(length=100), nullable=False, unique=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('model_version_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('input_payload', sa.JSON(), nullable=True),
        sa.Column('output_payload', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_jobs_request_id', 'jobs', ['request_id'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])


def downgrade():
    op.drop_index('ix_jobs_status', table_name='jobs')
    op.drop_index('ix_jobs_request_id', table_name='jobs')
    op.drop_table('jobs')
    op.drop_table('model_versions')
    op.drop_index('ix_models_name', table_name='models')
    op.drop_table('models')
    op.drop_index('ix_refresh_tokens_token', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
