"""pgvector + rag tables

Revision ID: 0002_pgvector_rag
Revises: 0001_init
Create Date: 2026-04-22
"""
from __future__ import annotations

import os
from alembic import op
import sqlalchemy as sa

revision = "0002_pgvector_rag"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dim = int(os.getenv("AEGIS_EMBEDDING_DIM", "1536"))

    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # record schema config for visibility
    op.create_table(
        "schema_settings",
        sa.Column("key", sa.String(length=128), primary_key=True),
        sa.Column("value", sa.String(length=1024), nullable=False),
    )
    op.execute(
        f"INSERT INTO schema_settings (key, value) VALUES ('embedding_dim', '{dim}') "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;"
    )

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_rag_docs_tenant_ns", "rag_documents", ["tenant_id", "namespace"])
    op.create_index("ix_rag_docs_doc_id", "rag_documents", ["doc_id"])

    # embedding column uses pgvector type: vector(dim)
    op.execute(
        f"""
        CREATE TABLE rag_chunks (
          id SERIAL PRIMARY KEY,
          tenant_id VARCHAR(64) NOT NULL,
          namespace VARCHAR(128) NOT NULL,
          doc_id VARCHAR(255) NOT NULL,
          chunk_id VARCHAR(255) NOT NULL,
          text TEXT NOT NULL,
          metadata_json JSON NOT NULL,
          embedding vector({dim}) NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX ix_rag_chunks_tenant_ns ON rag_chunks (tenant_id, namespace);")
    op.execute("CREATE INDEX ix_rag_chunks_doc_id ON rag_chunks (doc_id);")
    op.execute("CREATE INDEX ix_rag_chunks_embedding ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_embedding;")
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_doc_id;")
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_tenant_ns;")
    op.execute("DROP TABLE IF EXISTS rag_chunks;")

    op.drop_index("ix_rag_docs_doc_id", table_name="rag_documents")
    op.drop_index("ix_rag_docs_tenant_ns", table_name="rag_documents")
    op.drop_table("rag_documents")

    op.drop_table("schema_settings")
