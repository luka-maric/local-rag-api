"""add_hnsw_index_on_embedding

Revision ID: c9d4e5f6a7b8
Revises: b7c3a1f2e8d4
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'c9d4e5f6a7b8'
down_revision: Union[str, None] = 'b7c3a1f2e8d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
