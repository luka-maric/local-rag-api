"""add_hnsw_index_on_embedding

Revision ID: c9d4e5f6a7b8
Revises: b7c3a1f2e8d4

Adds an HNSW (Hierarchical Navigable Small World) index on
document_chunks.embedding for fast approximate nearest-neighbour search.

WHY HNSW?
  A full vector scan (no index) compares the query vector against every row
  in the table. With 100 000 chunks at 384 floats each that is 100 000 * 384
  = 38.4M float comparisons per query. At millions of rows this becomes slow.

  HNSW is a graph-based ANN (Approximate Nearest Neighbour) algorithm. It
  builds a multi-layer graph where each node (chunk) is connected to its
  nearest neighbours. A search starts at the top layer, greedily descends
  to the bottom, and returns the k nearest nodes found — in O(log n) time
  rather than O(n).

  "Approximate" means the result might occasionally miss a true nearest
  neighbour, but in practice recall is 95-99% with default settings while
  being orders of magnitude faster than an exact scan.

WHY vector_cosine_ops?
  pgvector has three index operator classes:
    vector_l2_ops   → Euclidean distance  (<->)
    vector_ip_ops   → inner product       (<#>)
    vector_cosine_ops → cosine distance   (<=>)

  The operator class used at index-build time MUST match the distance
  operator used in queries. We use cosine distance because semantic
  embeddings encode meaning as direction, not magnitude. Two paraphrases
  of the same sentence produce vectors pointing in the same direction but
  possibly with different magnitudes — cosine handles this correctly.

WHY op.execute() INSTEAD OF op.create_index()?
  Alembic's op.create_index() does not know about HNSW (it is a
  pgvector-specific index type). Passing postgresql_using="hnsw" and
  postgresql_with=... via op.create_index() produces invalid DDL.
  op.execute() lets us write the exact SQL that Postgres expects.

INDEX PARAMETERS:
  m=16             — connections per node. Default is 16. Higher values
                     improve recall at the cost of more memory and slower
                     build time.
  ef_construction=64 — candidate list size during build. Default is 64.
                     Higher values improve index quality at the cost of
                     slower build time. Does NOT affect query speed.

QUERY-TIME PARAMETER (not set here):
  SET hnsw.ef_search = 40;   — candidate list during search. Default 40.
  Higher values improve recall but slow queries. Can be tuned per-session.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'c9d4e5f6a7b8'
down_revision: Union[str, None] = 'b7c3a1f2e8d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Raw DDL: Alembic cannot generate HNSW index syntax automatically.
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
