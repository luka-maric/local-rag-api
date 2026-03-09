"""add_content_hash_to_documents

Revision ID: b7c3a1f2e8d4
Revises: 0c24de4e0ce7

Adds content_hash (SHA-256 hex digest of file bytes) to the documents table
plus a unique constraint on (tenant_id, content_hash) for deduplication.

The server_default='' is used during the ALTER TABLE so existing rows (if any)
get a valid non-null value. It is removed immediately after so the application
ORM layer is responsible for always providing the hash — no silent empty strings.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c3a1f2e8d4'
down_revision: Union[str, None] = '0c24de4e0ce7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add column with a temporary server default so existing rows satisfy NOT NULL.
    op.add_column(
        'documents',
        sa.Column('content_hash', sa.String(length=64), nullable=False, server_default=''),
    )
    # Remove the server default — the application is now responsible for providing
    # the hash. This prevents accidental empty-string hashes going forward.
    op.alter_column('documents', 'content_hash', server_default=None)

    # Unique constraint: prevents the same tenant uploading the same file twice.
    # Postgres automatically creates a B-tree index to enforce this, so we also
    # get fast O(log n) lookups for the deduplication check at no extra cost.
    op.create_unique_constraint(
        'uq_tenant_content_hash',
        'documents',
        ['tenant_id', 'content_hash'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_tenant_content_hash', 'documents', type_='unique')
    op.drop_column('documents', 'content_hash')
