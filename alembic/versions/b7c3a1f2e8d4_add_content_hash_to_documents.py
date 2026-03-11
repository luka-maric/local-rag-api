"""add_content_hash_to_documents

Revision ID: b7c3a1f2e8d4
Revises: 0c24de4e0ce7
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c3a1f2e8d4'
down_revision: Union[str, None] = '0c24de4e0ce7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('content_hash', sa.String(length=64), nullable=False, server_default=''),
    )
    op.alter_column('documents', 'content_hash', server_default=None)

    op.create_unique_constraint(
        'uq_tenant_content_hash',
        'documents',
        ['tenant_id', 'content_hash'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_tenant_content_hash', 'documents', type_='unique')
    op.drop_column('documents', 'content_hash')
