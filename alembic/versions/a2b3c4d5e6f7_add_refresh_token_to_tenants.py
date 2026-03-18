"""add_refresh_token_to_tenants

Revision ID: a2b3c4d5e6f7
Revises: f3a4b5c6d7e8
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'a2b3c4d5e6f7'
down_revision: Union[str, None] = 'f3a4b5c6d7e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'tenants',
        sa.Column('refresh_token_hash', sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('tenants', 'refresh_token_hash')
