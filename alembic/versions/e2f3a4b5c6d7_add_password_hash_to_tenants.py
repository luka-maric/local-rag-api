"""add_password_hash_to_tenants

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "password_hash",
            sa.String(255),
            nullable=False,
            server_default="",
        ),
    )
    op.alter_column("tenants", "password_hash", server_default=None)


def downgrade() -> None:
    op.drop_column("tenants", "password_hash")
