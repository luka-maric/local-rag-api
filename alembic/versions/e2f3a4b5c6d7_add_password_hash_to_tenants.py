"""add_password_hash_to_tenants

Adds the password_hash column to the tenants table for Phase 11 JWT auth.

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6

WHY server_default="" THEN ALTER TO REMOVE IT?
  Adding a NOT NULL column to a table with existing rows requires a default
  so Postgres can fill the existing rows. We set server_default="" as a
  temporary placeholder, then immediately remove the server_default so that
  future INSERTs without a password_hash are rejected by the DB constraint.
  Any existing tenants (from earlier dev sessions) get an empty string —
  they'll need to re-register after this migration.
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
    # Remove server_default — future rows must always provide the hash explicitly.
    op.alter_column("tenants", "password_hash", server_default=None)


def downgrade() -> None:
    op.drop_column("tenants", "password_hash")
