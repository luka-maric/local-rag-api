"""add_chat_tables

Revision ID: d1e2f3a4b5c6
Revises: c9d4e5f6a7b8

Adds:
  chat_sessions — one row per conversation thread (tenant-scoped)
  chat_messages — one row per user or assistant turn within a session

WHY TWO TABLES?
  A session is a container for messages. One session has many messages.
  This normalised design lets us:
    - Load a session by ID without pulling all its messages.
    - Load the last N messages without scanning the entire history.
    - Delete a session and cascade-delete all its messages in one transaction.

INDEXES:
  ix_chat_sessions_tenant_id  — fast lookup of all sessions for a tenant.
  ix_chat_messages_session_id — fast lookup of all messages for a session.

  Both are B-tree indexes on UUID foreign key columns. Without them, loading
  a session's messages would require a full-table scan of chat_messages.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c9d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_sessions_tenant_id", "chat_sessions", ["tenant_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_sessions_tenant_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
