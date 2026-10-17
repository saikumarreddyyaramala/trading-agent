"""Lessons learned from the system's own results, and which lessons each analysis used.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import JSONType, Timestamp

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("lesson_ids", JSONType))
    op.create_table(
        "lessons",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("source_run_id", sa.String(64)),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.Column("expires_at", Timestamp, nullable=False),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("evidence", JSONType, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("retired_at", Timestamp),
        sa.Column("retired_reason", sa.Text),
        sa.PrimaryKeyConstraint("id", name="pk_lessons"),
        sa.ForeignKeyConstraint(
            ["source_run_id"], ["runs.run_id"], name="fk_lessons_source_run_id_runs"
        ),
    )
    op.create_index("ix_lessons_source_run_id", "lessons", ["source_run_id"])


def downgrade() -> None:
    op.drop_index("ix_lessons_source_run_id", table_name="lessons")
    op.drop_table("lessons")
    with op.batch_alter_table("analyses") as batch:
        batch.drop_column("lesson_ids")
