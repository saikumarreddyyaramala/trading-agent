"""The guidebook: principles (promoted lessons), playbooks, and versions (ADR 0017).

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import JSONType, Timestamp

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("lessons") as batch:
        batch.add_column(sa.Column("promoted_at", Timestamp))
    op.create_table(
        "playbooks",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("name", sa.String(40), nullable=False),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("when_to_use", sa.Text, nullable=False),
        sa.Column("steps", JSONType, nullable=False),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source_lesson_ids", JSONType, nullable=False),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.Column("updated_at", Timestamp, nullable=False),
        sa.Column("status_reason", sa.Text),
        sa.PrimaryKeyConstraint("id", name="pk_playbooks"),
        sa.UniqueConstraint("name", name="uq_playbooks_name"),
    )
    op.create_table(
        "guidebook_versions",
        sa.Column("version", sa.Integer, autoincrement=False, nullable=False),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("changes", JSONType, nullable=False),
        sa.PrimaryKeyConstraint("version", name="pk_guidebook_versions"),
    )


def downgrade() -> None:
    op.drop_table("guidebook_versions")
    op.drop_table("playbooks")
    with op.batch_alter_table("lessons") as batch:
        batch.drop_column("promoted_at")
