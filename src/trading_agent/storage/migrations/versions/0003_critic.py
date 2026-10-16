"""Critic reviews, and the tool output each analyst call returned.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import JSONType, Timestamp

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tool_calls") as batch:
        batch.add_column(sa.Column("result", sa.Text))
    op.create_table(
        "critiques",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("model_verdict", sa.String(16)),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("reasons", JSONType, nullable=False),
        sa.Column("concerns", JSONType, nullable=False),
        sa.Column("evidence_checks", JSONType, nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_critiques"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_critiques_run_id_runs"),
        sa.UniqueConstraint("run_id", name="uq_critiques_run_id"),
    )


def downgrade() -> None:
    op.drop_table("critiques")
    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_column("result")
