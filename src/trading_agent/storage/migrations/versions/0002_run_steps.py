"""Daily cycle steps, for resuming a restarted cycle and for the audit trail.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import Timestamp

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_steps",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("step", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("detail", sa.Text, nullable=False),
        sa.Column("started_at", Timestamp, nullable=False),
        sa.Column("finished_at", Timestamp, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_run_steps"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_run_steps_run_id_runs"),
        sa.UniqueConstraint("run_id", "step", name="uq_run_steps_run_id_step"),
    )


def downgrade() -> None:
    op.drop_table("run_steps")
