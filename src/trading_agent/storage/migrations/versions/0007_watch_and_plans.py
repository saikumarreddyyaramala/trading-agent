"""Pre-market and after-hours scans, and the next-day plan (ADR 0016).

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import JSONType, Timestamp

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_watch",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64)),
        sa.Column("day", sa.Date, nullable=False),
        sa.Column("phase", sa.String(8), nullable=False),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.Column("report", JSONType, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_market_watch"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_market_watch_run_id_runs"),
    )
    op.create_index("ix_market_watch_run_id", "market_watch", ["run_id"])
    op.create_index("ix_market_watch_day", "market_watch", ["day"])
    op.create_table(
        "day_plans",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("source_run_id", sa.String(64)),
        sa.Column("source_day", sa.Date, nullable=False),
        sa.Column("for_day", sa.Date, nullable=False),
        sa.Column("created_at", Timestamp, nullable=False),
        sa.Column("items", JSONType, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_day_plans"),
        sa.ForeignKeyConstraint(
            ["source_run_id"], ["runs.run_id"], name="fk_day_plans_source_run_id_runs"
        ),
    )
    op.create_index("ix_day_plans_source_run_id", "day_plans", ["source_run_id"])
    op.create_index("ix_day_plans_for_day", "day_plans", ["for_day"])


def downgrade() -> None:
    op.drop_index("ix_day_plans_for_day", table_name="day_plans")
    op.drop_index("ix_day_plans_source_run_id", table_name="day_plans")
    op.drop_table("day_plans")
    op.drop_index("ix_market_watch_day", table_name="market_watch")
    op.drop_index("ix_market_watch_run_id", table_name="market_watch")
    op.drop_table("market_watch")
