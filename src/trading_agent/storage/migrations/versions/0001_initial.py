"""Initial audit trail: runs, analyses, tool calls, risk decisions, orders, position events.

Revision ID: 0001
Revises:
Create Date: 2026-10-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import Exact, JSONType, Timestamp

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("environment", sa.String(8), nullable=False),
        sa.Column("app_version", sa.String(32), nullable=False),
        sa.Column("started_at", Timestamp, nullable=False),
        sa.Column("finished_at", Timestamp),
        sa.Column("status", sa.String(32)),
        sa.Column("summary", sa.Text),
        sa.Column("config", JSONType, nullable=False),
        sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("decision", sa.String(16)),
        sa.Column("symbol", sa.String(16)),
        sa.Column("confidence", sa.String(8)),
        sa.Column("reference_price", Exact),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("turns", sa.Integer, nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("duration_seconds", sa.Float, nullable=False),
        sa.Column("started_at", Timestamp, nullable=False),
        sa.Column("proposal", JSONType),
        sa.PrimaryKeyConstraint("id", name="pk_analyses"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_analyses_run_id_runs"),
        sa.UniqueConstraint("run_id", name="uq_analyses_run_id"),
    )
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("call_id", sa.String(128), nullable=False),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("arguments", JSONType),
        sa.Column("is_error", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tool_calls"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_tool_calls_run_id_runs"),
        sa.UniqueConstraint("run_id", "seq", name="uq_tool_calls_run_id_seq"),
    )
    op.create_index("ix_tool_calls_run_id", "tool_calls", ["run_id"])
    op.create_table(
        "risk_decisions",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(16)),
        sa.Column("approved", sa.Boolean, nullable=False),
        sa.Column("evaluated_at", Timestamp, nullable=False),
        sa.Column("failed_checks", JSONType, nullable=False),
        sa.Column("checks", JSONType, nullable=False),
        sa.Column("plan", JSONType),
        sa.PrimaryKeyConstraint("id", name="pk_risk_decisions"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_risk_decisions_run_id_runs"),
    )
    op.create_index("ix_risk_decisions_run_id", "risk_decisions", ["run_id"])
    op.create_table(
        "orders",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("client_order_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(64)),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("order_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("quantity", Exact, nullable=False),
        sa.Column("filled_quantity", Exact, nullable=False),
        sa.Column("filled_avg_price", Exact),
        sa.Column("limit_price", Exact),
        sa.Column("stop_price", Exact),
        sa.Column("submitted_at", Timestamp),
        sa.Column("recorded_at", Timestamp, nullable=False),
        sa.Column("updated_at", Timestamp, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_orders"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_orders_run_id_runs"),
        sa.UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
    )
    op.create_index("ix_orders_run_id", "orders", ["run_id"])
    op.create_table(
        "position_events",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64)),
        sa.Column("at", Timestamp, nullable=False),
        sa.Column("symbol", sa.String(16)),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("detail", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_position_events"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_position_events_run_id_runs"),
    )
    op.create_index("ix_position_events_run_id", "position_events", ["run_id"])


def downgrade() -> None:
    for index, table in (
        ("ix_position_events_run_id", "position_events"),
        ("ix_orders_run_id", "orders"),
        ("ix_risk_decisions_run_id", "risk_decisions"),
        ("ix_tool_calls_run_id", "tool_calls"),
    ):
        op.drop_index(index, table_name=table)
    for table in ("position_events", "orders", "risk_decisions", "tool_calls", "analyses", "runs"):
        op.drop_table(table)
