"""Audit trail schema (SQLAlchemy Core). PostgreSQL in production; SQLite works for local use.

Every table hangs off a run id, so one query reconstructs what happened in a run: the
analyst's research and proposal, the risk decision with every check, and each order.
Records are append-only, except orders, whose status and fills are updated as they change.
Migrations in `storage/migrations` must match this module (a test checks it).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect


class Exact(sa.types.TypeDecorator[Decimal]):
    """An exact decimal: NUMERIC on PostgreSQL, text on SQLite (which has no exact decimal)."""

    impl = sa.Numeric(20, 9)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> sa.types.TypeEngine[Any]:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(sa.String(40))
        return dialect.type_descriptor(sa.Numeric(20, 9))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        return str(value) if dialect.name == "sqlite" else Decimal(str(value))

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        # Drop trailing zeros (NUMERIC pads to 9 places) without switching to exponent
        # notation: 102.000000000 reads back as 102, never 1.02E+2.
        return Decimal(f"{Decimal(str(value)).normalize():f}")


JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
Timestamp = sa.DateTime(timezone=True)

metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

runs = sa.Table(
    "runs",
    metadata,
    sa.Column("run_id", sa.String(64), primary_key=True),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("environment", sa.String(8), nullable=False),
    sa.Column("app_version", sa.String(32), nullable=False),
    sa.Column("started_at", Timestamp, nullable=False),
    sa.Column("finished_at", Timestamp),
    sa.Column("status", sa.String(32)),
    sa.Column("summary", sa.Text),
    # The effective configuration (never secrets), so every run is reproducible.
    sa.Column("config", JSONType, nullable=False),
)

analyses = sa.Table(
    "analyses",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False, unique=True),
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
)

tool_calls = sa.Table(
    "tool_calls",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False, index=True),
    sa.Column("seq", sa.Integer, nullable=False),
    sa.Column("call_id", sa.String(128), nullable=False),
    sa.Column("tool", sa.String(64), nullable=False),
    sa.Column("arguments", JSONType),
    sa.Column("is_error", sa.Boolean, nullable=False),
    sa.UniqueConstraint("run_id", "seq", name="uq_tool_calls_run_id_seq"),
)

risk_decisions = sa.Table(
    "risk_decisions",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False, index=True),
    sa.Column("symbol", sa.String(16)),
    sa.Column("approved", sa.Boolean, nullable=False),
    sa.Column("evaluated_at", Timestamp, nullable=False),
    sa.Column("failed_checks", JSONType, nullable=False),
    sa.Column("checks", JSONType, nullable=False),
    sa.Column("plan", JSONType),
)

orders = sa.Table(
    "orders",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),  # the broker's order id
    sa.Column("client_order_id", sa.String(128), nullable=False, unique=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), index=True),
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
)

position_events = sa.Table(
    "position_events",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), index=True),
    sa.Column("at", Timestamp, nullable=False),
    sa.Column("symbol", sa.String(16)),
    sa.Column("action", sa.String(32), nullable=False),
    sa.Column("detail", sa.Text, nullable=False),
)

# One row per step of a daily cycle; lets a restarted process resume instead of redoing work.
run_steps = sa.Table(
    "run_steps",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False),
    sa.Column("step", sa.String(32), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("detail", sa.Text, nullable=False),
    sa.Column("started_at", Timestamp, nullable=False),
    sa.Column("finished_at", Timestamp, nullable=False),
    sa.UniqueConstraint("run_id", "step", name="uq_run_steps_run_id_step"),
)
