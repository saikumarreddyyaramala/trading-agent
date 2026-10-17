"""The parametric trade score kept with every risk decision (ADR 0015).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from trading_agent.storage.schema import JSONType

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("risk_decisions") as batch:
        batch.add_column(sa.Column("score", JSONType))


def downgrade() -> None:
    with op.batch_alter_table("risk_decisions") as batch:
        batch.drop_column("score")
