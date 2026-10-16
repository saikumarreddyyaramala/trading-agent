"""Alembic environment. Run through `trading-agent db upgrade`, which supplies the URL."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context

from trading_agent.storage.schema import metadata

config = context.config


def _configure(connection: sa.Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        compare_type=True,
        # SQLite cannot ALTER most things; batch mode rebuilds tables instead.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:  # tests pass an open connection
        _configure(connection)
        return
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url") or "", poolclass=sa.NullPool)
    with engine.begin() as conn:
        _configure(conn)


def run_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
