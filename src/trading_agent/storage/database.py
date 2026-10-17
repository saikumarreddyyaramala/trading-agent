"""Database connections and schema migrations (Alembic, driven programmatically)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

MIGRATIONS = Path(__file__).parent / "migrations"


def normalize_url(url: str) -> str:
    """Use the psycopg 3 driver for PostgreSQL URLs written the common way."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def make_engine(url: str) -> sa.Engine:
    url = normalize_url(url)
    parsed = sa.make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return sa.create_engine(url, pool_pre_ping=True)

    if parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(url)

    @sa.event.listens_for(engine, "connect")
    def _enforce_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        # SQLite ignores foreign keys unless each connection turns them on.
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        # Write-ahead logging lets the dashboard read while the cycle writes (ADR 0018).
        if parsed.database not in (None, "", ":memory:"):
            dbapi_connection.execute("PRAGMA journal_mode=WAL")

    return engine


def alembic_config(connection: sa.Connection | None = None, url: str | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    if url is not None:
        # ConfigParser treats % as interpolation; escape it (passwords may contain it).
        config.set_main_option("sqlalchemy.url", normalize_url(url).replace("%", "%%"))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:  # pragma: no cover - the package always ships migrations
        raise RuntimeError("no migrations found")
    return head


def current_revision(engine: sa.Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade(engine: sa.Engine) -> None:
    """Apply every pending migration."""
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")
