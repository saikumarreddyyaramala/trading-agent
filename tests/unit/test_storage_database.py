from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext

from trading_agent.storage import schema
from trading_agent.storage.database import (
    alembic_config,
    current_revision,
    head_revision,
    make_engine,
    normalize_url,
    upgrade,
)


def test_migrations_match_the_schema_exactly(db_engine: sa.Engine) -> None:
    with db_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, schema.metadata) == []


def test_upgrade_is_idempotent_and_reaches_head(db_engine: sa.Engine) -> None:
    upgrade(db_engine)
    assert current_revision(db_engine) == head_revision()
    assert head_revision() == "0003"


def test_downgrade_removes_everything(db_engine: sa.Engine) -> None:
    with db_engine.begin() as connection:
        command.downgrade(alembic_config(connection), "base")
    assert set(sa.inspect(db_engine).get_table_names()) <= {"alembic_version"}
    assert current_revision(db_engine) is None


def test_fresh_database_has_no_revision(tmp_path: Path) -> None:
    assert current_revision(make_engine(f"sqlite:///{tmp_path / 'new.db'}")) is None


@pytest.mark.parametrize(
    "value", ["0.4899", "12.500000001", "99999999999.123456789", "0", "-95.09", "102"]
)
def test_exact_decimals_round_trip(db_engine: sa.Engine, value: str) -> None:
    table = sa.Table("exact_probe", sa.MetaData(), sa.Column("v", schema.Exact))
    table.create(db_engine)
    try:
        with db_engine.begin() as connection:
            connection.execute(table.insert().values(v=Decimal(value)))
            stored = connection.execute(sa.select(table.c.v)).scalar_one()
        assert stored == Decimal(value)
        assert isinstance(stored, Decimal)
        assert "E" not in str(stored)  # never scientific notation
    finally:
        table.drop(db_engine)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h:5432/d", "postgresql+psycopg://u:p@h:5432/d"),
        ("postgres://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
        ("postgresql+psycopg://u@h/d", "postgresql+psycopg://u@h/d"),
        ("sqlite:///data/audit.db", "sqlite:///data/audit.db"),
    ],
)
def test_normalize_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


def test_sqlite_parent_folder_is_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "audit.db"
    upgrade(make_engine(f"sqlite:///{target}"))
    assert target.exists()


def test_url_with_percent_sign_survives_config() -> None:
    config = alembic_config(url="postgresql://u:p%40ss@h/d")
    assert config.get_main_option("sqlalchemy.url") == "postgresql+psycopg://u:p%40ss@h/d"


def test_values_beyond_the_column_precision_are_rejected_on_postgresql(
    db_engine: sa.Engine,
) -> None:
    """NUMERIC(20, 9) holds 11 integer digits. PostgreSQL refuses more rather than rounding."""
    if db_engine.dialect.name != "postgresql":
        pytest.skip("SQLite stores exact decimals as text and has no precision limit")
    table = sa.Table("exact_overflow", sa.MetaData(), sa.Column("v", schema.Exact))
    table.create(db_engine)
    try:
        with pytest.raises(sa.exc.DataError), db_engine.begin() as connection:
            connection.execute(table.insert().values(v=Decimal("100000000000.1")))
    finally:
        table.drop(db_engine)
