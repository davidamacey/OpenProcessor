"""The revisions on a real PostgreSQL: upgrade, drift check, downgrade and one head."""

from __future__ import annotations

import pytest
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from scripts.db import migrate
from src.db.base import Base


pytestmark = pytest.mark.db

HEAD = '0002_research_seams'


def test_there_is_exactly_one_head() -> None:
    heads = ScriptDirectory.from_config(migrate.alembic_config()).get_heads()
    assert heads == [HEAD]


def test_upgrade_head_creates_every_modelled_table(pg) -> None:
    with pg.engine('owner').connect() as conn:
        tables = set(inspect(conn).get_table_names())
    assert set(Base.metadata.tables) <= tables
    assert migrate.current(pg.settings) == HEAD


def test_alembic_check_finds_no_difference_between_models_and_database(pg) -> None:
    migrate.check(pg.settings)


def test_downgrade_to_base_then_upgrade_again(pg) -> None:
    from alembic import command

    engine = pg.engine('owner')
    cfg = migrate.alembic_config()
    with engine.connect() as conn:
        cfg.attributes['connection'] = conn
        command.downgrade(cfg, 'base')
        conn.commit()
        left = set(inspect(conn).get_table_names()) - {'alembic_version'}
        functions = conn.execute(
            text(
                "SELECT proname FROM pg_proc WHERE proname LIKE 'op\\_%' OR proname LIKE 'jobs\\_%'"
            )
        ).all()
        assert not left
        assert not functions
    engine.dispose()
    # The runner upgrades again and re-applies the application role's grants.
    migrate.upgrade(pg.settings, super_user=pg.server.super_user)
    with pg.engine('owner').connect() as conn:
        assert set(Base.metadata.tables) <= set(inspect(conn).get_table_names())


def test_downgrade_to_0001_then_upgrade_again(pg) -> None:
    from alembic import command

    engine = pg.engine('owner')
    cfg = migrate.alembic_config()
    with engine.connect() as conn:
        cfg.attributes['connection'] = conn
        command.downgrade(cfg, '0001_control_plane')
        conn.commit()
        tables = set(inspect(conn).get_table_names())
        assert 'cluster_partitions' not in tables
        assert 'jobs' in tables
        functions = {
            r[0]
            for r in conn.execute(
                text("SELECT proname FROM pg_proc WHERE proname ~ '^(cluster|embedding)_'")
            )
        }
        assert not functions
    engine.dispose()
    migrate.upgrade(pg.settings, super_user=pg.server.super_user)
    assert migrate.current(pg.settings) == HEAD
    migrate.check(pg.settings)


def test_a_second_upgrade_changes_nothing(pg) -> None:
    migrate.upgrade(pg.settings, super_user=pg.server.super_user)
    assert migrate.current(pg.settings) == HEAD
    engine = create_engine(pg.url('super'))
    with engine.connect() as conn:
        assert conn.execute(text('SELECT count(*) FROM alembic_version')).scalar_one() == 1
    engine.dispose()
