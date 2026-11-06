"""Fixtures of the 0.6.0 upgrade tests: a clean database, owner sessions and a runner over the fixture."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.services.detection import profile_registry
from src.services.upgrade.from_060.runner import UpgradeRunner
from tests.upgrade._v060 import V060Install, install


if TYPE_CHECKING:
    from pathlib import Path


CLEAR = (
    'TRUNCATE project_slugs, deployment_counters, events CASCADE',
    'DELETE FROM config_documents WHERE project_id IS NULL',
    'DELETE FROM deployment_settings',
    'DELETE FROM upgrade_runs',
)
NOW = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def upgrade_environment(monkeypatch):
    """The environment the upgrade runs in names a VLM and a region detector, so a project that
    activated either built-in can be pointed at its stored body."""
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm-env:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'env-model')
    monkeypatch.setenv('OP_REGION_DETECTION_DETECTOR_MODEL', 'road_detector')
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def clean_db(pg_env):
    """The module's database with no upgrade rows (modules share one database)."""

    def clear() -> None:
        with pg_env.engine('super').begin() as conn:
            for statement in CLEAR:
                conn.execute(text(statement))

    clear()
    return pg_env


@pytest.fixture
def owner_sessions(clean_db):
    """Sessions as the schema owner: the upgrade runs as ``op_owner`` because ``COPY FROM`` is not
    supported on a table with row-level security for a role the policies apply to."""
    engine = create_async_engine(clean_db.url('owner'), poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def v060(tmp_path: Path) -> V060Install:
    return install(tmp_path / 'v060')


@pytest.fixture
def make_runner(owner_sessions, v060: V060Install, tmp_path: Path):
    def make(**overrides) -> UpgradeRunner:
        return UpgradeRunner(
            sessions=owner_sessions,
            reader=overrides.pop('reader', v060.reader),
            writer=overrides.pop('writer', v060.writer),
            layout=v060.layout,
            store=v060.store,
            install_dir=tmp_path / '.install',
            analysis_client=overrides.pop('analysis_client', v060.os),
            now=lambda: NOW,
            **overrides,
        )

    return make
