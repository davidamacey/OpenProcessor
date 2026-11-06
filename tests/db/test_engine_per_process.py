"""Engines are lazy and per process: importing opens nothing and a forked child gets its own."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from src.db import engine as db_engine


pytestmark = pytest.mark.db

REPO_ROOT = Path(__file__).resolve().parents[2]

_IMPORT_PROBE = """
import json, socket, sys
sys.path.insert(0, {root!r})
def refuse(*a, **k):
    raise AssertionError('a socket was opened at import')
socket.socket.connect = refuse
socket.create_connection = refuse
import src.db, src.db.models, src.db.engine, src.db.session
print(json.dumps({{'engines': len(src.db.engine._engines)}}))
"""


def test_importing_src_db_opens_no_socket_and_builds_no_engine() -> None:
    out = subprocess.run(  # fixed argv, our own interpreter
        [sys.executable, '-c', _IMPORT_PROBE.format(root=str(REPO_ROOT))],
        capture_output=True,
        text=True,
        check=True,
        cwd='/',
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) == {'engines': 0}


def test_an_engine_is_cached_per_process_and_kind(pg_env) -> None:
    assert db_engine.get_sync_engine() is db_engine.get_sync_engine()
    assert db_engine.get_async_engine() is db_engine.get_async_engine()
    assert db_engine.get_poll_engine() is not db_engine.get_sync_engine()


def _child(parent_engine_id: int, conn) -> None:
    inherited = list(db_engine._engines.values())
    db_engine.dispose_inherited_engines()
    registry_after_dispose = len(db_engine._engines)
    fresh = db_engine.get_sync_engine()
    with fresh.connect() as c:
        answer = c.execute(text('SELECT 1')).scalar_one()
    conn.send(
        {
            'inherited_before': len(inherited),
            'registry_after_dispose': registry_after_dispose,
            'same_object': id(fresh) == parent_engine_id,
            'answer': answer,
        }
    )


def test_a_forked_child_gets_its_own_engine_and_leaves_the_parents_connection_alone(
    pg_env,
) -> None:
    parent = db_engine.get_sync_engine()
    held = parent.connect()
    held.execute(text('SELECT 1'))
    ctx = multiprocessing.get_context('fork')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(id(parent), sender))
    process.start()
    report = receiver.recv()
    process.join(30)
    assert process.exitcode == 0
    assert report['inherited_before'] == 1
    assert report['registry_after_dispose'] == 0
    assert report['same_object'] is False
    assert report['answer'] == 1
    # The child disposed with close=False: the parent's socket is still usable.
    assert held.execute(text('SELECT 2')).scalar_one() == 2
    held.close()


def test_the_api_pool_follows_the_plan(pg_env) -> None:
    pool = db_engine.get_async_engine().sync_engine.pool
    assert pool.size() == 32  # 128 connections // (4 workers x 1 replica)
    assert pool.timeout() == 5
    assert pool._max_overflow == 0
    assert pool._pre_ping is True
    assert pool._pool.use_lifo is True
    assert pool._recycle == 1800


def test_the_poll_and_child_engines_are_small(pg_env) -> None:
    assert db_engine.get_poll_engine().pool.size() == 1
    assert db_engine.get_sync_engine().pool.size() == 2


def test_the_task_engine_does_not_pool(pg_env) -> None:
    engine = db_engine.new_task_async_engine()
    try:
        assert type(engine.sync_engine.pool).__name__ == 'NullPool'
    finally:
        engine.sync_engine.dispose()


def test_a_pgbouncer_deployment_turns_prepared_statements_off(pg_env) -> None:
    from dataclasses import replace

    from src.config.database import DatabaseSettings

    settings = DatabaseSettings.from_env()
    assert db_engine._connect_args(settings, job_limits=False) == {}
    pooled = replace(settings, pooler='pgbouncer')
    assert db_engine._connect_args(pooled, job_limits=False) == {'prepare_threshold': None}


def test_one_off_connection_uses_the_direct_host_and_no_pool(pg_env) -> None:
    with db_engine.one_off_connection('app') as conn:
        assert conn.execute(text('SELECT current_user')).scalar_one() == 'op_app'
