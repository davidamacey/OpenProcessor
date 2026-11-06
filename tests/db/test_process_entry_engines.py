"""Every non-API process binds its own database engine at its entry point (02b section 8).

Each case runs the real ``main`` of one kind of entry point with ``WEB_CONCURRENCY`` unset, as
in its container, and replaces only the work behind it with a probe that opens a session
through ``async_session_factory``. A one-shot step holds no idle connection (``NullPool``); a
long-running worker or maintenance script holds a pool of two.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import pool, text

from src.db import engine as db_engine
from src.db.session import async_session_factory


if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.db

ENTRY_KINDS = ['migrate_step', 'verify', 'detection_worker', 'vlm_worker', 'assist_loop', 'script']


@pytest.fixture
def probe(pg_env) -> Callable[..., Any]:
    seen: dict[str, Any] = {}

    async def probe(*_args: Any, **_kwargs: Any) -> int:
        async with async_session_factory()() as session:
            assert (await session.execute(text('SELECT 1'))).scalar_one() == 1
        seen['pool'] = db_engine.current_async_engine().sync_engine.pool
        return 0

    probe.seen = seen  # type: ignore[attr-defined]
    return probe


def _run_entry(kind: str, probe: Callable[..., Any], monkeypatch: pytest.MonkeyPatch) -> bool:
    """Run ``kind``'s ``main`` with its work replaced by ``probe``; True for a one-shot entry."""
    if kind == 'migrate_step':
        module = importlib.import_module('scripts.opensearch.migrate')
        monkeypatch.setattr(module, 'run', probe)
        assert module.main([]) == 0
        return True
    if kind == 'verify':
        module = importlib.import_module('scripts.db.verify')

        class Client:
            async def close(self) -> None:
                return None

        class Report:
            def to_json(self) -> dict[str, int]:
                return {'ok': 1}

        async def verify_database(session: Any, *_args: Any) -> Report:
            await probe()
            return Report()

        monkeypatch.setattr(module, 'build_client', Client)
        monkeypatch.setattr(module, 'ControlReader', lambda client: client)
        monkeypatch.setattr(module, 'build_rooted_store', lambda _paths: None)
        monkeypatch.setattr(module, 'storage_root_paths', lambda: None)
        monkeypatch.setattr(module, 'verify_database', verify_database)
        assert module.main([]) == 0
        return True
    if kind == 'detection_worker':
        module = importlib.import_module('scripts.curation.worker.__main__')
        monkeypatch.setattr(module, 'run', probe)
        assert module.main([]) == 0
        return False
    if kind == 'vlm_worker':
        module = importlib.import_module('scripts.curation.vlm_worker')
        monkeypatch.setattr(module, 'run', probe)
        monkeypatch.setattr(module, 'start_worker_metrics_server', lambda _port: None)
        assert module.main([]) == 0
        return False
    if kind == 'assist_loop':
        module = importlib.import_module('scripts.curation.assist_loop_worker')
        monkeypatch.setattr(module, '_main_loop', probe)
        assert module.main([]) == 0
        return False
    module = importlib.import_module('scripts.curation.backfill_scores')
    monkeypatch.setattr(module, 'bind_script_project', lambda _slug: None)
    monkeypatch.setattr(module, '_async_main', probe)
    monkeypatch.setattr(sys, 'argv', ['backfill_scores'])
    assert module.main() == 0
    return False


@pytest.mark.parametrize('kind', ENTRY_KINDS)
def test_an_entry_point_reaches_the_database_without_an_api_worker_count(
    kind: str, probe, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    one_shot = _run_entry(kind, probe, monkeypatch)
    held = probe.seen['pool']
    if one_shot:
        assert isinstance(held, pool.NullPool)
    else:
        assert held.size() == 2
        assert held._max_overflow == 0
    if kind == 'verify':
        assert json.loads(capsys.readouterr().out) == {'ok': 1}
