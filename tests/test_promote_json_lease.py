"""Two processes writing one model's promote.json: the lease serialises the read-compare-write.

Spawned processes share a real model dir and a manager-backed lease store (test scaffolding
for the server's per-document atomicity).
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch


def _child(action: str, path: str, store: Any, lock: Any, barrier: Any, out: Any) -> None:
    from conftest import DEFAULT_RECORD

    from src.config.project_context import bind_process_project
    from src.services.training import promote_json

    bind_process_project(DEFAULT_RECORD)  # type: ignore[arg-type]
    promote_json.new_lease_client = lambda: FakeLeaseOpenSearch(store, lock)  # type: ignore[assignment]
    real = promote_json.sharing_revision

    def slow(raw: dict[str, Any]) -> int:
        time.sleep(0.5)  # a slow writer between its read and its write
        return real(raw)

    promote_json.sharing_revision = slow  # type: ignore[assignment]

    async def run() -> str:
        target = Path(path)
        barrier.wait(timeout=60)
        try:
            if action == 'share':
                await promote_json.update_sharing(target, shared=True, expected_revision=1)
            else:
                await promote_json.write_promote_backpointer(target, {'version': '2'})
        except promote_json.SharingRevisionConflictError:
            return 'conflict'
        except promote_json.PromoteJsonBusyError:
            return 'busy'
        return 'ok'

    out.put((action, asyncio.run(run())))


def _run(tmp_path: Path, actions: list[str]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    model = tmp_path / 'models' / 'm1'
    model.mkdir(parents=True)
    path = model / 'promote.json'
    path.write_text(json.dumps({'version': '1', 'shared': False}))
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        store, lock = manager.dict(), ctx.Lock()
        barrier, out = ctx.Barrier(len(actions)), ctx.Queue()
        procs = [
            ctx.Process(target=_child, args=(a, str(path), store, lock, barrier, out))
            for a in actions
        ]
        for p in procs:
            p.start()
        results = [out.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=30)
    return json.loads(path.read_text()), results


@pytest.mark.real_lease_io
def test_two_sharing_writes_with_the_same_revision_give_one_success_and_one_conflict(
    tmp_path: Path,
) -> None:
    doc, results = _run(tmp_path, ['share', 'share'])

    assert sorted(r for _, r in results) == ['conflict', 'ok']
    assert doc['sharing_revision'] == 2


@pytest.mark.real_lease_io
def test_a_repromote_racing_a_sharing_write_keeps_the_new_shared_value(tmp_path: Path) -> None:
    doc, results = _run(tmp_path, ['share', 'repromote'])

    shared_ok = ('share', 'ok') in results
    # A sharing write that lost the lease answers its revision conflict (the caller
    # refreshes); one that won is never undone by the re-promote.
    assert doc['shared'] is shared_ok
    assert doc.get('sharing_revision', 1) == (2 if shared_ok else 1)


@pytest.mark.asyncio
async def test_control_a_single_sharing_write_bumps_the_revision_by_one(tmp_path: Path) -> None:
    from src.services.training import promote_json

    path = tmp_path / 'm1' / 'promote.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'version': '1'}))

    assert await promote_json.update_sharing(path, shared=True, expected_revision=1) == 2
    assert json.loads(path.read_text())['shared'] is True
