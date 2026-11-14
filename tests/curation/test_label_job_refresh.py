"""A label job refreshes the items index once at its end, never once per page; the synchronous
label and undo paths keep ``wait_for`` (a person reads right after the response)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.clients.occ_bulk import BulkWriteError
from src.services.curation import crop_bulk
from src.services.curation.label_ops import record, undo


PAGES = 3


class Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.fail_page_two = False

    def kinds(self) -> list[str]:
        return [e[0] for e in self.events]


class FakeOS:
    def __init__(self, log: Log, *, fail_refresh: bool = False) -> None:
        self.log = log
        self.fail_refresh = fail_refresh
        self.indices = SimpleNamespace(refresh=self._refresh)

    async def _refresh(self, **kw: Any) -> None:
        self.log.events.append(('refresh', kw))
        if self.fail_refresh:
            from opensearchpy.exceptions import OpenSearchException

            raise OpenSearchException('boom')


class Ctx:
    def __init__(self, log: Log, total: int) -> None:
        self.log = log
        self.params = {'op_id': 'op1', 'spec': {}, 'writer': 'w', 'total': total}
        self.resume_from = None
        self.job_id = 'j'

    def cancelled(self) -> bool:
        return False

    def session(self) -> Any:
        return contextlib.nullcontext(object())

    async def checkpoint(self, cursor: Any) -> None:
        self.log.events.append(('checkpoint', cursor))

    async def progress(self, *a: Any, **k: Any) -> None:
        self.log.events.append(('progress', a))


def make_ctx(log: Log, total: int) -> Any:
    return Ctx(log, total)


@pytest.fixture
def relabel_env(monkeypatch: pytest.MonkeyPatch) -> Log:
    log = Log()
    rec = SimpleNamespace(status='writing')

    async def read_op(_op_id: str) -> Any:
        return rec

    async def save_op(r: Any) -> None:
        log.events.append(('complete', r.status))

    async def write_members(*_a: Any, **_k: Any) -> None:
        return None

    async def read_items(_s: Any, _j: Any, *, start: int, limit: int) -> list[str]:
        return [f'c{i}' for i in range(start, min(start + limit, PAGES * OCC_BULK_PAGE_SIZE))]

    async def relabel(_os: Any, page: list[str], _merge: Any, **kw: Any) -> dict[str, Any]:
        log.events.append(('bulk', kw.get('refresh', 'wait_for')))
        if len([e for e in log.events if e[0] == 'bulk']) == 2 and log.fail_page_two:
            return {'updated_ids': [], 'conflicts': [], 'failed': [{'crop_id': 'x'}]}
        return {'updated_ids': page, 'conflicts': [], 'failed': []}

    monkeypatch.setattr(record, 'read_op', read_op)
    monkeypatch.setattr(record, 'save_op', save_op)
    monkeypatch.setattr(record, 'write_members', write_members)
    monkeypatch.setattr(record.job_items, 'read_items', read_items)
    monkeypatch.setattr(record, 'occ_bulk_human_relabel', relabel)
    monkeypatch.setattr(record, 'merger_for', lambda _s, _o: lambda s: s)
    return log


@pytest.mark.asyncio
async def test_label_job_one_refresh_after_last_page_before_complete(relabel_env: Log) -> None:
    log = relabel_env
    await record.run_relabel(make_ctx(log, PAGES * OCC_BULK_PAGE_SIZE), FakeOS(log))
    kinds = log.kinds()
    assert [e[1] for e in log.events if e[0] == 'bulk'] == [False] * PAGES
    assert kinds.count('refresh') == 1
    assert kinds.index('refresh') > max(i for i, k in enumerate(kinds) if k == 'bulk')
    assert kinds.index('refresh') < kinds.index('complete')


@pytest.mark.asyncio
async def test_label_job_failing_on_page_two_still_refreshes_once(relabel_env: Log) -> None:
    log = relabel_env
    log.fail_page_two = True
    with pytest.raises(BulkWriteError):
        await record.run_relabel(make_ctx(log, PAGES * OCC_BULK_PAGE_SIZE), FakeOS(log))
    assert log.kinds().count('refresh') == 1
    assert 'complete' not in log.kinds()


@pytest.mark.asyncio
async def test_a_failed_refresh_does_not_fail_the_job(relabel_env: Log) -> None:
    log = relabel_env
    out = await record.run_relabel(
        make_ctx(log, OCC_BULK_PAGE_SIZE),
        FakeOS(log, fail_refresh=True),
    )
    assert out['complete'] is True


@pytest.mark.asyncio
async def test_human_relabel_default_is_wait_for_and_refresh_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    async def fake_bulk(_c: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw['refresh'])
        return {'a': 'updated'}

    async def no_excludes() -> list[str]:
        return []

    monkeypatch.setattr(crop_bulk, 'occ_update_bulk', fake_bulk)
    monkeypatch.setattr(crop_bulk, 'vector_source_excludes', no_excludes)
    monkeypatch.setattr(crop_bulk, 'items_index', lambda: 'items')
    await crop_bulk.occ_bulk_human_relabel(None, ['a'], lambda s: s, writer_id='w')
    await crop_bulk.occ_bulk_human_relabel(None, ['a'], lambda s: s, writer_id='w', refresh=False)
    assert seen == ['wait_for', False]


@pytest.fixture
def undo_env(monkeypatch: pytest.MonkeyPatch) -> Log:
    log = Log()
    rec = SimpleNamespace(status='written', undo_passes=[], op_id='op1')

    async def read_op(_i: str) -> Any:
        return rec

    async def save_op(r: Any) -> None:
        log.events.append(('complete', r.status))

    async def no_check(*_a: Any) -> None:
        return None

    async def member_pages(_r: Any) -> Any:
        for _ in range(PAGES):
            yield []

    async def rounds(_os: Any, **kw: Any) -> None:
        log.events.append(('bulk', kw['refresh']))

    monkeypatch.setattr(undo, 'read_op', read_op)
    monkeypatch.setattr(undo, 'save_op', save_op)
    monkeypatch.setattr(undo, '_refuse_if_nothing_left', no_check)
    monkeypatch.setattr(undo, 'member_pages', member_pages)
    monkeypatch.setattr(undo, 'occ_bulk_rounds', rounds)
    monkeypatch.setattr(undo, 'items_index', lambda: 'items')
    return log


@pytest.mark.asyncio
async def test_undo_job_one_refresh_after_last_page_before_complete(undo_env: Log) -> None:
    log = undo_env
    await undo.restore_op(FakeOS(log), 'op1', lambda _i: None, ctx=make_ctx(log, 0))
    kinds = log.kinds()
    assert [e[1] for e in log.events if e[0] == 'bulk'] == [False] * PAGES
    assert kinds.count('refresh') == 1
    assert kinds.index('refresh') > max(i for i, k in enumerate(kinds) if k == 'bulk')
    assert kinds.index('refresh') < len(kinds) - 1 - kinds[::-1].index(
        'complete'
    )  # the 'undone' save


@pytest.mark.asyncio
async def test_undo_without_a_job_keeps_wait_for(undo_env: Log) -> None:
    log = undo_env
    await undo.restore_op(FakeOS(log), 'op1', lambda _i: None)
    assert [e[1] for e in log.events if e[0] == 'bulk'] == ['wait_for'] * PAGES
    assert 'refresh' not in log.kinds()
