"""Read-your-writes, by caller. A batch ingest refreshes nothing (a bulk run reads none of its
writes by search; a photo is searchable within the refresh interval). The callers that promise it
refresh once before they answer: ``/ingest/image`` (in the route), ``/ingest/upload`` and
``/ingest/objects`` (in ``stage_and_ingest``). A refresh failure never fails an ingest. The
open-vocabulary stamps are written in one bulk with ``refresh=wait_for``, so the drain's search
sees every stamped image."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.services.curation import open_vocab_ingest, upload_ingest


if TYPE_CHECKING:
    from pathlib import Path


class _Indices:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def refresh(self, index: str, **_kw: Any) -> dict[str, Any]:
        self.calls.append(index)
        if self._fail:
            raise OpenSearchConnectionError('N/A', 'refresh unavailable', OSError('refused'))
        return {}


def _upload_world(tmp_path: Path, *, fail: bool = False) -> tuple[Any, Any, Any]:
    svc, os_fake, _ = _make_service()
    svc.config = dataclasses.replace(
        svc.config, upload_root=tmp_path / 'uploads', upload_accepted_extensions=('.jpg',)
    )
    os_fake.indices = _Indices(fail=fail)
    return svc, os_fake, [upload_ingest.StagedEntry('a', _jpeg_bytes(seed=1), '.jpg')]


@pytest.mark.asyncio
async def test_a_batch_ingest_refreshes_nothing() -> None:
    svc, os_fake, _ = _make_service()
    os_fake.indices = _Indices()
    batch = await svc.ingest_batch(
        [_jpeg_bytes(seed=1), _jpeg_bytes(seed=2), _jpeg_bytes(seed=3)],
        ['/tmp/a.jpg', '/tmp/b.jpg', '/tmp/c.jpg'],
    )
    assert batch.summary.successful == 3
    assert os_fake.indices.calls == []


@pytest.mark.asyncio
async def test_an_upload_refreshes_once_covering_both_indexes(tmp_path: Path) -> None:
    svc, os_fake, entries = _upload_world(tmp_path)
    result, early = await upload_ingest.stage_and_ingest(
        os_fake, svc, entries, config=svc.config, source='upload', run_id=None
    )
    assert result is not None
    assert not early
    assert len(os_fake.indices.calls) == 1
    assert set(os_fake.indices.calls[0].split(',')) == {
        svc.config.images_index,
        svc.config.items_index,
    }


@pytest.mark.asyncio
async def test_a_failing_refresh_does_not_fail_the_upload(tmp_path: Path) -> None:
    svc, os_fake, entries = _upload_world(tmp_path, fail=True)
    result, _early = await upload_ingest.stage_and_ingest(
        os_fake, svc, entries, config=svc.config, source='upload', run_id=None
    )
    assert result is not None
    assert result.status == 'success'
    assert len(os_fake.indices.calls) == 1


@pytest.mark.asyncio
async def test_the_single_image_route_refreshes_before_it_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import ingest as route

    svc, os_fake, _ = _make_service()
    os_fake.indices = _Indices()

    async def service(_opensearch: Any, _registry: Any) -> Any:
        return svc

    async def read(_paths: list[str]) -> list[bytes]:
        return [_jpeg_bytes(seed=9)]

    async def schedule(_opensearch: Any, _ids: list[str]) -> bool:
        return False

    monkeypatch.setattr(route, 'get_ingest_service', service)
    monkeypatch.setattr(route, 'read_files', read)
    monkeypatch.setattr(route, 'is_servable_image_path', lambda _p: True)
    monkeypatch.setattr(route, 'schedule_open_vocab_after_ingest', schedule)
    answer = await route.curation_ingest_image(
        route.IngestImageRequest(path='/tmp/one.jpg'), os_fake, cast('Any', SimpleNamespace())
    )
    assert answer.status == 'success'
    assert len(os_fake.indices.calls) == 1


class _Bulk:
    def __init__(self, order: list[str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._order = order

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        self._order.append('bulk')
        self.calls.append({'body': body, 'refresh': refresh})
        return {
            'errors': False,
            'items': [{'update': {'status': 200}} for _ in body[::2]],
        }

    async def update(self, **_kw: Any) -> None:
        raise AssertionError('one update per image: the stamps go in one bulk')


@pytest.mark.asyncio
async def test_the_open_vocab_stamps_are_one_bulk_visible_before_the_drain_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    fake = _Bulk(order)

    async def active() -> Any:
        return (SimpleNamespace(run_on_ingest=True), 1)

    async def start() -> bool:
        order.append('drain')
        return True

    monkeypatch.setattr(open_vocab_ingest, 'current_active_set', active)
    monkeypatch.setattr(open_vocab_ingest, 'start_drain', start)
    assert await open_vocab_ingest.schedule_open_vocab_after_ingest(
        cast('Any', fake), ['i1', 'i2', 'i3']
    )
    [call] = fake.calls
    assert call['refresh'] == 'wait_for'
    assert [a['update']['_id'] for a in call['body'][::2]] == ['i1', 'i2', 'i3']
    assert all(d['doc']['open_vocab_status'] == 'pending' for d in call['body'][1::2])
    assert order == ['bulk', 'drain']
