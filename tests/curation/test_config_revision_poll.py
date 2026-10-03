"""Polling a project whose ``meta:config_revision`` doc is absent is quiet
and cheap; a project that has the doc still sees revision bumps."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from opensearchpy.exceptions import NotFoundError

from src.services.config_store import index as config_index
from src.services.config_store.index import bump_config_revision, get_config_revision


class _Client:
    """Raises NotFound like a real miss, but logs it as the transport would
    for a request that was not told to ignore 404."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.gets = 0
        self.ignore_seen: list[Any] = []

    async def get(self, index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        self.gets += 1
        self.ignore_seen.append(ignore)
        if id in self.docs:
            return {'_source': dict(self.docs[id])}
        if ignore != 404:
            logging.getLogger('opensearch').warning('GET %s/_doc/%s [status:404]', index, id)
            raise NotFoundError(404, 'nf', {})
        return {'found': False}

    async def update(self, index: str, id: str, body: dict[str, Any], **_: Any) -> None:  # noqa: A002
        del index
        if id in self.docs:
            self.docs[id]['config_revision'] += 1
        else:
            self.docs[id] = dict(body['upsert'])


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(config_index, '_monotonic', lambda: now[0])
    monkeypatch.setenv('OP_CONFIG_POLL_S', '5')
    config_index._absent_until.clear()
    return now


@pytest.mark.asyncio
async def test_absent_revision_doc_is_quiet_and_one_request_per_interval(
    clock: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    client = _Client()
    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            assert await get_config_revision(client, 'idx') == 0
        assert client.gets == 1
        clock[0] += 5.1
        assert await get_config_revision(client, 'idx') == 0
    assert client.gets == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert set(client.ignore_seen) == {404}


@pytest.mark.asyncio
async def test_project_with_doc_observes_bumps(clock: list[float]) -> None:
    client = _Client()
    assert await get_config_revision(client, 'idx') == 0
    assert await bump_config_revision(client, 'idx') == 1
    assert await get_config_revision(client, 'idx') == 1
    assert await bump_config_revision(client, 'idx') == 2
    assert await get_config_revision(client, 'idx') == 2
