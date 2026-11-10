"""A refused settings call leaves the index open and the upgrade stopped naming it (2F-6,
review fix RR-8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.upgrade.from_060 import analysis_settings
from src.services.upgrade.from_060.analysis_settings import AnalysisSettingsError
from tests.upgrade._v060 import install


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_a_refused_settings_call_leaves_the_index_open_and_names_it(tmp_path: Path) -> None:
    world = install(tmp_path / 'v060')
    names = sorted(n for n in world.os.indexes if n.endswith('__items'))
    world.os.refuse_settings = True

    with pytest.raises(AnalysisSettingsError) as err:
        await analysis_settings.apply(world.os, names)

    assert err.value.index == names[0]
    assert names[0] in str(err.value)
    assert world.os.closed == set()
    assert world.os.calls.count('indices.open') == 1
    assert 'analysis' not in world.os.indexes[names[0]]['settings'].get('index', {})
    assert world.os.calls.count('indices.close') == 1  # the next index was never touched


@pytest.mark.asyncio
async def test_a_normalizer_that_analyzes_wrongly_stops_the_upgrade(tmp_path: Path) -> None:
    world = install(tmp_path / 'v060')
    names = sorted(n for n in world.os.indexes if n.endswith('__items'))

    async def wrong(index: str, body: dict) -> dict:
        return {'tokens': [{'token': 'ab-1 cé 9'}]}

    world.os.indices.analyze = wrong  # type: ignore[method-assign]

    with pytest.raises(AnalysisSettingsError, match='expected'):
        await analysis_settings.apply(world.os, names)


@pytest.mark.asyncio
async def test_reopen_health_wait_is_accepted_by_the_real_client(monkeypatch) -> None:
    """opensearch-py reads a ``timeout`` argument as the transport timeout in seconds, so a
    duration string such as ``'120s'`` crashes the HTTP layer; the fakes of the other tests
    accept anything and cannot see that."""
    import json

    from opensearchpy import AIOHttpConnection, AsyncOpenSearch

    seen: list[tuple[str, object]] = []

    async def perform_request(_self, method, url, params=None, body=None, timeout=None, **_):
        assert timeout is None or isinstance(timeout, (int, float)), timeout
        seen.append((url, timeout))
        payload = {'status': 'yellow', 'timed_out': False, 'acknowledged': True}
        return 200, {'content-type': 'application/json'}, json.dumps(payload)

    monkeypatch.setattr(AIOHttpConnection, 'perform_request', perform_request)
    client = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    try:
        assert await analysis_settings._reopen(client, 'op_prj_x__items') == 'yellow'
    finally:
        await client.close()
    assert any('_cluster/health' in url for url, _ in seen)
