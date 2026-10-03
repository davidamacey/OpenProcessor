"""The wheel world's audit log is what "a sibling project was never touched"
rests on, so every cluster-changing call must show up in it."""

from __future__ import annotations

import pytest

from integration.wheel_world import WRITE_OPS, RoutedOpenSearch


DEFAULT_ITEMS = 'op_prj_default__items'


@pytest.mark.asyncio
async def test_a_settings_write_is_logged_as_a_write() -> None:
    cluster = RoutedOpenSearch()
    cluster.data.docs(DEFAULT_ITEMS)
    await cluster.indices.put_settings(
        index=DEFAULT_ITEMS, body={'index': {'refresh_interval': '1s'}}
    )
    assert ('put_settings', DEFAULT_ITEMS) in cluster.audit
    assert DEFAULT_ITEMS in cluster.touched(writes_only=True)


@pytest.mark.asyncio
async def test_a_mapping_write_is_logged_as_a_write() -> None:
    cluster = RoutedOpenSearch()
    await cluster.indices.put_mapping(index=DEFAULT_ITEMS)
    assert DEFAULT_ITEMS in cluster.touched(writes_only=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('call', ['update_by_query', 'delete_by_query'])
async def test_a_by_query_write_is_logged_before_it_fails(call: str) -> None:
    cluster = RoutedOpenSearch()
    with pytest.raises(NotImplementedError):
        await getattr(cluster, call)(index=DEFAULT_ITEMS, body={})
    assert (call, DEFAULT_ITEMS) in cluster.audit
    assert call in WRITE_OPS
