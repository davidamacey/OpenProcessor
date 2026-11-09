"""Probes and index creators answer False only for the transport/server errors
the libraries raise; a programming error propagates instead of reading as an outage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import ConnectionError as OsConnectionError
from tritonclient.utils import InferenceServerException

from src.clients.curation_opensearch.lifecycle import _create_one
from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.indexes import IndexesMixin
from src.clients.triton_pool_core import AsyncTritonPool


@pytest.mark.asyncio
async def test_ping_transport_error_is_false_bug_propagates() -> None:
    c = OpenSearchClient(hosts=['http://localhost:9200'])
    c.client = MagicMock()
    c.client.ping = AsyncMock(side_effect=OsConnectionError('N/A', 'refused', Exception('x')))
    assert await c.ping() is False
    c.client.ping = AsyncMock(side_effect=AttributeError('bug'))
    with pytest.raises(AttributeError):
        await c.ping()


def _pool() -> AsyncTritonPool:
    p = AsyncTritonPool()
    p._initialized = True
    p._clients = [MagicMock()]
    return p


@pytest.mark.asyncio
async def test_pool_probes_server_error_false_bug_propagates() -> None:
    p = _pool()
    p._clients[0].is_server_live = AsyncMock(side_effect=InferenceServerException('down'))
    p._clients[0].is_model_ready = AsyncMock(side_effect=InferenceServerException('down'))
    assert await p.health_check() is False
    assert await p.is_model_ready('m') is False
    p._clients[0].is_server_live = AsyncMock(side_effect=KeyError('bug'))
    p._clients[0].is_model_ready = AsyncMock(side_effect=KeyError('bug'))
    with pytest.raises(KeyError):
        await p.health_check()
    with pytest.raises(KeyError):
        await p.is_model_ready('m')


@pytest.mark.asyncio
async def test_create_one_bug_propagates() -> None:
    client = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock(side_effect=OsConnectionError('N/A', 'refused', Exception()))
    assert await _create_one(client, 'i', {}, False) is False
    client.indices.create = AsyncMock(side_effect=TypeError('bug'))
    with pytest.raises(TypeError):
        await _create_one(client, 'i', {}, False)


@pytest.mark.asyncio
async def test_create_index_bug_propagates() -> None:
    mgr = IndexesMixin.__new__(IndexesMixin)
    mgr.client = MagicMock()
    mgr.client.indices.exists = AsyncMock(return_value=False)
    mgr.client.indices.create = AsyncMock(side_effect=TypeError('bug'))
    with pytest.raises(TypeError):
        await mgr._create_index('i', {})
