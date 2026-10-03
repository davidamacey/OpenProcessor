"""A replaced worker container leaves a ``runtime:`` doc behind (hostname =
container id); it must age out of ``applied[]`` instead of showing as a
permanently lagging host."""

from __future__ import annotations

import asyncio
import datetime
from unittest.mock import AsyncMock

from src.services.config_store.index import RUNTIME_DOC_MAX_AGE_S, get_runtime_docs


def _hit(host: str, age_s: float) -> dict[str, object]:
    at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=age_s)
    return {
        '_id': f'runtime:detection_worker:{host}',
        '_source': {'host': host, 'applied_at': at.isoformat()},
    }


def test_stale_runtime_docs_are_dropped() -> None:
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    _hit('live', 30),
                    _hit('replaced', RUNTIME_DOC_MAX_AGE_S + 600),
                ]
            }
        }
    )
    docs = asyncio.run(get_runtime_docs(client, 'idx', process='detection_worker'))
    assert [d['host'] for d in docs] == ['live']
