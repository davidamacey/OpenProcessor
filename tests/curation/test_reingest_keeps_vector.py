"""A re-ingest must not contradict the vector already stored on an item.

An update without ``pe_embedding`` leaves the stored vector in place, so an
encoder failure on the re-run must not flip ``embedding_state`` to ``failed``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.occ_fakes import make_bulk_update_item, make_mget_response
from src.clients.occ import occ_upsert_bulk
from src.services.curation.embedding_state import keep_stored_vector_state


def test_failed_rerun_keeps_embedded_state() -> None:
    merged = {'crop_id': 'c', 'embedding_state': 'failed'}
    keep_stored_vector_state(merged, {'embedding_state': 'embedded'})
    assert 'embedding_state' not in merged


def test_failed_rerun_over_failed_or_unknown_is_written() -> None:
    for existing in ({'embedding_state': 'failed'}, {}):
        merged = {'embedding_state': 'failed'}
        keep_stored_vector_state(merged, existing)
        assert merged['embedding_state'] == 'failed'


def test_new_vector_always_restates_embedded() -> None:
    merged = {'pe_embedding': [0.1], 'embedding_state': 'embedded'}
    keep_stored_vector_state(merged, {'embedding_state': 'failed'})
    assert merged['embedding_state'] == 'embedded'


@pytest.mark.asyncio
async def test_upsert_bulk_does_not_write_failed_over_an_embedded_item() -> None:
    client = AsyncMock()
    client.mget = AsyncMock(
        return_value=make_mget_response({'c1': {'crop_id': 'c1', 'embedding_state': 'embedded'}})
    )
    sent: list[Any] = []

    async def _bulk(body: list[Any], **_: Any) -> dict[str, Any]:
        sent.extend(body)
        return {'items': [make_bulk_update_item('c1')]}

    client.bulk = _bulk
    await occ_upsert_bulk(
        client,
        [{'crop_id': 'c1', 'embedding_state': 'failed', 'confidence': 0.5}],
        human_field_guards=[],
    )
    updates = [x['doc'] for x in sent if isinstance(x, dict) and 'doc' in x]
    assert updates
    assert all('embedding_state' not in u for u in updates)
