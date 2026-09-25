"""The VLM worker's pending-id fetch must actually get ids back.

OpenSearch's ``stored_fields: "_none_"`` also suppresses the ``_id``
metadata field, so an "ids only" query written that way returns hits with no
``_id`` and the worker crashed on every fetch. The fake below mirrors that
real behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.curation.vlm_worker import fetch_pending_ids


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _OpenSearchLike:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.bodies: list[dict[str, Any]] = []

    async def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _Resp:  # noqa: ARG002
        self.bodies.append(json)
        drop_metadata = json.get('stored_fields') == '_none_'
        hits = [
            {'_index': 'op_items'} if drop_metadata else {'_index': 'op_items', '_id': i}
            for i in self.ids
        ]
        return _Resp({'hits': {'hits': hits}})


@pytest.mark.asyncio
async def test_fetch_pending_ids_returns_ids_without_loading_source() -> None:
    client = _OpenSearchLike(['a', 'b'])
    ids = await fetch_pending_ids(
        client,  # type: ignore[arg-type]
        opensearch_url='http://os:9200',
        batch_size=2,
        classifier_skip_conf=0.9,
    )
    assert ids == ['a', 'b']
    assert client.bodies[0]['_source'] is False
