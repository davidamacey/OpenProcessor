"""``load_global_fields`` reads every probe doc (paged) and tolerates a doc
with no ``probe_key``."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import NotFoundError

from src.services.config_store import vlm_snapshot


class _PagedClient:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.calls: list[dict[str, Any]] = []

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append(body)
        start = 0 if 'search_after' not in body else body['search_after'][0] + 1
        page = self.docs[start : start + body['size']]
        return {
            'hits': {
                'hits': [
                    {'_id': f'd{start + i}', '_source': d, 'sort': [start + i]}
                    for i, d in enumerate(page)
                ]
            }
        }

    async def get(self, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        raise NotFoundError(404, 'not found', {})


def _probe(i: int) -> dict[str, Any]:
    return {'doc_type': 'vlm_probe', 'probe_key': f'ep{i}@1', 'body': {'n': i}}


@pytest.mark.asyncio
async def test_probes_beyond_one_page_are_all_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_snapshot, '_PROBE_PAGE_SIZE', 3)
    client = _PagedClient([_probe(i) for i in range(8)])
    got = await vlm_snapshot.load_global_fields(client, 'idx')
    assert sorted(got['vlm_probes']) == sorted(f'ep{i}@1' for i in range(8))


@pytest.mark.asyncio
async def test_a_probe_doc_without_a_key_is_skipped_not_fatal() -> None:
    client = _PagedClient([_probe(0), {'doc_type': 'vlm_probe', 'body': {}}, _probe(2)])
    got = await vlm_snapshot.load_global_fields(client, 'idx')
    assert sorted(got['vlm_probes']) == ['ep0@1', 'ep2@1']
