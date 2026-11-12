"""``GET /clusters`` says how many items its filter leaves out (test holdout, ignored)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from curation._cluster_cards_helpers import bucket, curation_client
from src.services.curation import cluster_empty_state as ces


if TYPE_CHECKING:
    import pytest


def test_clusters_serve_what_they_hide(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = AsyncMock()
    seen: list[dict[str, Any]] = []

    async def search(*, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        if 'hidden' in (body.get('aggs') or {}):
            seen.append(body)
            return {
                'aggregations': {
                    'hidden': {
                        'buckets': {
                            'test_holdout': {'doc_count': 25},
                            'excluded': {'doc_count': 4},
                        }
                    }
                }
            }
        return {'aggregations': {'clusters': {'buckets': [bucket(1, 5)]}}}

    fake.search = search
    client = curation_client(fake, monkeypatch)
    monkeypatch.setattr(
        'src.routers.curation.clusters.read_partition_state',
        AsyncMock(return_value=ces.partition_state([])),
    )
    body = client.get(
        '/curation/projects/default/clusters',
        params={'per_cluster': 0, 'class_name': 'van'},
    ).json()
    assert body['hidden'] == {'test_holdout': 25, 'excluded': 4}
    assert body['total'] == 1
    assert 'van' in str(seen[0]['query'])
