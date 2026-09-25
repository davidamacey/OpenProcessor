"""auto_promote's classifier-source gate must not emit a dead
``terms: {class_source: []}`` clause when ``classifier_class_sources()``
is empty in this environment -- that clause sits in ``must`` context, so
an empty list would silently promote zero crops (the query matches
nothing) even though the dry-run summary already counted them as
promotable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

# Import order matters (see auto_promote.py's module docstring): the app
# always imports orchestrator.py first, which imports auto_promote_clusters
# from this module at the bottom of its file -- importing auto_promote in
# isolation before orchestrator fails on the intentional circular import.
import src.services.curation.clustering.orchestrator as _orchestrator  # noqa: F401
from src.services.curation.clustering import auto_promote


def _cluster_agg_response(cluster_id: int, class_name: str, count: int) -> dict[str, Any]:
    # Cluster buckets now come from a composite agg (paged by
    # cluster_id), not a single terms:size=10000 agg — the composite
    # bucket key is a dict of source-name -> value.
    return {
        'aggregations': {
            'clusters': {
                'buckets': [
                    {
                        'key': {'cluster_id': cluster_id},
                        'doc_count': count,
                        'top_class': {'buckets': [{'key': class_name, 'doc_count': count}]},
                    }
                ]
            }
        }
    }


def _empty_scroll_response() -> dict[str, Any]:
    return {'_scroll_id': None, 'hits': {'hits': []}}


@pytest.mark.asyncio
async def test_promote_query_omits_empty_terms_clause_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_promote, 'classifier_class_sources', lambda: set())
    monkeypatch.setattr(auto_promote, '_classifier_sources_empty_warned', False)

    client = AsyncMock()
    responses = [_cluster_agg_response(5, 'van', 10), _empty_scroll_response()]
    client.search = AsyncMock(side_effect=responses)

    await auto_promote.auto_promote_clusters(client, dry_run=False)

    # Second call is _scroll_ids' scroll-open search for the promotable
    # cluster's promote_query.
    scroll_call = client.search.await_args_list[1]
    query = scroll_call.kwargs['body']['query']
    must = query['bool']['filter']
    assert not any('terms' in c and c['terms'].get('class_source') == [] for c in must)
    # No classifier-source terms clause at all when the source set is empty.
    assert not any('terms' in c and 'class_source' in c.get('terms', {}) for c in must)


@pytest.mark.asyncio
async def test_promote_query_keeps_classifier_gate_when_sources_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_promote, 'classifier_class_sources', lambda: {'classifier'})

    client = AsyncMock()
    responses = [_cluster_agg_response(5, 'van', 10), _empty_scroll_response()]
    client.search = AsyncMock(side_effect=responses)

    await auto_promote.auto_promote_clusters(client, dry_run=False)

    scroll_call = client.search.await_args_list[1]
    query = scroll_call.kwargs['body']['query']
    must = query['bool']['filter']
    assert {'terms': {'class_source': ['classifier']}} in must
