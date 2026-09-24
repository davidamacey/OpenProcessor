"""F-10: force_cluster_id_equals_class_id must not visit every doc with a
class_id.

Before the fix, the ``update_by_query`` filter only required
``exists: class_id`` — at scale this matches nearly every doc (234,575 on
the legacy index) and, worse, it pulled unconfident/residual crops that
carry a low-confidence ``class_id`` but sit in a candidate cluster
(``cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET``) back out of that candidate
cluster on every run, undoing the residual pool's clustering work.

The fix narrows the query to confident/validated docs only (the same
``CONFIDENT_CLASS_SOURCES`` set the residual-pool builder excludes), plus a
doc-values script filter so OpenSearch's own query-time engine skips
already-correct docs instead of loading + reconstructing the full
``_source`` (including derived kNN vectors) just for the script to noop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.curation.clustering.embedding_reduce import CONFIDENT_CLASS_SOURCES
from src.services.curation.clustering.id_normalize import force_cluster_id_equals_class_id


def _client() -> AsyncMock:
    client = AsyncMock()
    client.update_by_query = AsyncMock(return_value={'task': 'node1:1'})
    client.tasks = AsyncMock()
    client.tasks.get = AsyncMock(
        return_value={'completed': True, 'response': {'updated': 0, 'batches': 1}}
    )
    return client


@pytest.mark.asyncio
async def test_query_still_requires_class_id_exists():
    client = _client()
    await force_cluster_id_equals_class_id(client)
    _, kwargs = client.update_by_query.call_args
    filters = kwargs['body']['query']['bool']['filter']
    assert {'exists': {'field': 'class_id'}} in filters


@pytest.mark.asyncio
async def test_query_restricts_to_confident_or_validated_docs():
    """The should/minimum_should_match clause is the narrowing fix."""
    client = _client()
    await force_cluster_id_equals_class_id(client)
    _, kwargs = client.update_by_query.call_args
    filters = kwargs['body']['query']['bool']['filter']
    should_clauses = [f for f in filters if 'bool' in f and 'should' in f.get('bool', {})]
    assert len(should_clauses) == 1
    should = should_clauses[0]['bool']
    assert should['minimum_should_match'] == 1
    assert {'term': {'class_validated': True}} in should['should']
    terms_clause = next(c for c in should['should'] if 'terms' in c)
    assert terms_clause == {'terms': {'class_source': sorted(CONFIDENT_CLASS_SOURCES)}}


@pytest.mark.asyncio
async def test_query_has_doc_values_script_filter_for_already_correct_docs():
    """Belt-and-suspenders: a script filter so already-matching docs are
    excluded at query time (doc-values, no _source load), not just noop'd
    inside the update script after a full _source reconstruction."""
    client = _client()
    await force_cluster_id_equals_class_id(client)
    _, kwargs = client.update_by_query.call_args
    filters = kwargs['body']['query']['bool']['filter']
    script_filters = [f for f in filters if 'script' in f]
    assert len(script_filters) == 1
    src = script_filters[0]['script']['script']['source']
    assert "doc['cluster_id'].size()==0" in src
    assert "doc['cluster_id'].value != doc['class_id'].value" in src


@pytest.mark.asyncio
async def test_update_script_still_has_noop_guard_as_second_layer():
    client = _client()
    await force_cluster_id_equals_class_id(client)
    _, kwargs = client.update_by_query.call_args
    script_src = kwargs['body']['script']['source']
    assert "ctx.op = 'noop'" in script_src
