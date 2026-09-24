"""S4: raw VLM label clustering — pure clustering logic plus the end-to-end
contract with ``GET /review/raw_label_clusters`` (the script must write
exactly the fields that endpoint aggregates on)."""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import pytest

from src.config import get_curation_config
from src.services.curation import raw_label_clusters as rlc


# =============================================================================
# In-memory OpenSearch covering the script's and the endpoint's query shapes
# =============================================================================


def _matches(doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
    if not query:
        return True
    if 'bool' in query:
        return all(_matches(doc, q) for q in query['bool'].get('must', []))
    if 'exists' in query:
        return doc.get(query['exists']['field']) is not None
    if 'term' in query:
        [(field, value)] = query['term'].items()
        return doc.get(field) == value
    raise AssertionError(f'unsupported query {query}')


def _agg(docs: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if 'filter' in spec:
        return {'doc_count': sum(1 for d in docs if _matches(d, spec['filter']))}
    terms = spec['terms']
    groups: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in docs:
        if d.get(terms['field']) is not None:
            groups[d[terms['field']]].append(d)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    buckets = []
    for key, members in ordered[: terms.get('size', 10)]:
        bucket: dict[str, Any] = {'key': key, 'doc_count': len(members)}
        for name, sub in (spec.get('aggs') or {}).items():
            bucket[name] = _agg(members, sub)
        buckets.append(bucket)
    return {'buckets': buckets}


class FakeItemsOpenSearch:
    def __init__(self, items: dict[str, dict[str, Any]]) -> None:
        self.items = items
        self.bulk_bodies: list[list[dict[str, Any]]] = []
        self.mapping_puts: list[dict[str, Any]] = []
        self.indices = self._Indices(self)

    class _Indices:
        def __init__(self, outer: FakeItemsOpenSearch) -> None:
            self._outer = outer

        async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
            self._outer.mapping_puts.append(body)
            return {'acknowledged': True}

        async def exists(self, index: str) -> bool:  # noqa: ARG002
            return True

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        docs = [d for d in self.items.values() if _matches(d, body.get('query'))]
        aggs = body.get('aggs')
        if aggs:
            return {
                'hits': {'hits': [], 'total': {'value': len(docs)}},
                'aggregations': {name: _agg(docs, spec) for name, spec in aggs.items()},
            }
        docs.sort(key=lambda d: d['crop_id'])
        after = body.get('search_after')
        if after is not None:
            docs = [d for d in docs if d['crop_id'] > after[0]]
        page = docs[: body.get('size', 10)]
        return {
            'hits': {
                'hits': [{'_id': d['crop_id'], '_source': d, 'sort': [d['crop_id']]} for d in page]
            }
        }

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:  # noqa: ARG002
        self.bulk_bodies.append(body)
        out = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            self.items[doc_id].update(doc['doc'])
            out.append({'update': {'_id': doc_id, 'status': 200}})
        return {'errors': False, 'items': out}


def _item(crop_id: str, raw: str | None, *, unmatched: bool, class_name: str | None = None):
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'class_source': rlc.UNMATCHED_CLASS_SOURCE if unmatched else 'vlm_matched',
        'class_name': class_name,
    }
    if raw is not None:
        doc[rlc.RAW_LABEL_FIELD] = raw
    return doc


def _corpus() -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = 0

    def add(raw: str | None, count: int, *, unmatched: bool, class_name: str | None = None):
        nonlocal n
        for _ in range(count):
            rows.append(_item(f'c{n:04d}', raw, unmatched=unmatched, class_name=class_name))
            n += 1

    add('pickup truck', 6, unmatched=True)
    add('pick-up truck', 4, unmatched=True)
    add('pickup-truck', 3, unmatched=False, class_name='truck')
    add('forklift', 5, unmatched=True)
    add('fork lift', 3, unmatched=True)
    add('zebra crossing', 1, unmatched=True)  # below min_count -> singleton
    add(None, 4, unmatched=False)  # no raw label: must be left untouched
    return {r['crop_id']: r for r in rows}


# =============================================================================
# Pure logic
# =============================================================================


class TestClusterTerms:
    def test_stable_id_is_positive_int32_and_deterministic(self) -> None:
        a = rlc.stable_cluster_id('pickup truck')
        assert a == rlc.stable_cluster_id('pickup truck')
        assert 0 <= a <= 0x7FFFFFFF
        assert a != rlc.stable_cluster_id('forklift')

    def test_hash_embed_groups_spelling_variants(self) -> None:
        v = rlc.hash_embed(['pickup truck', 'pick-up truck', 'forklift'])
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)

    def test_clusters_named_after_most_frequent_member(self) -> None:
        terms = ['pickup truck', 'pick-up truck', 'forklift', 'fork lift', 'zebra crossing']
        counts = [6, 4, 5, 3, 1]
        out = rlc.cluster_terms(terms, counts, rlc.hash_embed, distance_threshold=0.6)
        assert out['pick-up truck'].cluster_name == 'pickup truck'
        assert out['pick-up truck'].cluster_id == out['pickup truck'].cluster_id
        assert out['fork lift'].cluster_name == 'forklift'
        assert out['forklift'].cluster_id != out['pickup truck'].cluster_id
        # Below min_count: its own singleton, zero distance.
        assert out['zebra crossing'].cluster_name == 'zebra crossing'
        assert out['zebra crossing'].distance == 0.0

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match='same length'):
            rlc.cluster_terms(['a'], [1, 2], rlc.hash_embed)

    def test_rank_orders_by_item_volume(self) -> None:
        a = rlc.ClusterAssignment(1, 'big', 0.0)
        b = rlc.ClusterAssignment(2, 'small', 0.0)
        ranked = rlc.rank_clusters({'x': a, 'y': a, 'z': b}, {'x': 3, 'y': 4, 'z': 5})
        assert [c['cluster_name'] for c in ranked] == ['big', 'small']
        assert ranked[0]['n_items'] == 7
        assert ranked[0]['sample_terms'][0] == {'label': 'y', 'count': 4}


# =============================================================================
# Script <-> endpoint contract
# =============================================================================


@pytest.fixture
def script_mod():
    from scripts.curation import cluster_raw_labels

    return cluster_raw_labels


@pytest.mark.asyncio
async def test_script_populates_what_the_review_endpoint_reads(
    script_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.routers.curation import review

    fake = FakeItemsOpenSearch(_corpus())

    async def _noop(_client: Any) -> None:
        return None

    monkeypatch.setattr(review, '_ensure_indexes', _noop)

    # Before the script runs, the endpoint is the documented empty state.
    before = await review.review_raw_label_clusters(fake, size=50, samples_per_cluster=5)
    assert before['status'] == 'empty'

    summary = await script_mod.run(
        fake,
        index=get_curation_config().items_index,
        embed=rlc.hash_embed,
        distance_threshold=0.6,
        page_size=4,  # force several search_after pages
    )
    assert summary['write']['updated'] == 22
    assert summary['write']['errors'] == 0
    assert len(fake.bulk_bodies) > 1
    assert fake.mapping_puts, 'cluster-field mapping must be ensured before writing'

    after = await review.review_raw_label_clusters(fake, size=50, samples_per_cluster=5)
    assert after['status'] == 'ok'
    by_name = {c['cluster_name']: c for c in after['clusters']}
    pickup = by_name['pickup truck']
    assert pickup['n_crops'] == 13
    assert pickup['n_unmatched'] == 10
    assert {s['label'] for s in pickup['sample_terms']} == {
        'pickup truck',
        'pick-up truck',
        'pickup-truck',
    }
    assert pickup['parent_class_suggestion'] == 'truck'
    assert by_name['forklift']['n_crops'] == 8
    assert by_name['forklift']['parent_class_suggestion'] is None
    # Ranked by volume, as the endpoint orders by count.
    assert after['clusters'][0]['cluster_name'] == 'pickup truck'
    # Items without a raw label were never touched.
    untouched = [d for d in fake.items.values() if rlc.RAW_LABEL_FIELD not in d]
    assert untouched
    assert all(rlc.CLUSTER_ID_FIELD not in d for d in untouched)


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(script_mod: Any) -> None:
    fake = FakeItemsOpenSearch(_corpus())
    summary = await script_mod.run(
        fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6, dry_run=True
    )
    assert summary['n_clusters'] >= 3
    assert fake.bulk_bodies == []
    assert fake.mapping_puts == []
    assert all(rlc.CLUSTER_ID_FIELD not in d for d in fake.items.values())


@pytest.mark.asyncio
async def test_unmatched_only_skips_resolved_items(script_mod: Any) -> None:
    fake = FakeItemsOpenSearch(_corpus())
    await script_mod.run(
        fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6, unmatched_only=True
    )
    resolved = [
        d
        for d in fake.items.values()
        if d.get(rlc.RAW_LABEL_FIELD) and d['class_source'] != rlc.UNMATCHED_CLASS_SOURCE
    ]
    assert resolved
    assert all(rlc.CLUSTER_ID_FIELD not in d for d in resolved)


@pytest.mark.asyncio
async def test_rerun_keeps_cluster_ids_stable(script_mod: Any) -> None:
    fake = FakeItemsOpenSearch(_corpus())
    await script_mod.run(fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6)
    first = {k: d.get(rlc.CLUSTER_ID_FIELD) for k, d in fake.items.items()}
    await script_mod.run(fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6)
    assert {k: d.get(rlc.CLUSTER_ID_FIELD) for k, d in fake.items.items()} == first


@pytest.mark.asyncio
async def test_rerun_on_unchanged_corpus_writes_nothing(script_mod: Any) -> None:
    """F-29: a re-cluster of an unchanged label corpus must not rewrite
    every row — write_back skips docs whose stored cluster id already
    matches the fresh assignment."""
    fake = FakeItemsOpenSearch(_corpus())
    await script_mod.run(fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6)
    assert fake.bulk_bodies  # first run did write something
    fake.bulk_bodies.clear()

    await script_mod.run(fake, index='items', embed=rlc.hash_embed, distance_threshold=0.6)
    assert fake.bulk_bodies == []


def test_auto_embedder_falls_back_to_hash(script_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise ImportError('not installed')

    monkeypatch.setattr(script_mod, '_pe_embedder', _boom)
    monkeypatch.setattr(script_mod, '_st_embedder', _boom)
    fn, label = script_mod.resolve_embedder('auto', 'm')
    assert label == 'hash'
    assert fn is rlc.hash_embed
