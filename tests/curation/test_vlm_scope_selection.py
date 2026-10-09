"""The per-project ``vlm_scope`` policy decides which crops the VLM selectors fetch.

Both selectors (the continuous worker and the auto-label sweep) share one
scope function, so one policy selects the same crops through either.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch, SettingsFakeOpenSearch
from scripts.curation.vlm_worker import _build_pending_query
from src.config.curation import base_curation_config
from src.services.curation.autolabel.selection import resolve_vlm_selection, vlm_selection_query
from src.services.curation.vlm_class_attempt import VLM_CLASS_ATTEMPTED_AT_FIELD
from src.services.curation.vlm_policy import VlmPolicy, VlmPolicyBody
from src.services.curation.vlm_policy_store import put_vlm_policy
from src.services.curation.vlm_scope import (
    daily_budget_remaining,
    representative_ids,
    vlm_scope_clauses,
)


ITEMS = base_curation_config().items_index


def _doc(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': None,
        'class_name': None,
        'class_source': 'det_proposal',
        'class_validated': False,
        'confidence': 0.9,
        'pe_embedding': [0.1],
        'cluster_id': -1,
        **extra,
    }


def _docs() -> dict[str, dict[str, Any]]:
    docs = [
        # cluster 1, nearest first
        _doc('c1a', cluster_id=1, cluster_distance=0.1, confidence=0.95),
        _doc('c1b', cluster_id=1, cluster_distance=0.2, confidence=0.5),
        _doc('c1c', cluster_id=1, cluster_distance=0.3, confidence=0.97),
        # cluster 2
        _doc('c2a', cluster_id=2, cluster_distance=0.1, confidence=0.99),
        _doc('c2b', cluster_id=2, cluster_distance=0.4, confidence=0.99),
        # unassigned
        _doc('u1', cluster_id=-1, confidence=0.99),
        _doc('u2', cluster_id=-1, confidence=0.3),
        # no detector confidence recorded
        _doc('noconf', cluster_id=-1, confidence=None),
        # never selected under any scope
        _doc('human', cluster_id=1, cluster_distance=0.05, class_validated=True),
        _doc(
            'excluded',
            cluster_id=2,
            cluster_distance=0.01,
            class_excluded=True,
            class_validated=True,
        ),
    ]
    return {d['crop_id']: d for d in docs}


class _MsearchFake(SettingsFakeOpenSearch):
    async def msearch(self, *, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        responses = []
        for i in range(0, len(body), 2):
            sub = await self.search(index=body[i]['index'], body=body[i + 1])
            responses.append(sub)
        return {'responses': responses}


def _fake() -> _MsearchFake:
    return _MsearchFake({ITEMS: _docs()})


async def _ids(fake: QueryFakeOpenSearch, query: dict[str, Any]) -> set[str]:
    resp = await fake.search(index=ITEMS, body={'size': 100, 'query': query})
    return {h['_id'] for h in resp['hits']['hits']}


def _policy(**kw: Any) -> VlmPolicy:
    return VlmPolicy(**kw)


ALL = {'c1a', 'c1b', 'c1c', 'c2a', 'c2b', 'u1', 'u2', 'noconf'}


@pytest.mark.asyncio
async def test_default_policy_is_all_and_changes_nothing() -> None:
    fake = _fake()
    assert await _ids(fake, _build_pending_query(0.8)) == ALL
    assert (
        await _ids(
            fake,
            vlm_selection_query(class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8),
        )
        == ALL
    )


@pytest.mark.asyncio
async def test_off_selects_nothing_through_both_selectors() -> None:
    fake = _fake()
    policy = _policy(scope='off')
    assert await _ids(fake, _build_pending_query(0.8, policy=policy)) == set()
    sweep = vlm_selection_query(
        class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8, policy=policy
    )
    assert await _ids(fake, sweep) == set()


@pytest.mark.asyncio
async def test_uncertain_keeps_low_or_missing_detector_confidence_only() -> None:
    fake = _fake()
    policy = _policy(scope='uncertain', conf_max=0.8)
    expected = {'c1b', 'u2', 'noconf'}
    assert await _ids(fake, _build_pending_query(0.8, policy=policy)) == expected
    sweep = vlm_selection_query(
        class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8, policy=policy
    )
    assert await _ids(fake, sweep) == expected


@pytest.mark.asyncio
async def test_representatives_are_the_k_nearest_per_cluster_plus_unassigned() -> None:
    fake = _fake()
    reps = await representative_ids(fake, per_cluster=2)
    # excluded items never represent a cluster; validated ones still rank
    assert reps == {'human', 'c1a', 'c2a', 'c2b'}
    policy = _policy(scope='representatives', per_cluster=2)
    expected = {'c1a', 'c2a', 'c2b', 'u1', 'u2', 'noconf'}  # c1b/c1c rank behind the human crop
    worker = _build_pending_query(0.8, policy=policy, representative_ids=reps)
    sweep = vlm_selection_query(
        class_id=None,
        cluster_id=None,
        classifier_confidence_skip_vlm=0.8,
        policy=policy,
        representative_ids=reps,
    )
    assert await _ids(fake, worker) == expected
    assert await _ids(fake, sweep) == expected


def test_representatives_without_the_id_set_fails_closed() -> None:
    with pytest.raises(ValueError, match='representative'):
        vlm_scope_clauses(_policy(scope='representatives'), representative_ids=None)


@pytest.mark.parametrize('scope', ['all', 'uncertain', 'representatives', 'off'])
@pytest.mark.asyncio
async def test_human_validated_and_excluded_are_never_selected(scope: str) -> None:
    fake = _fake()
    policy = _policy(scope=scope)
    reps = await representative_ids(fake, per_cluster=10)
    worker = await _ids(fake, _build_pending_query(0.8, policy=policy, representative_ids=reps))
    sweep = await _ids(
        fake,
        vlm_selection_query(
            class_id=None,
            cluster_id=None,
            classifier_confidence_skip_vlm=0.8,
            policy=policy,
            representative_ids=reps,
        ),
    )
    assert not {'human', 'excluded'} & (worker | sweep)


def test_worker_and_sweep_share_one_scope_function() -> None:
    policy = _policy(scope='uncertain', conf_max=0.6, sample_frac=0.5)
    clauses = vlm_scope_clauses(policy, representative_ids=None)
    worker = _build_pending_query(0.8, policy=policy)
    sweep = vlm_selection_query(
        class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8, policy=policy
    )
    for clause in (*clauses.filter, *clauses.must_not):
        assert clause in worker['bool']['filter'] + worker['bool']['must_not']
        assert clause in sweep['bool']['filter'] + sweep['bool']['must_not']


def test_sample_frac_below_one_adds_a_stable_per_crop_hash_filter() -> None:
    full = vlm_scope_clauses(_policy(sample_frac=1.0), representative_ids=None)
    half = vlm_scope_clauses(_policy(sample_frac=0.5), representative_ids=None)
    assert full.filter == []
    assert full.must_not == []
    (clause,) = half.filter
    script = clause['script']['script']
    assert 'hashCode' in script['source']
    assert script['params']['frac'] == 0.5
    # No randomness: the same crop gets the same answer on every poll.
    assert 'random' not in script['source'].lower()


@pytest.mark.asyncio
async def test_cluster_scope_run_ignores_the_policy() -> None:
    fake = _fake()
    query = vlm_selection_query(
        class_id=None,
        cluster_id=1,
        classifier_confidence_skip_vlm=0.8,
        policy=_policy(scope='off'),
    )
    assert await _ids(fake, query) == {'c1a', 'c1b', 'c1c'}


@pytest.mark.asyncio
async def test_daily_budget_counts_todays_attempts_and_stops_at_the_cap() -> None:
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    docs = _docs()
    for crop_id, when in (
        ('c1a', now - timedelta(hours=1)),
        ('c1b', now - timedelta(hours=2)),
        ('c2a', now - timedelta(days=1)),  # yesterday: not counted
    ):
        docs[crop_id][VLM_CLASS_ATTEMPTED_AT_FIELD] = when.isoformat()
    fake = QueryFakeOpenSearch({ITEMS: docs})
    assert await daily_budget_remaining(fake, _policy(max_crops_per_day=0), now=now) is None
    assert await daily_budget_remaining(fake, _policy(max_crops_per_day=5), now=now) == 3
    assert await daily_budget_remaining(fake, _policy(max_crops_per_day=2), now=now) == 0
    assert await daily_budget_remaining(fake, _policy(max_crops_per_day=1), now=now) == 0


async def _stored(fake: QueryFakeOpenSearch, *, revision: int = 0, **kw: Any) -> None:
    await put_vlm_policy(fake, VlmPolicyBody(**kw), expected_revision=revision)


async def _resolve(fake: QueryFakeOpenSearch, **kw: Any) -> Any:
    return await resolve_vlm_selection(
        fake,
        class_id=None,
        cluster_id=kw.pop('cluster_id', None),
        classifier_confidence_skip_vlm=0.8,
        item_filter=None,
        max_vlm_crops=kw.pop('max_vlm_crops', 0),
        scope_override=kw.pop('scope_override', None),
        now=kw.pop('now', None),
    )


@pytest.mark.asyncio
async def test_pipeline_reads_the_stored_policy_and_a_query_override_wins() -> None:
    fake = _fake()
    assert await _ids(fake, (await _resolve(fake)).query) == ALL  # no policy stored: all
    await _stored(fake, scope='off')
    assert await _ids(fake, (await _resolve(fake)).query) == set()
    assert await _ids(fake, (await _resolve(fake, scope_override='all')).query) == ALL
    # an explicit cluster request is never limited by the policy
    assert await _ids(fake, (await _resolve(fake, cluster_id=2)).query) == {'c2a', 'c2b'}


@pytest.mark.asyncio
async def test_pipeline_cap_is_the_smaller_of_max_crops_and_the_daily_budget() -> None:
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    docs = _docs()
    docs['c1a'][VLM_CLASS_ATTEMPTED_AT_FIELD] = (now - timedelta(hours=1)).isoformat()
    fake = _MsearchFake({ITEMS: docs})
    await _stored(fake, max_crops_per_day=4)
    assert (await _resolve(fake, now=now)).cap == 3
    assert (await _resolve(fake, now=now, max_vlm_crops=2)).cap == 2
    assert (await _resolve(fake, now=now, max_vlm_crops=50)).cap == 3
    assert (await _resolve(fake, now=now, cluster_id=1)).cap is None  # explicit request
    await _stored(fake, revision=1, max_crops_per_day=1)  # one already attempted today
    exhausted = await _resolve(fake, now=now)
    assert exhausted.cap == 0
    assert await _ids(fake, exhausted.query) == set()


# ---------------------------------------------------------------- worker fetch


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_worker_fetch_applies_scope_budget_and_caches_the_policy_for_30s() -> None:
    from scripts.curation.vlm_worker import fetch_pending_ids
    from src.services.curation.vlm_scope import ScopeCache

    docs = _docs()
    docs['c1a'][VLM_CLASS_ATTEMPTED_AT_FIELD] = datetime.now(UTC).isoformat()
    fake = _MsearchFake({ITEMS: docs})
    clock = _Clock()
    cache = ScopeCache(ttl_s=30.0, clock=clock)

    async def fetch() -> set[str]:
        ids = await fetch_pending_ids(
            fake,
            slug='default',
            batch_size=50,
            classifier_skip_conf=0.8,
            exclude_ids=None,
            scope_cache=cache,
        )
        return set(ids)

    assert await fetch() == ALL
    await _stored(fake, scope='off')
    assert await fetch() == ALL  # the cached policy is still in force (under 30 s)
    clock.now += 31
    assert await fetch() == set()  # re-read: off
    await _stored(fake, revision=1, scope='all', max_crops_per_day=3)
    clock.now += 31
    assert len(await fetch()) == 2  # 3 per day, 1 already attempted today
    await _stored(fake, revision=2, scope='all', max_crops_per_day=1)
    clock.now += 31
    assert await fetch() == set()  # budget exhausted
