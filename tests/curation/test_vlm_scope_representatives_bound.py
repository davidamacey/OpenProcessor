"""Scope ``representatives`` is bounded: the VLM is asked about at most
``per_cluster`` crops of each ORIGINAL cluster, however often the reps refresh.

A VLM class write moves the labelled crop out of its candidate cluster into the
class cluster, so a ranking that only looks at the cluster's *current* members
picks the next K on every refresh and drains the whole cluster (issue #192).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from curation.query_fakes import SettingsFakeOpenSearch
from scripts.curation.vlm_worker import _build_pending_query
from src.config.curation import base_curation_config
from src.services.curation.vlm_class_attempt import VLM_CLASS_ATTEMPTED_AT_FIELD
from src.services.curation.vlm_policy import VlmPolicy, VlmPolicyBody
from src.services.curation.vlm_policy_store import put_vlm_policy
from src.services.curation.vlm_scope import ScopeCache, representative_ids


ITEMS = base_curation_config().items_index
CAND = 10000  # candidate cluster ids start here
CLASS_CLUSTER = 3  # a VLM answer moves the crop to cluster_id == class_id


def _doc(crop_id: str, cluster_id: int, dist: float, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_id': None,
        'class_name': None,
        'class_source': 'det_proposal',
        'class_validated': False,
        'confidence': 0.5,
        'pe_embedding': [0.1],
        'cluster_id': cluster_id,
        'cluster_distance': dist,
        'cluster_distance_cluster_id': cluster_id,
        **extra,
    }


def _clusters(sizes: dict[int, int]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for cid, size in sizes.items():
        for n in range(size):
            crop_id = f'k{cid}_{n:02d}'
            docs[crop_id] = _doc(crop_id, cid, n / 100)
    return docs


class _Fake(SettingsFakeOpenSearch):
    async def msearch(self, *, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        return {
            'responses': [
                await self.search(index=body[i]['index'], body=body[i + 1])
                for i in range(0, len(body), 2)
            ]
        }


def _label(fake: _Fake, crop_ids: list[str]) -> None:
    """What a successful VLM class write does to an item (``set_cluster=True``):
    class set, moved to the class cluster, the distance reference left behind."""
    now = datetime.now(UTC).isoformat()
    for crop_id in crop_ids:
        doc = fake.docs(ITEMS)[crop_id]
        doc.update(
            class_id=CLASS_CLUSTER,
            class_name='car',
            class_source='vlm',
            cluster_id=CLASS_CLUSTER,
            **{VLM_CLASS_ATTEMPTED_AT_FIELD: now},
        )


async def _pending(fake: _Fake, reps: set[str], policy: VlmPolicy) -> list[str]:
    query = _build_pending_query(0.8, policy=policy, representative_ids=reps)
    resp = await fake.search(index=ITEMS, body={'size': 1000, 'query': query})
    return sorted(h['_id'] for h in resp['hits']['hits'])


async def _run_rounds(fake: _Fake, policy: VlmPolicy, rounds: int = 8) -> set[str]:
    attempted: set[str] = set()
    for _ in range(rounds):
        reps = await representative_ids(fake, per_cluster=policy.per_cluster)
        todo = await _pending(fake, reps, policy)
        attempted.update(todo)
        _label(fake, todo)
    return attempted


def _bound(sizes: dict[int, int], k: int) -> int:
    return sum(min(k, size) for size in sizes.values())


@pytest.mark.asyncio
async def test_labelled_reps_leaving_their_cluster_are_not_replaced_by_the_next_members() -> None:
    fake = _Fake({ITEMS: _clusters({CAND: 8})})
    policy = VlmPolicy(scope='representatives', per_cluster=2)
    first = await representative_ids(fake, per_cluster=2)
    assert first == {f'k{CAND}_00', f'k{CAND}_01'}
    _label(fake, sorted(first))
    # The next refresh (30 s later, or after a restart) must not hand out k02, k03.
    second = await representative_ids(fake, per_cluster=2)
    assert second == first
    assert await _pending(fake, second, policy) == []


@pytest.mark.asyncio
async def test_total_attempts_over_many_refresh_rounds_stay_within_the_bound() -> None:
    sizes = {CAND: 12, CAND + 1: 3, CAND + 2: 40, CAND + 3: 1}
    fake = _Fake({ITEMS: _clusters(sizes)})
    policy = VlmPolicy(scope='representatives', per_cluster=5)
    attempted = await _run_rounds(fake, policy)
    assert len(attempted) == _bound(sizes, 5)
    per_cluster = {cid: sum(1 for a in attempted if a.startswith(f'k{cid}_')) for cid in sizes}
    assert per_cluster == {cid: min(5, size) for cid, size in sizes.items()}


@pytest.mark.asyncio
async def test_a_policy_revision_change_or_a_restart_does_not_reopen_the_pool() -> None:
    sizes = {CAND: 10, CAND + 1: 10}
    fake = _Fake({ITEMS: _clusters(sizes)})
    await put_vlm_policy(
        fake, VlmPolicyBody(scope='representatives', per_cluster=3), expected_revision=0
    )
    attempted: set[str] = set()
    for revision in range(1, 6):
        # a fresh cache each round = a worker restart; the revision moves too
        policy, reps = await ScopeCache().get(fake, 'default')
        assert policy.revision == revision
        todo = await _pending(fake, reps or set(), policy)
        attempted.update(todo)
        _label(fake, todo)
        await put_vlm_policy(
            fake,
            VlmPolicyBody(scope='representatives', per_cluster=3, conf_max=0.5 + revision / 100),
            expected_revision=revision,
        )
    assert len(attempted) == _bound(sizes, 3)


@pytest.mark.asyncio
async def test_raising_per_cluster_claims_only_the_difference() -> None:
    sizes = {CAND: 10}
    fake = _Fake({ITEMS: _clusters(sizes)})
    attempted = await _run_rounds(fake, VlmPolicy(scope='representatives', per_cluster=2), 3)
    assert len(attempted) == 2
    attempted |= await _run_rounds(fake, VlmPolicy(scope='representatives', per_cluster=5), 3)
    assert len(attempted) == 5
    # lowering it never un-claims or re-claims anything
    attempted |= await _run_rounds(fake, VlmPolicy(scope='representatives', per_cluster=1), 3)
    assert len(attempted) == 5


def _regroup(fake: _Fake, new_ids: list[int]) -> None:
    """A re-cluster: every still-unlabelled crop gets a new candidate cluster id."""
    rest = [d for d in fake.docs(ITEMS).values() if d['cluster_id'] >= CAND]
    for n, doc in enumerate(rest):
        doc['cluster_id'] = new_ids[n % len(new_ids)]
        doc['cluster_distance_cluster_id'] = doc['cluster_id']
        doc['cluster_distance'] = n / 100


@pytest.mark.asyncio
async def test_a_recluster_gives_the_new_clusters_a_fresh_k() -> None:
    fake = _Fake({ITEMS: _clusters({CAND: 10, CAND + 1: 10})})
    policy = VlmPolicy(scope='representatives', per_cluster=2)
    assert len(await _run_rounds(fake, policy, 3)) == 4
    _regroup(fake, [CAND + 5, CAND + 6])  # the old ids are gone
    assert len(await _run_rounds(fake, policy, 4)) == 4  # K per NEW cluster, no cascade


@pytest.mark.asyncio
async def test_a_recluster_reusing_an_id_with_pending_reps_elsewhere_re_picks() -> None:
    fake = _Fake({ITEMS: _clusters({CAND: 10})})
    policy = VlmPolicy(scope='representatives', per_cluster=2)
    claimed = await representative_ids(fake, per_cluster=2)  # claimed, never attempted
    assert claimed == {f'k{CAND}_00', f'k{CAND}_01'}
    _regroup(fake, [CAND + 1, CAND])  # CAND is reused for different members
    reps = await representative_ids(fake, per_cluster=2)
    # CAND's old reps now sit in another cluster: its claim is stale, so CAND is
    # ranked afresh (and CAND+1 is new): two reps each.
    assert len(reps) == 4
    assert len(await _pending(fake, reps, policy)) == 4


@pytest.mark.asyncio
async def test_a_recluster_reusing_an_id_whose_reps_were_all_labelled_fails_closed() -> None:
    fake = _Fake({ITEMS: _clusters({CAND: 10})})
    policy = VlmPolicy(scope='representatives', per_cluster=2)
    assert len(await _run_rounds(fake, policy, 2)) == 2
    _regroup(fake, [CAND])  # same id, no way to tell it is a different cluster
    assert await _run_rounds(fake, policy, 3) == set()


@pytest.mark.asyncio
async def test_unassigned_crops_stay_in_scope() -> None:
    docs = _clusters({CAND: 4})
    docs['u1'] = _doc('u1', -1, 0.0)
    fake = _Fake({ITEMS: docs})
    policy = VlmPolicy(scope='representatives', per_cluster=1)
    attempted = await _run_rounds(fake, policy, 3)
    assert attempted == {f'k{CAND}_00', 'u1'}
