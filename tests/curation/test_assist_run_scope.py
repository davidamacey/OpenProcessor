"""Assist run scopes: what a target resolves to, what is skipped, and how the
sample is drawn. Method independent: only the stored ``cluster_uid`` is read."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.core.exceptions import RequestRejectedError
from src.domain.assist_sampling import MAX_ASSIST_ITEMS
from src.services.curation.assist.models import AssistRunRequest, validate_request
from src.services.curation.assist.scope import membership_digest, membership_overlap, resolve_scope


ITEMS = base_curation_config().items_index
CLUSTER = '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'
OTHER = '0c9a9f5e-3b7d-4a6f-8f0e-5d1b2c3d4e5f'


def _doc(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'cluster_uid': CLUSTER,
        'class_source': 'unlabeled_proposal',
        **extra,
    }


def _world() -> QueryFakeOpenSearch:
    docs = {f'a{i:02d}': _doc(f'a{i:02d}') for i in range(30)}
    docs.update({f'b{i:02d}': _doc(f'b{i:02d}') for i in range(10)})
    docs['hold'] = _doc('hold', test_holdout=True)
    docs['excl'] = _doc('excl', class_excluded=True)
    docs['human'] = _doc('human', class_source='human', class_validated=True)
    docs['elsewhere'] = {**_doc('elsewhere'), 'cluster_uid': OTHER}
    return QueryFakeOpenSearch({ITEMS: docs})


def _request(target: dict[str, Any], **over: Any) -> AssistRunRequest:
    return AssistRunRequest.model_validate({'target': target, 'actions': ['classify'], **over})


def _cluster() -> dict[str, Any]:
    return {'kind': 'cluster', 'cluster_uid': CLUSTER}


async def _code(coro: Any) -> tuple[int, str]:
    with pytest.raises(RequestRejectedError) as info:
        await coro
    return info.value.status_code, info.value.detail['error']


def test_segment_without_its_spec_is_refused() -> None:
    request = AssistRunRequest.model_validate({'target': _cluster(), 'actions': ['segment']})
    with pytest.raises(RequestRejectedError) as info:
        validate_request(request)
    assert info.value.detail['error'] == 'invalid_request'


@pytest.mark.asyncio
async def test_a_cluster_resolves_visible_members_and_counts_the_skipped() -> None:
    scope = await resolve_scope(_world(), _request(_cluster()))
    assert len(scope.members) == 41  # 30 + 10 + the human-labelled one
    assert 'hold' not in scope.members
    assert 'excl' not in scope.members
    assert 'elsewhere' not in scope.members
    assert (scope.skipped.holdout, scope.skipped.excluded) == (1, 1)
    assert scope.skipped.class_locked == 1
    assert scope.class_locked_ids == {'human'}
    assert set(scope.sampled_ids) == set(scope.members)


@pytest.mark.asyncio
async def test_an_unknown_cluster_is_a_404() -> None:
    assert await _code(
        resolve_scope(
            _world(),
            _request({**_cluster(), 'cluster_uid': '9d2b7e44-1c3a-4e8b-a7d6-0b5f4c3e2a10'}),
        )
    ) == (
        404,
        'not_found',
    )


def test_the_digest_is_stable_under_reordering_and_sensitive_to_membership() -> None:
    ids = ['c', 'a', 'b']
    assert membership_digest(ids) == membership_digest(['a', 'b', 'c'])
    assert membership_digest(ids) != membership_digest(['a', 'b'])
    assert membership_digest(ids).startswith('sha256:')


def test_overlap_reports_how_much_membership_survived() -> None:
    same = membership_overlap(['a', 'b'], ['b', 'a'])
    assert (same['changed'], same['jaccard']) == (False, 1.0)
    moved = membership_overlap(['a', 'b', 'c', 'd'], ['c', 'd', 'e', 'f'])
    assert (moved['changed'], moved['overlap'], moved['count']) == (True, 2, 4)
    assert moved['jaccard'] == pytest.approx(2 / 6)


@pytest.mark.asyncio
async def test_a_representative_sample_is_reproducible() -> None:
    request = _request(_cluster(), sample={'mode': 'representative', 'seed': 4})
    scope = await resolve_scope(_world(), request)
    assert len(scope.sampled_ids) == 13  # ceil(2 * sqrt(41)) = 13
    again = await resolve_scope(_world(), request)
    assert again.sampled_ids == scope.sampled_ids
    other = await resolve_scope(
        _world(), _request(_cluster(), sample={'mode': 'representative', 'seed': 5})
    )
    assert other.sampled_ids != scope.sampled_ids


@pytest.mark.asyncio
async def test_a_crops_target_keeps_the_callers_order_and_skips_holdout_and_missing() -> None:
    target = {'kind': 'crops', 'crop_ids': ['b03', 'hold', 'a01', 'nope', 'excl']}
    scope = await resolve_scope(_world(), _request(target))
    assert scope.sampled_ids == ['b03', 'a01']
    assert (scope.skipped.holdout, scope.skipped.excluded, scope.skipped.not_found) == (1, 1, 1)


@pytest.mark.asyncio
async def test_a_filter_target_selects_by_the_item_filter() -> None:
    target = {'kind': 'filter', 'filter': {'class_source': 'human'}}
    scope = await resolve_scope(_world(), _request(target))
    assert scope.members == ['human']


def test_a_crops_target_with_a_representative_sample_is_refused() -> None:
    request = _request({'kind': 'crops', 'crop_ids': ['a01']}, sample={'mode': 'representative'})
    with pytest.raises(RequestRejectedError) as info:
        validate_request(request)
    assert (info.value.status_code, info.value.detail['error']) == (422, 'invalid_request')


def test_an_empty_filter_is_refused() -> None:
    with pytest.raises(RequestRejectedError) as info:
        validate_request(_request({'kind': 'filter', 'filter': {}}))
    assert info.value.detail['error'] == 'invalid_request'


def test_a_size_without_a_representative_sample_is_refused() -> None:
    with pytest.raises(RequestRejectedError):
        validate_request(_request(_cluster(), sample={'mode': 'all', 'size': 5}))


@pytest.mark.parametrize('bad', [{'actions': []}, {'actions': ['bogus']}, {'bogus': 1}])
def test_unknown_or_empty_actions_and_keys_fail_the_schema(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='validation error'):
        AssistRunRequest.model_validate({'target': _cluster(), 'actions': ['classify'], **bad})


@pytest.mark.asyncio
async def test_mode_all_over_the_cap_is_assist_too_large() -> None:
    docs = {f'm{i}': _doc(f'm{i}') for i in range(MAX_ASSIST_ITEMS + 1)}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    assert await _code(resolve_scope(fake, _request(_cluster()))) == (422, 'assist_too_large')
    ok = await resolve_scope(fake, _request(_cluster(), sample={'mode': 'representative'}))
    assert len(ok.sampled_ids) == 64
