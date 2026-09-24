"""Tests for ``src.services.curation.review_sorts`` (curation-strategy plan
§3.2 / §7 Phase 3 / §9 — "the single most important test in this plan").

The golden-body regression guard: every one of the 9 existing
``GET /curation/review/{tab}`` tabs must resolve, with ``?sort`` absent, to the
*exact* OpenSearch sort clause `the review router hardcoded before this
registry existed. A drift here silently reorders a production review queue.
"""

from __future__ import annotations

import pytest

from src.services.curation import review_sorts


# The literal legacy `sort = [...]` the review router hardcoded per tab before
# this registry existed (read directly off the pre-refactor router source —
# see git history for the reference review router). Any change here must be deliberate.
LEGACY_TAB_CLAUSES: dict[str, list[dict]] = {
    'all': [{'cluster_distance': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}],
    'mismatches': [{'updated_at': {'order': 'desc'}}],
    'vlm_low_conf': [{'updated_at': {'order': 'desc'}}],
    'outliers': [
        {'cluster_distance': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'uncertainty': [
        {'probe_pred_entropy': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'regions': [
        {'region_score': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {
            'region_candidate_score': {
                'order': 'desc',
                'missing': '_last',
                'unmapped_type': 'double',
            }
        },
    ],
    'model_disagreements': [
        {'probe_pred_entropy': {'order': 'asc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'primary_low_conf': [
        {'crop_area_norm': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {
            'confidence': {
                'order': 'asc',
                'missing': '_last',
                'unmapped_type': 'double',
            }
        },
    ],
    'coco_blind_spots': [
        {'crop_area_norm': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {'confidence': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
    ],
}

# The legacy default sort id each tab resolves to (independently declared
# here, not imported, so a typo in review_sorts._TAB_DEFAULTS can't also
# hide from this test).
EXPECTED_DEFAULT_IDS: dict[str, str] = {
    'all': 'atypicality',
    'mismatches': 'recent',
    'vlm_low_conf': 'recent',
    'outliers': 'atypicality',
    'uncertainty': 'uncertainty_entropy',
    'regions': 'region_score',
    'model_disagreements': 'disagreement_entropy_asc',
    'primary_low_conf': 'primary_low_conf_default',
    'coco_blind_spots': 'coco_blind_spots_default',
}


@pytest.mark.parametrize('tab', sorted(LEGACY_TAB_CLAUSES))
def test_default_sort_for_tab_matches_expected_literal(tab: str) -> None:
    assert review_sorts.default_sort_for_tab(tab) == EXPECTED_DEFAULT_IDS[tab]


@pytest.mark.asyncio
@pytest.mark.parametrize('tab', sorted(LEGACY_TAB_CLAUSES))
async def test_build_sort_none_is_byte_identical_to_legacy_clause(tab: str) -> None:
    """The golden-body regression guard: ?sort absent must reproduce the
    exact legacy sort clause for every tab, key order and all (OpenSearch
    request bodies are order-sensitive for tie-breaking multi-field sorts).
    No ``opensearch`` client passed -- no shared-settings override lookup,
    same as every pre-existing caller of this function."""
    clause, applied_id, fallback_reason = await review_sorts.build_sort(None, tab=tab)
    assert clause == LEGACY_TAB_CLAUSES[tab]
    assert applied_id == EXPECTED_DEFAULT_IDS[tab]
    assert fallback_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize('tab', sorted(LEGACY_TAB_CLAUSES))
async def test_build_sort_default_literal_matches_none(tab: str) -> None:
    """?sort=default must behave identically to omitting ?sort."""
    via_none = await review_sorts.build_sort(None, tab=tab)
    via_default = await review_sorts.build_sort('default', tab=tab)
    assert via_none == via_default


def test_default_sort_for_tab_rejects_unknown_tab() -> None:
    with pytest.raises(ValueError, match='no default review sort registered'):
        review_sorts.default_sort_for_tab('not-a-real-tab')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'sort_id',
    ['recent', 'representativeness', 'atypicality', 'uncertainty_entropy', 'region_score'],
)
async def test_stable_sorts_resolve_to_their_own_clause(sort_id: str) -> None:
    registry = review_sorts.get_review_sorts()
    expected = registry[sort_id]
    assert expected.status == 'stable'
    clause, applied_id, fallback_reason = await review_sorts.build_sort(sort_id, tab='all')
    assert clause == expected.clause
    assert applied_id == sort_id
    assert fallback_reason is None


def test_representativeness_is_atypicality_flipped() -> None:
    """Plan §2.1 / §6: representativeness's acceptance bar IS this exact
    equality (asc vs desc on the same field), not new math."""
    registry = review_sorts.get_review_sorts()
    rep = registry['representativeness'].clause[0]['cluster_distance']
    atyp = registry['atypicality'].clause[0]['cluster_distance']
    assert rep['order'] == 'asc'
    assert atyp['order'] == 'desc'
    assert {k: v for k, v in rep.items() if k != 'order'} == {
        k: v for k, v in atyp.items() if k != 'order'
    }


@pytest.mark.asyncio
async def test_uniqueness_is_shadow_and_not_selectable() -> None:
    registry = review_sorts.get_review_sorts()
    assert registry['uniqueness'].status == 'shadow'
    with pytest.raises(ValueError, match='not selectable'):
        await review_sorts.build_sort('uniqueness', tab='all')


@pytest.mark.asyncio
async def test_unknown_sort_id_raises() -> None:
    with pytest.raises(ValueError, match='unknown review sort'):
        await review_sorts.build_sort('not-a-real-sort', tab='all')


@pytest.mark.asyncio
async def test_mistakenness_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    registry = review_sorts.get_review_sorts()
    assert registry['mistakenness'].status == 'disabled'
    with pytest.raises(ValueError, match='not selectable'):
        await review_sorts.build_sort('mistakenness', tab='all')


@pytest.mark.asyncio
async def test_mistakenness_promoted_to_experimental_when_enabled_and_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors strategy_registry.VALIDATED_SCORERS's live promotion — this
    status must NEVER be hardcoded in review_sorts.py, only computed."""
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setenv('OP_SCORES_SHADOW', '1')
    registry = review_sorts.get_review_sorts()
    assert registry['mistakenness'].status == 'experimental'
    clause, applied_id, fallback_reason = await review_sorts.build_sort('mistakenness', tab='all')
    assert applied_id == 'mistakenness'
    assert fallback_reason is None
    assert clause == [
        {'mistakenness_score': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}
    ]


@pytest.mark.asyncio
async def test_mistakenness_experimental_when_enabled_not_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    registry = review_sorts.get_review_sorts()
    assert registry['mistakenness'].status == 'experimental'
    await review_sorts.build_sort('mistakenness', tab='all')  # must not raise


def test_no_sort_clause_ever_references_cluster_id_fields() -> None:
    """Plan §8 non-goal #3 / hard constraint: sorting must never touch
    cluster *assignment* fields. cluster_distance (a metric, not an id) is
    fine; cluster_id/cluster_subid must never appear in any sort clause."""
    forbidden = {'cluster_id', 'cluster_subid'}
    for rs in review_sorts.get_review_sorts().values():
        for entry in rs.clause:
            assert not (set(entry.keys()) & forbidden), rs.id


def test_every_review_sort_id_matches_dataclass_id_key() -> None:
    registry = review_sorts.get_review_sorts()
    for key, rs in registry.items():
        assert key == rs.id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
