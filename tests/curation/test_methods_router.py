"""Tests for GET /curation/methods (curation-strategy plan §3.6/§7 Phase 0/§9).

Mounts the real curation router with OpenSearch stubbed (the endpoint does
no OpenSearch I/O, but the shared app fixture follows the project's
established pattern — see test_review_disagreements.py). Verifies:

* Every real cluster method (ivf/ahc/hdbscan) is reported stable, ivf is
  the sole default (mirrors DEFAULT_METHOD — plan §8 non-goal #1).
* Score-axis entries reflect OP_SCORES_ENABLED / OP_SCORES_SHADOW.
* Disabled entries are never marked default; every advertised id resolves
  via the real registries (cluster_methods.get_method /
  crop_scores.get_scorer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_field_coverage_cache() -> Iterator[None]:
    """The Phase 6 field-coverage lookup is cached at module scope
    (``strategy_registry._COVERAGE_CACHE``, a 60s TTL) so ``GET
    /curation/methods`` stays O(1) per request. Reset it around every test in
    this module so one test's ``fake_os.count`` stub can never leak into
    the next test's assertions."""
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as kb_router

    fake_os = AsyncMock()
    fake_os.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    monkeypatch.delenv('OP_SELECT_DIVERSE_ENABLED', raising=False)
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    monkeypatch.delenv('OP_SEMANTIC_SEARCH_ENABLED', raising=False)

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_cluster_methods_are_stable_and_ivf_is_default(app_client: TestClient) -> None:
    from src.services.curation.clustering.methods import DEFAULT_METHOD, available_methods

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    cluster_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'cluster'}
    assert set(cluster_entries) == set(available_methods())
    for entry in cluster_entries.values():
        assert entry['status'] == 'stable'
    assert cluster_entries[DEFAULT_METHOD]['default'] is True
    non_default = [e for name, e in cluster_entries.items() if name != DEFAULT_METHOD]
    assert all(e['default'] is False for e in non_default)


def test_score_entries_disabled_by_default(app_client: TestClient) -> None:
    from src.services.curation.item_scores import available_scorers

    r = app_client.get('/curation/methods')
    body = r.json()
    score_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'score'}
    assert set(score_entries) == set(available_scorers())
    for entry in score_entries.values():
        assert entry['status'] == 'disabled'
        assert entry['default'] is False
    assert body['flags']['kb_scores_enabled'] is False


def test_score_entries_shadow_when_enabled_and_shadow(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 2 validation (curation-strategy plan §6, see
    docs/design/curation_scores.md) promoted ``mistakenness`` one notch
    (shadow -> experimental) — its full synthetic gate (AUROC + precision@100)
    passed outright with no human/GPU step left unexecuted. ``uniqueness``
    and ``near_dup`` only cleared their cheap pre-screens on real data; each
    method's plan-table *full* gate still needs a step this pass couldn't
    run (blind operator A/B; manually-judged near-dup pairs), so they stay
    ``shadow`` here."""
    from src.services.curation.strategy_registry import VALIDATED_SCORERS

    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setenv('OP_SCORES_SHADOW', '1')
    r = app_client.get('/curation/methods')
    body = r.json()
    score_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'score'}
    for scorer_id, entry in score_entries.items():
        expected = 'experimental' if scorer_id in VALIDATED_SCORERS else 'shadow'
        assert entry['status'] == expected, (
            f'{scorer_id}: expected {expected}, got {entry["status"]}'
        )
    assert VALIDATED_SCORERS  # sanity: at least one scorer has been validated
    # Phase 4/5/P2-14 additive flags (OP_SELECT_DIVERSE_ENABLED,
    # OP_VIZ_PROJECTION_ENABLED, OP_SEMANTIC_SEARCH_ENABLED) joined this
    # envelope; unset here so this test's env matches its own setup above.
    assert body['flags'] == {
        'kb_scores_enabled': True,
        'kb_scores_shadow': True,
        'kb_select_diverse_enabled': False,
        'kb_viz_projection_enabled': False,
        'kb_semantic_search_enabled': False,
    }


def test_score_entries_experimental_when_enabled_not_shadow(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    r = app_client.get('/curation/methods')
    body = r.json()
    score_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'score'}
    for entry in score_entries.values():
        assert entry['status'] == 'experimental'


def test_every_advertised_id_resolves(app_client: TestClient) -> None:
    from src.services.curation.clustering.methods import get_method
    from src.services.curation.item_scores import get_scorer

    r = app_client.get('/curation/methods')
    body = r.json()
    for entry in body['strategies']:
        if entry['axis'] == 'cluster':
            get_method(entry['id'])  # raises ValueError if unknown
        elif entry['axis'] == 'score':
            get_scorer(entry['id'])  # raises ValueError if unknown


def test_diverse_overlay_entry_present_and_disabled_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/methods')
    body = r.json()
    overlay_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'overlay'}
    assert set(overlay_entries) == {'diverse', 'viz_projection', 'semantic_search'}
    entry = overlay_entries['diverse']
    assert entry['status'] == 'disabled'
    assert entry['default'] is False
    assert entry['writes'] == []
    assert body['flags']['kb_select_diverse_enabled'] is False


def test_diverse_overlay_experimental_when_flag_on_but_never_stable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Curation-strategy plan §6/§10.2: only diversity's cheap pre-screen
    passed (docs/design/curation_scores.md §6); the full training A/B gate
    has not run, so this overlay must never advertise 'stable' regardless
    of OP_SELECT_DIVERSE_ENABLED."""
    monkeypatch.setenv('OP_SELECT_DIVERSE_ENABLED', '1')
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'diverse')
    assert entry['status'] == 'experimental'
    assert entry['status'] != 'stable'
    assert body['flags']['kb_select_diverse_enabled'] is True


def test_viz_projection_entry_present_and_disabled_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    assert entry['axis'] == 'overlay'
    assert entry['status'] == 'disabled'
    assert entry['default'] is False
    assert set(entry['writes']) == {'viz_x', 'viz_y', 'viz_projection_version'}
    assert entry['requires_field'] == 'viz_x'
    assert body['flags']['kb_viz_projection_enabled'] is False


def test_viz_projection_experimental_when_flag_on_but_never_stable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Curation-strategy plan §6's UMAP row / §7 Phase 5: the purity half of
    the protocol passed for real (docs/design/curation_scores.md's UMAP-viz
    section), but the interactive-perf half is a frontend check this
    backend-only pass never ran -- same "capped at experimental" reasoning
    ``diverse`` uses for its own still-outstanding gate half."""
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    assert entry['status'] == 'experimental'
    assert entry['status'] != 'stable'
    assert body['flags']['kb_viz_projection_enabled'] is True


def test_viz_projection_carries_measured_purity_and_banner_flag(app_client: TestClient) -> None:
    """The real number from the offline purity-evaluation script (this
    pass, not a placeholder) plus the frontend-facing banner flag --
    ``requires_banner`` is False because the measured purity landed in the
    plan §6 "ship plain" tier (>=0.30), not the 0.15-0.30 banner tier."""
    from src.services.curation.strategy_registry import (
        VIZ_PROJECTION_PURITY,
        VIZ_PROJECTION_REQUIRES_BANNER,
    )

    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    assert entry['purity'] == VIZ_PROJECTION_PURITY
    assert entry['purity'] >= 0.30
    assert entry['requires_banner'] == VIZ_PROJECTION_REQUIRES_BANNER
    assert entry['requires_banner'] is False


def test_export_axis_advertises_yolo_stable_and_omits_lpr(app_client: TestClient) -> None:
    """cropwright_backend_integration_plan.md §4.3/T-C2: the frontend gates
    its LPR export panel on this axis rather than probing the write
    endpoint. ``lpr`` (proprietary, never ported -- Bucket B) must not
    appear at all, not even as a disabled entry."""
    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    export_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'export'}
    assert set(export_entries) == {'yolo'}
    assert export_entries['yolo']['status'] == 'stable'
    assert 'lpr' not in export_entries


def test_detection_profile_axis_advertises_the_registered_default(
    app_client: TestClient,
) -> None:
    """Labeling-assist plan task (b): today exactly one ``DetectionProfile``
    is ever constructed (``cascade_detect.DEFAULT_PROFILE``, registered as
    the default the moment that module is imported -- see
    ``src.services.detection.profile_registry``). This axis must list it,
    keyed by the profile's own ``name`` field, as the sole stable/default
    entry."""
    from src.services.detection.cascade_detect import DEFAULT_PROFILE

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    profile_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'detection_profile'}
    assert set(profile_entries) == {DEFAULT_PROFILE.name}
    entry = profile_entries[DEFAULT_PROFILE.name]
    assert entry['status'] == 'stable'
    assert entry['default'] is True


def test_detection_profile_registry_supports_more_than_one_profile() -> None:
    """The mechanism itself must not be hardcoded to a single entry --
    registering a second profile must surface both, with only the
    explicitly-default one flagged."""
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    saved = profile_registry.get_profiles()
    saved_default = profile_registry.get_default_profile_name()
    try:
        profile_registry._reset_registry_for_tests()
        first = DetectionProfile(name='license_plate')
        second = DetectionProfile(name='shipping_label')
        profile_registry.register_profile(first, default=True)
        profile_registry.register_profile(second)

        from src.services.curation.strategy_registry import _detection_profile_strategies

        entries = {e['id']: e for e in _detection_profile_strategies()}
        assert set(entries) == {'license_plate', 'shipping_label'}
        assert entries['license_plate']['default'] is True
        assert entries['shipping_label']['default'] is False
        assert all(e['axis'] == 'detection_profile' for e in entries.values())
    finally:
        profile_registry._reset_registry_for_tests()
        for profile in saved.values():
            profile_registry.register_profile(profile, default=profile.name == saved_default)


def test_writes_never_include_cluster_fields(app_client: TestClient) -> None:
    forbidden = {'cluster_id', 'cluster_subid', 'cluster_distance'}
    r = app_client.get('/curation/methods')
    body = r.json()
    for entry in body['strategies']:
        writes = set(entry.get('writes') or [])
        assert not (writes & forbidden)


def _fake_field_counts(*, total: int, per_field: dict[str, int] | None = None, default: int = 5):
    """Build an ``AsyncMock`` side_effect distinguishing the ``match_all``
    total-count call from a per-field ``exists`` count call — mirrors
    ``test_scores_router.py::test_coverage_reports_per_field_counts``'s
    convention exactly."""
    per_field = per_field or {}

    async def _count(index: str, body: dict) -> dict:
        query = body['query']
        if 'exists' not in query:
            return {'count': total}
        field = query['exists']['field']
        return {'count': per_field.get(field, default)}

    return AsyncMock(side_effect=_count)


def test_methods_emits_field_coverage_per_entry(app_client: TestClient) -> None:
    """Phase 6 (P1-2/P1-3): every entry — including ones with no
    ``requires_field`` — now carries a ``field_coverage`` key.
    Before this fix, ``/curation/methods`` never emitted the key at all."""
    app_client.fake_os.count = _fake_field_counts(  # type: ignore[attr-defined]
        total=347_837, per_field={'cluster_distance': 124_921, 'crop_area_norm': 347_837}
    )

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    assert body['strategies'], 'sanity: the registry is non-empty'
    for entry in body['strategies']:
        assert 'field_coverage' in entry, entry

    by_id = {s['id']: s for s in body['strategies'] if s['axis'] == 'sort'}
    assert by_id['representativeness']['requires_field'] == 'cluster_distance'
    assert by_id['representativeness']['field_coverage'] == 124_921
    assert by_id['representativeness']['field_coverage_total'] == 347_837
    assert by_id['atypicality']['field_coverage'] == 124_921

    # requires_field is None for 'recent' (updated_at -- every crop always
    # has it) -- coverage doesn't apply, so field_coverage stays None. Note
    # 'default' is deliberately excluded from the sort registry entirely
    # (review_sorts.py's per-tab sentinel, not an independent entry).
    assert by_id['recent']['requires_field'] is None
    assert by_id['recent']['field_coverage'] is None


def test_methods_reports_zero_coverage_for_a_genuinely_inert_sort(
    app_client: TestClient,
) -> None:
    """mistakenness_score/probe_pred_entropy/uniqueness_score/dup_group_id
    are 0% covered on the real pool today (audit-remediation plan §0.1) --
    the whole point of Phase 6 is that this must come through as a real
    zero (hide the control), distinct from an unknown/None."""
    app_client.fake_os.count = _fake_field_counts(  # type: ignore[attr-defined]
        total=347_837,
        per_field={
            'probe_pred_entropy': 0,
            'mistakenness_score': 0,
            'uniqueness_score': 0,
            'dup_group_id': 0,
        },
    )

    r = app_client.get('/curation/methods')
    body = r.json()
    by_id = {s['id']: s for s in body['strategies']}
    assert by_id['uncertainty_entropy']['field_coverage'] == 0
    assert by_id['mistakenness']['field_coverage'] == 0


def test_methods_coverage_is_null_not_zero_on_opensearch_failure(
    app_client: TestClient,
) -> None:
    """The fail-open direction matters (plan Phase 6): a dead OpenSearch
    must not silently hide every control by reporting 0 coverage
    everywhere. Before the fix there was no field_coverage at all; a naive
    fix that defaults failures to 0 (mirroring
    crop_scores/job.py::compute_coverage's precedent) would also fail this
    test."""
    app_client.fake_os.count = AsyncMock(side_effect=RuntimeError('opensearch unreachable'))  # type: ignore[attr-defined]

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    for entry in body['strategies']:
        if entry.get('requires_field'):
            assert entry['field_coverage'] is None, entry
            assert entry['field_coverage_total'] is None, entry


def test_methods_does_not_query_per_entry(app_client: TestClient) -> None:
    """O(1)-ish, not O(entries): distinct requires_field values are far
    fewer than the number of strategy entries (several sorts share a
    field, e.g. both uncertainty_entropy and disagreement_entropy_asc need
    probe_pred_entropy), and a second request inside the 60s TTL must not
    issue any new OpenSearch queries at all."""
    app_client.fake_os.count = _fake_field_counts(total=1000)  # type: ignore[attr-defined]

    r1 = app_client.get('/curation/methods')
    assert r1.status_code == 200
    n_entries = len(r1.json()['strategies'])
    first_call_count = app_client.fake_os.count.call_count  # type: ignore[attr-defined]
    assert 0 < first_call_count < n_entries, (
        f'{first_call_count} queries for {n_entries} entries -- expected '
        'O(distinct fields), not O(entries)'
    )

    r2 = app_client.get('/curation/methods')
    assert r2.status_code == 200
    assert app_client.fake_os.count.call_count == first_call_count, (  # type: ignore[attr-defined]
        'a second request inside the TTL window must be served from cache'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
