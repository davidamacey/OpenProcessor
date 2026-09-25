"""Tests for GET /curation/methods.

Mounts the real curation router with OpenSearch stubbed (the endpoint does
no OpenSearch I/O, but the shared app fixture follows the project's
established pattern — see test_review_disagreements.py). Verifies:

* Every real cluster method (ivf/ahc/hdbscan) is reported stable, ivf is
  the sole default (mirrors DEFAULT_METHOD).
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
    from pathlib import Path


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
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    fake_os.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    monkeypatch.delenv('OP_SELECT_DIVERSE_ENABLED', raising=False)
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    monkeypatch.delenv('OP_SEMANTIC_SEARCH_ENABLED', raising=False)

    app = FastAPI()
    app.include_router(curation_router)
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
    assert body['flags']['scores_enabled'] is False


def test_score_entries_shadow_when_enabled_and_shadow(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation promoted
    ``mistakenness`` and ``uniqueness`` one notch each (shadow ->
    experimental) — see ``strategy_registry.VALIDATED_SCORERS``.
    ``near_dup`` only cleared its cheap pre-screen on real data; its full
    gate still needs a step this pass couldn't run (manually-judged
    near-dup pairs), so it stays ``shadow`` here."""
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
        'scores_enabled': True,
        'scores_shadow': True,
        'select_diverse_enabled': False,
        'viz_projection_enabled': False,
        'semantic_search_enabled': False,
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
    assert body['flags']['select_diverse_enabled'] is False


def test_diverse_overlay_experimental_when_flag_on_but_never_stable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only diversity's cheap pre-screen
    passed; the full training A/B gate
    has not run, so this overlay must never advertise 'stable' regardless
    of OP_SELECT_DIVERSE_ENABLED."""
    monkeypatch.setenv('OP_SELECT_DIVERSE_ENABLED', '1')
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'diverse')
    assert entry['status'] == 'experimental'
    assert entry['status'] != 'stable'
    assert body['flags']['select_diverse_enabled'] is True


def test_viz_projection_entry_present_and_disabled_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    assert entry['axis'] == 'overlay'
    assert entry['status'] == 'disabled'
    assert entry['default'] is False
    assert set(entry['writes']) == {'viz_x', 'viz_y', 'viz_projection_version'}
    assert entry['requires_field'] == 'viz_x'
    assert body['flags']['viz_projection_enabled'] is False


def test_viz_projection_experimental_when_flag_on_but_never_stable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The purity half of
    the protocol passed for real (the UMAP-viz
    validation), but the interactive-perf half is a frontend check this
    backend-only pass never ran -- same "capped at experimental" reasoning
    ``diverse`` uses for its own still-outstanding gate half."""
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    assert entry['status'] == 'experimental'
    assert entry['status'] != 'stable'
    assert body['flags']['viz_projection_enabled'] is True


def test_viz_projection_carries_measured_purity_and_banner_flag(app_client: TestClient) -> None:
    """The real number from the offline purity-evaluation script (this
    pass, not a placeholder) plus the frontend-facing banner flag --
    ``requires_banner`` is False because the measured purity landed in the
    "ship plain" tier (>=0.30), not the 0.15-0.30 banner tier."""
    r = app_client.get('/curation/methods')
    body = r.json()
    entry = next(s for s in body['strategies'] if s['id'] == 'viz_projection')
    # Literal expected values (not re-imported from the module under
    # test) — a change to either would be a real, dashboard-visible
    # behavior change this test must catch.
    assert entry['purity'] == pytest.approx(0.472)
    assert entry['purity'] >= 0.30
    assert entry['requires_banner'] is False


def test_export_axis_advertises_yolo_and_single_class_and_omits_lpr(
    app_client: TestClient,
) -> None:
    """The frontend gates
    its export panels on this axis rather than probing the write endpoint.

    ``single_class`` is the generic narrowed export (G2); ``lpr`` must not
    appear at all, not even as a disabled entry -- a domain-named export
    kind is exactly the hardcoding this axis exists to avoid."""
    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    export_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'export'}
    assert set(export_entries) == {'yolo', 'single_class'}
    assert export_entries['yolo']['status'] == 'stable'
    assert export_entries['single_class']['status'] == 'stable'
    assert export_entries['yolo']['default'] is True
    assert export_entries['single_class']['default'] is False
    assert 'lpr' not in export_entries


def test_detection_profile_axis_is_empty_by_default(app_client: TestClient) -> None:
    """Neutral default: with no region profile configured, nothing is
    advertised -- the built-in reference plate profile is selectable by
    name but no longer self-registers as the default."""
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    assert [s for s in body['strategies'] if s['axis'] == 'detection_profile'] == []


@pytest.mark.usefixtures('reference_region_profile')
def test_detection_profile_axis_advertises_the_selected_profile(
    app_client: TestClient,
) -> None:
    """``reference_region_profile`` (``OP_REGION_PROFILE_PATH`` pointed at
    the ``license_plate`` example file) selects that profile; the axis
    lists it, keyed by its ``name``, as the sole stable/default entry."""
    from _region_profile_fixture import (
        EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE,
    )

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    profile_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'detection_profile'}
    assert set(profile_entries) == {REFERENCE_LICENSE_PLATE_PROFILE.name}
    entry = profile_entries[REFERENCE_LICENSE_PLATE_PROFILE.name]
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

        # No shared-settings override configured for this test -- pass the
        # registry's own hardcoded default straight through, same as
        # resolve_effective_default('detection_profile', opensearch=None).
        entries = {
            e['id']: e
            for e in _detection_profile_strategies(profile_registry.get_default_profile_name())
        }
        assert set(entries) == {'license_plate', 'shipping_label'}
        assert entries['license_plate']['default'] is True
        assert entries['shipping_label']['default'] is False
        assert all(e['axis'] == 'detection_profile' for e in entries.values())
    finally:
        profile_registry._reset_registry_for_tests()
        for profile in saved.values():
            profile_registry.register_profile(profile, default=profile.name == saved_default)


def test_prompt_pack_axis_advertises_the_resolved_pack(app_client: TestClient) -> None:
    """With no ``OP_PROMPT_PACK_PATH``
    configured, the axis must advertise the built-in generic pack by its
    own ``name`` field."""
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    pack_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'prompt_pack'}
    assert set(pack_entries) == {GENERIC_ITEM_PACK.name}
    entry = pack_entries[GENERIC_ITEM_PACK.name]
    assert entry['status'] == 'stable'
    assert entry['default'] is True


def test_prompt_pack_axis_advertises_a_deployment_supplied_pack(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment pointing ``OP_PROMPT_PACK_PATH`` at its own pack file
    (task a) sees that pack on the axis as the default, still alongside
    the built-in generic pack (which stays selectable)."""
    import json

    from src.config.curation import CurationConfig
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    custom = GENERIC_ITEM_PACK.to_dict()
    custom['name'] = 'pallet_v1'
    pack_path = tmp_path / 'pack.json'
    pack_path.write_text(json.dumps(custom))

    custom_cfg = CurationConfig(prompt_pack_path=pack_path)
    monkeypatch.setattr('src.config.curation.get_curation_config', lambda: custom_cfg)

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    pack_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'prompt_pack'}
    assert set(pack_entries) == {'pallet_v1', GENERIC_ITEM_PACK.name}
    assert pack_entries['pallet_v1']['default'] is True
    assert pack_entries[GENERIC_ITEM_PACK.name]['default'] is False


def test_prompt_pack_axis_advertises_every_configured_pack(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OP_PROMPT_PACK_PATHS`` adds more selectable packs: the axis lists
    the generic pack, every extra pack, and the default pack, keyed by
    name, with only the ``OP_PROMPT_PACK_PATH`` pack flagged default. An
    unloadable extra is skipped, not fatal."""
    import json

    from src.config.curation import CurationConfig
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    paths = {}
    for name in ('pallet_v1', 'food_v2', 'tools_v1'):
        data = GENERIC_ITEM_PACK.to_dict()
        data['name'] = name
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(data))
    broken = tmp_path / 'broken.json'
    broken.write_text('{not json')

    custom_cfg = CurationConfig(
        prompt_pack_path=paths['pallet_v1'],
        prompt_pack_paths=(paths['food_v2'], broken, paths['tools_v1']),
    )
    monkeypatch.setattr('src.config.curation.get_curation_config', lambda: custom_cfg)

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    pack_entries = {s['id']: s for s in r.json()['strategies'] if s['axis'] == 'prompt_pack'}
    assert set(pack_entries) == {'pallet_v1', 'food_v2', 'tools_v1', GENERIC_ITEM_PACK.name}
    assert [k for k, e in pack_entries.items() if e['default']] == ['pallet_v1']


def test_writes_never_include_cluster_fields(app_client: TestClient) -> None:
    forbidden = {'cluster_id', 'cluster_subid', 'cluster_distance'}
    r = app_client.get('/curation/methods')
    body = r.json()
    for entry in body['strategies']:
        writes = set(entry.get('writes') or [])
        assert not (writes & forbidden)


def _fake_field_counts(*, total: int, per_field: dict[str, int] | None = None, default: int = 5):
    """Build an ``AsyncMock`` side_effect for the one-``_search``-per-field
    coverage query: ``size:0``/``track_total_hits:true`` for the
    pool size, one ``filter: {exists}`` sub-agg per requested field."""
    per_field = per_field or {}

    async def _search(index: str, body: dict) -> dict:
        fields = list(body['aggs'])
        return {
            'hits': {'total': {'value': total}},
            'aggregations': {f: {'doc_count': per_field.get(f, default)} for f in fields},
        }

    return AsyncMock(side_effect=_search)


def test_methods_emits_field_coverage_per_entry(app_client: TestClient) -> None:
    """Phase 6 (P1-2/P1-3): every entry — including ones with no
    ``requires_field`` — now carries a ``field_coverage`` key.
    Before this fix, ``/curation/methods`` never emitted the key at all."""
    app_client.fake_os.search = _fake_field_counts(  # type: ignore[attr-defined]
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
    are 0% covered on the real pool today -- this must come through as a
    real zero (hide the control), distinct from an unknown/None."""
    app_client.fake_os.search = _fake_field_counts(  # type: ignore[attr-defined]
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
    """The fail-open direction matters: a dead OpenSearch
    must not silently hide every control by reporting 0 coverage
    everywhere. Before the fix there was no field_coverage at all; a naive
    fix that defaults failures to 0 (mirroring
    crop_scores/job.py::compute_coverage's precedent) would also fail this
    test."""
    app_client.fake_os.search = AsyncMock(side_effect=RuntimeError('opensearch unreachable'))  # type: ignore[attr-defined]

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    for entry in body['strategies']:
        if entry.get('requires_field'):
            assert entry['field_coverage'] is None, entry
            assert entry['field_coverage_total'] is None, entry


def test_methods_does_not_query_per_entry(app_client: TestClient) -> None:
    """O(1) request, not O(entries) or O(distinct fields): every
    distinct ``requires_field`` (several sorts share one, e.g. both
    uncertainty_entropy and disagreement_entropy_asc need
    probe_pred_entropy) is covered by one ``_search`` with a filter agg
    per field -- not one query per field. A second request inside the 60s
    TTL must not issue any new OpenSearch queries at all."""
    app_client.fake_os.search = _fake_field_counts(total=1000)  # type: ignore[attr-defined]

    r1 = app_client.get('/curation/methods')
    assert r1.status_code == 200
    n_entries = len(r1.json()['strategies'])
    first_call_count = app_client.fake_os.search.call_count  # type: ignore[attr-defined]
    assert first_call_count == 1, (
        f'{first_call_count} queries for {n_entries} entries -- expected exactly one '
        '_search covering every distinct field via filter aggs'
    )

    r2 = app_client.get('/curation/methods')
    assert r2.status_code == 200
    assert app_client.fake_os.search.call_count == first_call_count, (  # type: ignore[attr-defined]
        'a second request inside the TTL window must be served from cache'
    )


def test_methods_contract_declares_field_coverage() -> None:
    """The generated OpenAPI contract must type ``field_coverage`` /
    ``field_coverage_total`` as real StrategyEntry properties, not leave
    the whole response as a free-form (``additionalProperties: true``)
    object -- that's the W2.3 contract gap this test locks down."""
    import json
    from pathlib import Path

    contract_path = Path(__file__).parents[2] / 'contracts' / 'openapi' / 'curation.json'
    spec = json.loads(contract_path.read_text())

    schemas = spec['components']['schemas']
    assert 'StrategyEntry' in schemas, (
        'GET /curation/methods has no typed response_model in the contract'
    )
    entry_props = schemas['StrategyEntry']['properties']
    assert 'field_coverage' in entry_props
    assert 'field_coverage_total' in entry_props


def test_every_sort_with_requires_field_has_integer_coverage(app_client: TestClient) -> None:
    """Regression guard (passes before this change too, per the plan): every
    ``axis == 'sort'`` entry with a ``requires_field`` gets a real int
    ``field_coverage`` / ``field_coverage_total``, never left null just
    because a sort happens to declare one."""
    app_client.fake_os.search = _fake_field_counts(total=1000)  # type: ignore[attr-defined]

    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    body = r.json()
    bad = [
        e['id']
        for e in body['strategies']
        if e['axis'] == 'sort'
        and e.get('requires_field')
        and not isinstance(e.get('field_coverage'), int)
    ]
    assert bad == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
