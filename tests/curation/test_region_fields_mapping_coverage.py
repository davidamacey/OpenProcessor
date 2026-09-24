"""Mapping/query agreement pin for `RegionFields`.

Guards the silent-divergence failure mode described in
``docs/design/curation_design_rationale.md`` §4: the
items index-mapping body and any OpenSearch query/write path must
resolve region-of-interest field names through the *same*
``RegionFields`` instance, never a re-typed literal. If the mapping
builder ever hardcodes a field name that drifts from the module's `F`
singleton, this test catches it.

Also pins overridability: rebuilding the mapping body with a
differently-named ``RegionFields`` instance (the shape a future
proprietary-dataset overlay would use) changes the produced keys with
zero edits to the generic mapping-builder code — proving a rename is a
config flip, not a code change.
"""

from __future__ import annotations

from src.clients import curation_opensearch as cop
from src.config import IndexRole, RegionFields


def test_items_mapping_region_fields_match_the_module_singleton() -> None:
    """Every region-of-interest key the items mapping declares must equal
    the corresponding attribute on the module's live `F` (RegionFields)
    instance — not a re-typed / stale literal."""
    F = cop.F
    props = cop.INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']

    region_attrs_declared = (
        'bbox_norm',
        'bbox_frame',
        'score',
        'verified',
        'reason',
        'detector',
        'detector_version',
        'detector_chain',
        'detected_at',
        'verifier',
        'verifier_version',
        'verified_at',
        'rejection_reason',
        'bbox_norm_legacy',
        'score_legacy',
        'status_legacy',
        'validated',
        'embedding',
        'cluster_id',
        'cluster_distance',
        'cluster_subid',
        # CFG-8: this was a domain-named, vendor-named literal
        # ('gemma_plate_visible', historic) baked into the mapping until
        # it was wired through RegionFields.visible.
        'visible',
    )
    for attr in region_attrs_declared:
        key = getattr(F, attr)
        assert key in props, f'expected region field {attr} ({key!r}) declared in items mapping'
        # Sanity: the generic default really is a `region_*` name, not a
        # leaked `roi_*` literal, when using the default singleton.
        if F is RegionFields():
            assert key.startswith('region_') or attr in (
                'cluster_id',
                'cluster_distance',
                'cluster_subid',
            )


def test_items_mapping_rebuilt_with_overridden_region_fields_uses_override_keys(
    monkeypatch,
) -> None:
    """Constructing a `RegionFields` instance with pre-existing (e.g.
    `roi_*`) names and rebuilding the mapping body with it produces
    `roi_*` keys throughout — no generic code edits required. This is
    the public-branch stand-in for a future proprietary-dataset overlay
    and the reason no reindex is ever needed for a field rename."""
    custom = RegionFields(
        prefix='roi',
        bbox_norm='roi_bbox_norm',
        bbox_frame='roi_bbox_frame',
        score='roi_score',
        verified='roi_verified',
        reason='roi_reason',
        detector='roi_detector',
        detector_version='roi_detector_version',
        detector_chain='roi_detector_chain',
        detected_at='roi_detected_at',
        verifier='roi_verifier',
        verifier_version='roi_verifier_version',
        verified_at='roi_verified_at',
        rejection_reason='roi_rejection_reason',
        bbox_norm_legacy='roi_bbox_norm_legacy',
        score_legacy='roi_score_legacy',
        status_legacy='roi_status_legacy',
        validated='roi_validated',
        embedding='roi_pe_embedding',
        cluster_id='roi_cluster_id',
        cluster_distance='roi_cluster_distance',
        cluster_subid='roi_cluster_subid',
        visible='roi_visible',
    )
    monkeypatch.setattr(cop, 'F', custom)
    rebuilt = cop._items_body()
    props = rebuilt['mappings']['properties']

    for roi_key in (
        'roi_bbox_norm',
        'roi_score',
        'roi_verified',
        'roi_detector',
        'roi_detector_chain',
        'roi_pe_embedding',
        'roi_cluster_id',
        'roi_visible',
    ):
        assert roi_key in props, f'expected override key {roi_key!r} after F swap'

    # The generic region_* defaults must be gone — proves the builder
    # reads through `F` rather than a cached/hardcoded literal.
    for region_key in ('region_bbox_norm', 'region_score', 'region_verified', 'region_visible'):
        assert region_key not in props
