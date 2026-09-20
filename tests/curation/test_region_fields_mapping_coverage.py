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
        # ('gemma_plate_visible') baked into the mapping until it was
        # wired through RegionFields.visible.
        'visible',
    )
    for attr in region_attrs_declared:
        key = getattr(F, attr)
        assert key in props, f'expected region field {attr} ({key!r}) declared in items mapping'
        # Sanity: the generic default really is a `region_*` name, not a
        # leaked `plate_*` literal, when using the default singleton.
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
    `plate_*`) names and rebuilding the mapping body with it produces
    `plate_*` keys throughout — no generic code edits required. This is
    the public-branch stand-in for a future proprietary-dataset overlay
    and the reason no reindex is ever needed for a field rename."""
    custom = RegionFields(
        prefix='plate',
        bbox_norm='plate_bbox_norm',
        bbox_frame='plate_bbox_frame',
        score='plate_score',
        verified='plate_verified',
        reason='plate_reason',
        detector='plate_detector',
        detector_version='plate_detector_version',
        detector_chain='plate_detector_chain',
        detected_at='plate_detected_at',
        verifier='plate_verifier',
        verifier_version='plate_verifier_version',
        verified_at='plate_verified_at',
        rejection_reason='plate_rejection_reason',
        bbox_norm_legacy='plate_bbox_norm_legacy',
        score_legacy='plate_score_legacy',
        status_legacy='plate_status_legacy',
        validated='plate_validated',
        embedding='plate_pe_embedding',
        cluster_id='plate_cluster_id',
        cluster_distance='plate_cluster_distance',
        cluster_subid='plate_cluster_subid',
        visible='plate_visible',
    )
    monkeypatch.setattr(cop, 'F', custom)
    rebuilt = cop._items_body()
    props = rebuilt['mappings']['properties']

    for plate_key in (
        'plate_bbox_norm',
        'plate_score',
        'plate_verified',
        'plate_detector',
        'plate_detector_chain',
        'plate_pe_embedding',
        'plate_cluster_id',
        'plate_visible',
    ):
        assert plate_key in props, f'expected override key {plate_key!r} after F swap'

    # The generic region_* defaults must be gone — proves the builder
    # reads through `F` rather than a cached/hardcoded literal.
    for region_key in ('region_bbox_norm', 'region_score', 'region_verified', 'region_visible'):
        assert region_key not in props
