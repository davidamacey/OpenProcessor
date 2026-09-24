"""Pins for ``RegionFields`` (Chunk 0).

See ``docs/design/curation_design_rationale.md`` §4. Chunk 0
covers defaults + overridability only — the mapping/query agreement
property (the index-mapping builder produces the same key set as a
``RegionFields`` instance) is deferred to Chunk 1, when
``curation_opensearch.py`` (and its mapping builder) is ported.

This file is one of the two hardcoded exemptions in
``scripts/codegen/check_no_literal_region_fields.py`` — it legitimately
constructs a ``plate_*``-named instance to prove overridability without
a reindex.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from src.config import RegionFields, get_region_fields
from src.config.region_fields import get_region_fields as get_region_fields_direct


if TYPE_CHECKING:
    import pytest


def test_generic_defaults() -> None:
    f = RegionFields()
    assert f.prefix == 'region'
    assert f.bbox_norm == 'region_bbox_norm'
    assert f.status == 'region_status'
    assert f.score == 'region_score'
    assert f.text == 'region_text'
    assert f.validated == 'region_validated'
    assert f.detector == 'region_detector'
    assert f.embedding == 'region_embedding'
    assert f.cluster_id == 'region_cluster_id'
    assert f.class_id == 'region_class_id'
    assert f.bbox_norm_legacy == 'region_bbox_norm_legacy'
    assert f.score_legacy == 'region_score_legacy'
    assert f.status_legacy == 'region_status_legacy'


def test_is_frozen() -> None:
    f = RegionFields()
    try:
        f.status = 'mutated'  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError('RegionFields must be immutable (frozen dataclass)')


def test_overridability_expresses_a_pre_existing_deployment_shape() -> None:
    """A deployment with pre-existing data under other names is fully
    expressible by constructing a `RegionFields` instance with those
    names — no reindex, no code change. This is the plan's stand-in for
    a future overlay instance (never authored on this branch).
    """
    f = RegionFields(
        prefix='plate',
        bbox_norm='plate_bbox_norm',
        bbox_frame='plate_bbox_frame',
        bbox_correct='plate_bbox_correct',
        status='plate_status',
        score='plate_score',
        confidence='plate_confidence',
        reason='plate_reason',
        rejection_reason='plate_rejection_reason',
        text='plate_text',
        text_raw='plate_text_raw',
        text_confidence='plate_text_confidence',
        text_source='plate_text_source',
        text_engine_version='plate_text_engine_version',
        text_vlm='plate_text_vlm',
        text_ocr='plate_text_ocr',
        text_disagreement='plate_text_disagreement',
        validated='plate_validated',
        verified='plate_verified',
        verified_at='plate_verified_at',
        verifier='plate_verifier',
        verifier_version='plate_verifier_version',
        visible='plate_visible',
        detector='plate_detector',
        detector_version='plate_detector_version',
        detector_chain='plate_detector_chain',
        detected_at='plate_detected_at',
        embedding='plate_pe_embedding',
        cluster_id='plate_cluster_id',
        cluster_subid='plate_cluster_subid',
        cluster_distance='plate_cluster_distance',
        class_id='plate_class_id',
        label_source='plate_label_source',
        source='plate_source',
        pairing='plate_pairing',
        skip_verify='plate_skip_vlm_verify',
        bbox_norm_legacy='plate_bbox_norm_legacy',
        score_legacy='plate_score_legacy',
        status_legacy='plate_status_legacy',
    )
    assert f.status == 'plate_status'
    assert f.bbox_norm == 'plate_bbox_norm'
    assert f.embedding == 'plate_pe_embedding'
    assert f.status_legacy == 'plate_status_legacy'
    # Every field really did take the override — nothing silently kept
    # its generic default.
    for field in dataclasses.fields(f):
        value = getattr(f, field.name)
        assert value.startswith('plate') or field.name == 'prefix', (
            f'{field.name} did not take its override: {value!r}'
        )


def test_from_env_overrides_single_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REGION_FIELD_STATUS', 'plate_status')
    f = RegionFields.from_env()
    assert f.status == 'plate_status'
    # Unset fields keep their generic default.
    assert f.bbox_norm == 'region_bbox_norm'


def test_from_env_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MYAPP_SCORE', 'custom_score')
    f = RegionFields.from_env(env_prefix='MYAPP_')
    assert f.score == 'custom_score'


def test_get_region_fields_returns_singleton() -> None:
    assert get_region_fields() is get_region_fields()
    assert get_region_fields() is get_region_fields_direct()


def test_get_region_fields_singleton_observes_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ``get_region_fields()`` must build the process-wide
    singleton via ``RegionFields.from_env()`` so ``OP_REGION_FIELD_*``
    overrides actually take effect — it previously constructed a bare
    ``RegionFields()`` and silently ignored the env, the same bug class
    ``get_curation_config()`` was already fixed for (mirrors
    ``test_curation_config.test_from_env_overrides_only_set_vars``).
    """
    import src.config.region_fields as region_fields_module

    monkeypatch.setenv('OP_REGION_FIELD_STATUS', 'plate_status_env_override')
    monkeypatch.setattr(region_fields_module, '_default_region_fields', None)
    try:
        f = get_region_fields()
        assert f.status == 'plate_status_env_override'
        # Unset fields keep their generic default even on the singleton.
        assert f.bbox_norm == 'region_bbox_norm'
    finally:
        monkeypatch.setattr(region_fields_module, '_default_region_fields', None)
