"""Pins for ``RegionFields``.

See ``docs/design/curation_design_rationale.md`` §4. This file
covers defaults + overridability only — the mapping/query agreement
property (the index-mapping builder produces the same key set as a
``RegionFields`` instance) is covered separately, alongside
``curation_opensearch/bodies_core.py`` (and its mapping builder).

This file is one of the two hardcoded exemptions in
``scripts/codegen/check_no_literal_region_fields.py`` — it legitimately
constructs a ``roi_*``-named instance to prove overridability without
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
    assert f.status == 'region_status'
    assert f.validated == 'region_validated'
    assert f.detector_chain == 'region_detector_chain'
    assert f.boxes == 'region_boxes'
    assert f.box_embeddings == 'region_box_embeddings'
    assert f.class_id == 'region_class_id'
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
    names — no reindex, no code change. This stands in for
    a future overlay instance (never authored on this branch).
    """
    f = RegionFields(
        prefix='roi',
        status='roi_status',
        reason='roi_reason',
        rejection_reason='roi_rejection_reason',
        validated='roi_validated',
        auto_confirmed='roi_auto_confirmed',
        verified='roi_verified',
        verified_at='roi_verified_at',
        verifier='roi_verifier',
        verifier_version='roi_verifier_version',
        visible='roi_visible',
        detector_chain='roi_detector_chain',
        detected_at='roi_detected_at',
        class_id='roi_class_id',
        label_source='roi_label_source',
        pairing='roi_pairing',
        skip_verify='roi_skip_vlm_verify',
        gate_skip='roi_gate_skip',
        status_legacy='roi_status_legacy',
        profile='roi_profile',
        profile_revision='roi_profile_revision',
        boxes='roi_boxes',
        boxes_state='roi_state',
        box_embeddings='roi_box_embeddings',
        count='roi_count',
        rejected_count='roi_rejected_count',
        max_score='roi_max_score',
        set_complete='roi_set_complete',
        revision='roi_revision',
        box_seq='roi_box_seq',
    )
    assert f.status == 'roi_status'
    assert f.boxes == 'roi_boxes'
    assert f.status_legacy == 'roi_status_legacy'
    # Every field really did take the override — nothing silently kept
    # its generic default.
    for field in dataclasses.fields(f):
        value = getattr(f, field.name)
        assert value.startswith('roi') or field.name == 'prefix', (
            f'{field.name} did not take its override: {value!r}'
        )


def test_from_env_overrides_single_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REGION_FIELD_STATUS', 'roi_status')
    f = RegionFields.from_env()
    assert f.status == 'roi_status'
    # Unset fields keep their generic default.
    assert f.boxes == 'region_boxes'


def test_from_env_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MYAPP_MAX_SCORE', 'custom_max_score')
    f = RegionFields.from_env(env_prefix='MYAPP_')
    assert f.max_score == 'custom_max_score'


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

    monkeypatch.setenv('OP_REGION_FIELD_STATUS', 'roi_status_env_override')
    monkeypatch.setattr(region_fields_module, '_default_region_fields', None)
    try:
        f = get_region_fields()
        assert f.status == 'roi_status_env_override'
        # Unset fields keep their generic default even on the singleton.
        assert f.boxes == 'region_boxes'
    finally:
        monkeypatch.setattr(region_fields_module, '_default_region_fields', None)
