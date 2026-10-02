"""``AppliedRuntime``: profile/pack are required and keep their served shape;
the VLM axis is explicit (``null`` for a worker that never reported one)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.routers.curation._config_common_models import AppliedRuntime


def _doc(**extra: object) -> dict[str, object]:
    return {
        'process': 'detection_worker',
        'host': 'w1',
        'applied_config_revision': 3,
        'profile': 'det',
        'profile_revision': 2,
        'pack': 'pk',
        'pack_revision': 1,
        **extra,
    }


def test_profile_and_pack_are_required_in_the_schema() -> None:
    schema = AppliedRuntime.model_json_schema()
    assert {'profile', 'pack', 'vlm'} <= set(schema['required'])


def test_vlm_has_no_silent_default() -> None:
    with pytest.raises(ValidationError):
        AppliedRuntime.model_validate(
            {
                'process': 'p',
                'host': 'h',
                'applied_config_revision': 1,
                'profile': {},
                'pack': {},
            }
        )


def test_profile_and_pack_served_shape_is_unchanged() -> None:
    out = AppliedRuntime.from_runtime_doc(_doc(), config_revision=3).model_dump()
    assert out['profile'] == {'name': 'det', 'revision': 2}
    assert out['pack'] == {'name': 'pk', 'revision': 1}
    assert out['lagging'] is False


def test_vlm_reported_by_the_worker_is_served() -> None:
    out = AppliedRuntime.from_runtime_doc(
        _doc(vlm='v', vlm_revision=4), config_revision=3
    ).model_dump()
    assert out['vlm'] == {'name': 'v', 'revision': 4}


def test_vlm_never_reported_is_null_not_an_empty_ref() -> None:
    out = AppliedRuntime.from_runtime_doc(_doc(), config_revision=3).model_dump()
    assert out['vlm'] is None


def test_vlm_reported_as_off_is_an_empty_ref() -> None:
    out = AppliedRuntime.from_runtime_doc(
        _doc(vlm=None, vlm_revision=None), config_revision=3
    ).model_dump()
    assert out['vlm'] == {'name': None, 'revision': None}
