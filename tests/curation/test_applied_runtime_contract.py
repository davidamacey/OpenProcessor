"""``AppliedRuntime``: profile/pack/vlm are required and keep their served shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.curation.config_common import AppliedRuntime


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
