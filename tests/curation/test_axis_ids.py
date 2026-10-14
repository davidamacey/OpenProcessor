"""One axis id per axis: the axis ids
served by ``GET /methods`` ``axes[]`` agree with
``ActiveConfigResponse.axis`` and ``config.changed.axis`` -- no served
payload ever carries ``"axis": "region_profile"``."""

from __future__ import annotations

from typing import get_args

import pytest


@pytest.mark.asyncio
async def test_methods_axes_ids_are_the_config_axis_ids() -> None:
    from src.services.curation.strategy_registry import get_registry

    registry = await get_registry(None)
    served_axis_ids = {a['axis'] for a in registry['axes']}
    assert 'region_profile' not in served_axis_ids
    assert {'prompt_pack', 'detection_profile', 'vlm'} <= served_axis_ids


def test_active_config_response_axis_literal_matches() -> None:
    from src.schemas.curation.config_common import ActiveConfigResponse

    literal_values = set(get_args(ActiveConfigResponse.model_fields['axis'].annotation))
    assert literal_values == {'prompt_pack', 'detection_profile', 'vlm', 'open_vocab'}
    assert 'region_profile' not in literal_values


def test_no_strategy_entry_uses_region_profile_as_its_axis() -> None:
    """Static guard: the *resource* name ``region_profile`` (routes,
    item stamp) must never leak onto the axis field itself."""
    import inspect

    import src.services.curation.strategy_builders as builders_mod
    import src.services.curation.strategy_registry as registry_mod

    source = inspect.getsource(registry_mod) + inspect.getsource(builders_mod)
    assert "'axis': 'region_profile'" not in source
