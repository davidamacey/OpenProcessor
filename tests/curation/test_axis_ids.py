"""One axis id per axis (any_domain_plan.md §7.1, §9 W2): the axis ids
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
    from src.routers.curation._config_common_models import ActiveConfigResponse

    literal_values = set(get_args(ActiveConfigResponse.model_fields['axis'].annotation))
    assert literal_values == {'prompt_pack', 'detection_profile', 'vlm', 'open_vocab'}
    assert 'region_profile' not in literal_values


def test_no_strategy_entry_uses_region_profile_as_its_axis() -> None:
    """Static guard: the *resource* name ``region_profile`` (routes,
    item stamp) must never leak onto the axis field itself."""
    import inspect

    import src.services.curation.strategy_registry as registry_mod

    source = inspect.getsource(registry_mod)
    assert "'axis': 'region_profile'" not in source
