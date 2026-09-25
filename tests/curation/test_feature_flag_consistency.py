"""Guard against feature-flag drift (see docs/design/curation_design_rationale.md):
the router's own feature-flag gate and
``GET /curation/methods``' capability status for the same capability
used to read *different* env vars (the router read ``OP_SEMANTIC_SEARCH_ENABLED``
while the registry read a differently-prefixed name for the same flag, same
split for viz projection -- both since unified onto ``OP_*``, following an
earlier env-var rename). Both existing test suites passed the whole time because
neither crossed the seam. This test sets each flag exactly once and
asserts the router's gate and the ``/curation/methods`` status agree,
parameterized over every gated capability so a future name split fails
loudly here instead of silently splitting the UI from the endpoint again.
"""

from __future__ import annotations

import pytest

from src.routers.curation import search as search_router, viz as viz_router
from src.services.curation.strategy_registry import get_registry


# (capability id in GET /curation/methods, env var, router-side gate
# callable, statuses meaning "enabled" for that capability's entry).
_FLAG_CASES = [
    (
        'semantic_search',
        'OP_SEMANTIC_SEARCH_ENABLED',
        search_router._semantic_search_enabled,
        {'experimental', 'stable'},
    ),
    (
        'viz_projection',
        'OP_VIZ_PROJECTION_ENABLED',
        viz_router._viz_enabled,
        {'experimental', 'stable'},
    ),
]


def _entry_status(registry: dict, capability_id: str) -> str:
    for entry in registry['strategies']:
        if entry['id'] == capability_id:
            return entry['status']
    raise AssertionError(f'no /curation/methods entry with id={capability_id!r}')


@pytest.mark.parametrize(
    ('capability_id', 'env_var', 'router_gate', 'enabled_statuses'),
    _FLAG_CASES,
    ids=[c[0] for c in _FLAG_CASES],
)
@pytest.mark.asyncio
async def test_flag_enabled_agrees_between_router_and_methods(
    monkeypatch, capability_id, env_var, router_gate, enabled_statuses
) -> None:
    monkeypatch.setenv(env_var, '1')

    assert router_gate() is True

    registry = await get_registry(None)
    assert _entry_status(registry, capability_id) in enabled_statuses, (
        f'router gate for {capability_id!r} reads {env_var}=1 as enabled, but '
        f'GET /curation/methods still reports status={_entry_status(registry, capability_id)!r} '
        '-- the router and the registry are reading different env vars again'
    )


@pytest.mark.parametrize(
    ('capability_id', 'env_var', 'router_gate', 'enabled_statuses'),
    _FLAG_CASES,
    ids=[c[0] for c in _FLAG_CASES],
)
@pytest.mark.asyncio
async def test_flag_disabled_agrees_between_router_and_methods(
    monkeypatch, capability_id, env_var, router_gate, enabled_statuses
) -> None:
    monkeypatch.delenv(env_var, raising=False)

    assert router_gate() is False

    registry = await get_registry(None)
    assert _entry_status(registry, capability_id) == 'disabled', (
        f'router gate for {capability_id!r} reads {env_var} as unset/disabled, but '
        f'GET /curation/methods reports status={_entry_status(registry, capability_id)!r} '
        '-- the router and the registry are reading different env vars again'
    )
