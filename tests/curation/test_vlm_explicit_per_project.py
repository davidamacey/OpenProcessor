"""A project calls NO VLM until it activates one (#282).

The deployment's ``OP_VLM_URL`` is an endpoint a project opts into. A project
created before this change inherited it; a one-time pin at start writes that
inheritance down as an explicit activation so nothing changes for it.
"""

from __future__ import annotations

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_worker_vlm_hot_reload import _go_cold, _record
from src.config.project_context import bind_project
from src.services.config_store import get_config_store
from src.services.config_store.index import get_activation
from src.services.config_store.vlm_activation import activate_vlm, deactivate_vlm, plan_default_vlm
from src.services.labeling.vlm_endpoints import (
    active_vlm_endpoint,
    refresh_vlm_state,
    vlm_configured,
)
from src.services.projects.vlm_default import pin_legacy_vlm, pin_new_project_vlm_off


@pytest.fixture(autouse=True)
def _cold() -> None:
    _go_cold()


@pytest.mark.asyncio
async def test_a_project_that_never_activated_one_calls_no_vlm(vlm_env: str) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('alpha')):
        await refresh_vlm_state(client)
        assert active_vlm_endpoint() is None
        assert vlm_configured() is False


@pytest.mark.asyncio
async def test_activating_the_env_builtin_opts_the_project_in(vlm_env: str) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('alpha')):
        await activate_vlm(client, name='env', revision=None, expected_active=None)
        endpoint = active_vlm_endpoint()
        assert endpoint is not None
        assert endpoint.name == 'env'


@pytest.mark.asyncio
async def test_settings_null_means_off_not_inherit(vlm_env: str) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('alpha')):
        plan = await plan_default_vlm(client, None)
        assert plan.prepared is None


@pytest.mark.asyncio
async def test_the_legacy_pin_keeps_an_inheriting_project_on_the_env_endpoint(
    vlm_env: str,
) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('old')):
        assert await pin_legacy_vlm(client) == 'env'
        assert (await get_activation(client, get_config_store().index, 'vlm'))['name'] == 'env'
        _go_cold()
        await refresh_vlm_state(client)
        endpoint = active_vlm_endpoint()
        assert endpoint is not None
        assert endpoint.name == 'env'
        assert await pin_legacy_vlm(client) is None  # idempotent


@pytest.mark.asyncio
async def test_the_legacy_pin_without_an_env_endpoint_writes_off_so_a_later_url_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_VLM_URL', raising=False)
    client = FakeConfigOpenSearch()
    with bind_project(_record('old')):
        assert await pin_legacy_vlm(client) == 'off'
        monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000/v1')
        _go_cold()
        await refresh_vlm_state(client)
        assert active_vlm_endpoint() is None


@pytest.mark.asyncio
async def test_a_new_project_is_pinned_off_and_never_migrated_to_env(vlm_env: str) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('fresh')):
        await pin_new_project_vlm_off(client)
        doc = await get_activation(client, get_config_store().index, 'vlm')
        assert doc is not None
        assert doc['name'] is None
        assert await pin_legacy_vlm(client) is None  # the start-up pin leaves it off
        _go_cold()
        await refresh_vlm_state(client)
        assert active_vlm_endpoint() is None


@pytest.mark.asyncio
async def test_deactivate_after_opt_in_is_off(vlm_env: str) -> None:
    client = FakeConfigOpenSearch()
    with bind_project(_record('alpha')):
        await activate_vlm(client, name='env', revision=None, expected_active=None)
        await deactivate_vlm(client, expected_active={'name': 'env', 'revision': None})
        assert active_vlm_endpoint() is None
