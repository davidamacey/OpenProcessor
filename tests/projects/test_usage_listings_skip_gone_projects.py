"""Every "who uses X" listing counts only projects that exist and are not
being deleted. A tombstoned (``deleted``) or ``deleting`` project's configs
index is gone, and a project with no readable activation is counted as
running the ``env`` built-in -- so a deleted project used to show up in
``active_in`` and could make ``DELETE /vlm/endpoints/env`` 409 ``in_use``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from curation import test_cross_project_leak as _sweep
from curation.test_cross_project_leak import LeakEnv, _promoted_model
from fastapi.testclient import TestClient

from projects.test_model_sharing_used_by import MODEL, _activate_profile, _put
from src.config.curation import IndexRole
from src.services.projects.registry import projects_index


leak_env = _sweep.leak_env

SEEDED_VLM = 'shared-vlm'


def _remove(env: LeakEnv, slug: str, status: str) -> None:
    """What delete-finish leaves behind: the project's indexes are gone and
    its registry record carries the new status."""
    for index in set(env.records[slug].resources.indexes.values()):
        env.transport.store.pop(index, None)
    env.transport.store[projects_index()][f'project:{slug}']['status'] = status


def _vlm_runners(name: str) -> list[str]:
    """What ``GET /vlm/endpoints`` (``active_in``) and ``DELETE`` read."""
    from src.core.dependencies import app_state
    from src.services.config_store.vlm_usage import activations_by_project, slugs_running

    wrapper = app_state._opensearch_client
    assert wrapper is not None
    activations = asyncio.run(activations_by_project(wrapper.client))
    return slugs_running(activations, name)


@pytest.mark.parametrize('status', ['deleted', 'deleting'])
def test_every_listing_drops_a_project_once_it_is_deleted(leak_env: LeakEnv, status: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector=MODEL)
    assert _put(client, shared=True, revision=1).status_code == 200

    assert 'beta' in _vlm_runners(SEEDED_VLM)
    refused = _put(client, shared=False, revision=2)
    assert refused.status_code == 409
    assert refused.json()['detail']['projects'] == ['beta']

    _remove(leak_env, 'beta', status)

    assert 'beta' not in _vlm_runners(SEEDED_VLM)
    assert 'beta' not in _vlm_runners('env')
    assert _put(client, shared=False, revision=2).json()['used_by'] == []


def test_a_project_with_no_activation_that_still_exists_runs_env(leak_env: LeakEnv) -> None:
    index = leak_env.records['beta'].resources.indexes[IndexRole.CONFIGS]
    leak_env.transport.store[index].pop('activation:vlm')
    assert _vlm_runners('env') == ['beta']


def test_read_each_project_skips_gone_projects(leak_env: LeakEnv) -> None:
    from src.services.config_store.project_usage import read_each_project

    async def _slug(_index: str) -> str:
        return 'seen'

    _remove(leak_env, 'beta', 'deleted')
    seen: Any = asyncio.run(read_each_project(_slug))
    assert set(seen) == {'alpha', 'default'}
