"""A new project owns two stored indexes, ``images`` and ``items``, and nothing else is derived
from the retired control roles (2F-1).

The one other index is the transitional lease index (``leases``): the B1 job leases live there
until lane 6A deletes them, and it is the only index the control roles left behind.
"""

from __future__ import annotations

import pytest

from src.clients.curation_opensearch.lifecycle import INDEX_BODIES, create_curation_indexes
from src.config.curation_accessors import base_curation_config
from src.config.curation_roles import INDEX_ROLE_ATTR, IndexRole
from src.config.projects import new_project_names, resources_for_new
from tests.opensearch.fake_cluster import FakeCluster


RETIRED = {'labels_confirmed', 'classes', 'settings', 'umap_state', 'umap_viz_state', 'configs'}


def test_the_stored_roles_of_a_new_project_are_images_and_items() -> None:
    assert set(new_project_names('alpha').indexes) == {'images', 'items'}


def test_no_retired_control_role_exists() -> None:
    assert RETIRED.isdisjoint({role.value for role in IndexRole})
    assert RETIRED.isdisjoint(INDEX_ROLE_ATTR.values())
    assert {role.value for role in INDEX_BODIES} == {'images', 'items', 'leases'}


def test_a_new_project_resolves_exactly_its_stored_roles_and_the_lease_index() -> None:
    resources = resources_for_new('alpha', base_curation_config())

    assert {role.value for role in resources.indexes} == {'images', 'items', 'leases'}
    assert set(resources.indexes.values()) == {
        'op_prj_alpha__images',
        'op_prj_alpha__items',
        'op_prj_alpha__leases',
    }


@pytest.mark.asyncio
async def test_creating_a_project_creates_no_control_index() -> None:
    cluster = FakeCluster()

    await create_curation_indexes(cluster)

    created = set(cluster.indexes)
    assert not [n for n in created if n.rsplit('__', 1)[-1] in RETIRED]
    assert {n.rsplit('__', 1)[-1] for n in created} == {'images', 'items', 'leases'}
