"""Combine preview warnings that read the sources' stored state:
``embedding_model_mismatch`` and ``region_profiles_differ``."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import get_curation_config
from src.config.project_context import bind_project
from src.services.config_store.index import activation_doc_id, config_doc_id
from src.services.projects.combine import service

from .test_combine_preview import BOX
from .world import DIM, World


MAPPING = {
    'a': [{'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'}],
    'b': [{'dataset_class': 'car', 'action': 'map', 'new_class_name': 'car'}],
}


async def _warnings(world: World, **extra: Any) -> dict[str, Any]:
    result, _ = await service.preview(world.fake, world.request(['a', 'b'], MAPPING, **extra))
    assert result.ok, result.errors
    return {w.code + (f':{w.project}' if w.project else ''): w for w in result.warnings}


def _two_projects(world: World, *, b_dim: int = DIM) -> None:
    for slug in ('a', 'b'):
        world.project(slug, ['car'])
    world.add_image('a', items=[{'cls': 'car', 'bbox': BOX}], vector=True)
    world.add_image('b', items=[{'cls': 'car', 'bbox': BOX, 'pe_embedding': [0.3] * b_dim}])


@pytest.mark.asyncio
async def test_a_source_vector_of_another_dimension_warns_with_its_item_count(
    world: World,
) -> None:
    _two_projects(world, b_dim=DIM + 3)
    found = await _warnings(world)
    assert 'embedding_model_mismatch:a' not in found
    warning = found['embedding_model_mismatch:b']
    assert warning.severity == 'warning'
    assert warning.detail == {
        'project': 'b',
        'items': 1,
        'source_dim': DIM + 3,
        'target_dim': DIM,
    }


@pytest.mark.asyncio
async def test_matching_vector_dimensions_do_not_warn(world: World) -> None:
    _two_projects(world)
    assert not [c for c in await _warnings(world) if c.startswith('embedding_model_mismatch')]


def _activate_profile(world: World, slug: str, name: str, body: dict[str, Any]) -> None:
    """Seed the stored revision copy and the activation doc, as ``activate`` leaves them."""
    with bind_project(world.records[slug]):
        docs = world.fake.docs(get_curation_config().configs_index)
    docs[config_doc_id('region_profile', name, 1)] = {'name': name, 'revision': 1, 'body': body}
    docs[activation_doc_id('detection_profile')] = {'name': name, 'revision': 1}


@pytest.mark.asyncio
async def test_different_region_profiles_warn_and_name_the_target_profile(world: World) -> None:
    _two_projects(world)
    _activate_profile(world, 'a', 'wheels', {'max_regions_per_item': 3})
    _activate_profile(world, 'b', 'wheels', {'max_regions_per_item': 5})
    warning = (await _warnings(world, settings_from='b'))['region_profiles_differ']
    names = {p['project']: p['name'] for p in warning.detail['profiles']}
    assert names == {'a': 'wheels', 'b': 'wheels'}
    hashes = {p['hash'] for p in warning.detail['profiles']}
    assert len(hashes) == 2, 'same name, different content must differ'
    assert warning.detail['target_profile']['name'] == 'wheels'


@pytest.mark.asyncio
async def test_identical_region_profiles_do_not_warn(world: World) -> None:
    _two_projects(world)
    for slug in ('a', 'b'):
        _activate_profile(world, slug, 'wheels', {'max_regions_per_item': 3})
    assert 'region_profiles_differ' not in await _warnings(world)
