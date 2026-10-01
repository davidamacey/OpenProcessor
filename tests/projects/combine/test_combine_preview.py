"""Combine preview (projects plan section 6): mapping validation, suggestions,
capacity, the pinned ``preview_sha`` and the guarantee that a preview writes
nothing."""

from __future__ import annotations

import json
from typing import Any

import pytest
from curation.query_fakes import QueryFakeOpenSearch
from fastapi import HTTPException
from pydantic import ValidationError

from src.routers.curation._config_common_models import api_error
from src.services.projects import lifecycle
from src.services.projects.combine import service
from src.services.projects.combine.models import CombineRequest, CombineStartRequest
from src.services.projects.combine.store import combine_base_dir

from .world import World, snapshot_indexes


BOX = [0.1, 0.1, 0.5, 0.5]


def _two_sources(world: World) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['Car', 'bus'])
    world.add_image(
        'cars-a',
        items=[{'cls': 'car', 'bbox': BOX}, {'cls': 'truck', 'bbox': [0.6] * 2 + [0.9] * 2}],
    )
    world.add_image(
        'cars-b', items=[{'cls': 'Car', 'bbox': BOX}, {'cls': 'bus', 'bbox': [0.6] * 2 + [0.9] * 2}]
    )


FULL_MAPPING = {
    'cars-a': [
        {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'},
        {'dataset_class': 'truck', 'action': 'create', 'new_class_name': 'truck'},
    ],
    'cars-b': [
        {'dataset_class': 'Car', 'action': 'map', 'new_class_name': 'car'},
        {'dataset_class': 'bus', 'action': 'skip'},
    ],
}


async def _preview(world: World, request: CombineRequest):
    result, _analysis = await service.preview(world.fake, request)
    return result


@pytest.mark.asyncio
async def test_unmapped_class_is_an_error_with_its_count(world: World) -> None:
    _two_sources(world)
    mapping = {**FULL_MAPPING, 'cars-b': [FULL_MAPPING['cars-b'][0]]}
    result = await _preview(world, world.request(['cars-a', 'cars-b'], mapping))
    assert not result.ok
    err = next(e for e in result.errors if e.code == 'unmapped_class')
    assert (err.project, err.detail['class'], err.detail['count']) == ('cars-b', 'bus', 1)


@pytest.mark.asyncio
async def test_a_full_mapping_is_ok_and_maps_by_name(world: World) -> None:
    _two_sources(world)
    result = await _preview(world, world.request(['cars-a', 'cars-b'], FULL_MAPPING))
    assert result.ok, result.errors
    classes = {c['name']: c for c in result.target['classes']}
    assert set(classes) == {'car', 'truck'}
    assert classes['car']['count'] == 2  # a's car and b's Car
    assert {(o['project'], o['class']) for o in classes['car']['from']} == {
        ('cars-a', 'car'),
        ('cars-b', 'Car'),
    }
    assert result.target['images'] == 2
    assert result.target['items'] == 3  # a: car + truck, b: Car; the bus is skipped


def test_a_to_or_drop_body_is_not_a_class_mapping_entry() -> None:
    for bad in ({'dataset_class': 'car', 'to': 'vehicle'}, {'dataset_class': 'car', 'drop': True}):
        with pytest.raises(ValidationError):
            CombineRequest.model_validate(
                {
                    'target': {'slug': 't', 'display_name': 't'},
                    'sources': [{'project': 'a'}],
                    'class_mapping': {'a': [bad]},
                }
            )


@pytest.mark.asyncio
async def test_map_to_a_name_no_row_creates_is_invalid(world: World) -> None:
    _two_sources(world)
    mapping = {
        **FULL_MAPPING,
        'cars-b': [
            {'dataset_class': 'Car', 'action': 'map', 'new_class_name': 'vehicle'},
            {'dataset_class': 'bus', 'action': 'skip'},
        ],
    }
    result = await _preview(world, world.request(['cars-a', 'cars-b'], mapping))
    assert [e.code for e in result.errors] == ['mapping_target_invalid']
    assert result.errors[0].detail['new_class_name'] == 'vehicle'


@pytest.mark.asyncio
async def test_suggestions_come_from_w10_name_matching(world: World) -> None:
    _two_sources(world)
    result = await _preview(world, world.request(['cars-a', 'cars-b'], {}))
    suggested = {
        (project, row.dataset_class): (row.action, row.new_class_name)
        for project, rows in result.suggested_mapping.items()
        for row in rows
    }
    assert suggested[('cars-a', 'car')] == ('create', 'car')
    assert suggested[('cars-b', 'Car')] == ('map', 'car')  # case-insensitive match
    assert suggested[('cars-b', 'bus')] == ('create', 'bus')


@pytest.mark.asyncio
async def test_capacity_warn_and_blocked(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_sources(world)
    request = world.request(['cars-a', 'cars-b'], FULL_MAPPING)

    async def warn(_client: Any) -> list[dict[str, str]]:
        return [{'code': 'shard_budget_high', 'message': 'near the shard limit'}]

    monkeypatch.setattr(lifecycle, '_capacity_error_or_warning', warn)
    result = await _preview(world, request)
    assert result.ok
    assert [w.code for w in result.warnings] == ['shard_budget_high']

    async def blocked(_client: Any) -> list[dict[str, str]]:
        raise api_error(409, 'shard_budget_exceeded', 'no room for another project')

    monkeypatch.setattr(lifecycle, '_capacity_error_or_warning', blocked)
    result = await _preview(world, request)
    assert not result.ok
    assert 'shard_budget_exceeded' in [e.code for e in result.errors]


@pytest.mark.asyncio
async def test_environment_errors(world: World) -> None:
    _two_sources(world)
    world.project('taken', ['car'])
    mapping = FULL_MAPPING
    gone = await _preview(world, world.request(['cars-a', 'nope'], mapping))
    assert 'source_not_found' in [e.code for e in gone.errors]
    taken = await _preview(world, world.request(['cars-a', 'cars-b'], mapping, target='taken'))
    assert 'slug_taken' in [e.code for e in taken.errors]
    assert taken.target['slug_available'] is False
    bad = await _preview(world, world.request(['cars-a', 'cars-b'], mapping, target='combine'))
    assert 'slug_invalid' in [e.code for e in bad.errors]
    dup = await _preview(world, world.request(['cars-a', 'cars-a'], mapping))
    assert 'duplicate_source' in [e.code for e in dup.errors]
    self_ = await _preview(world, world.request(['cars-a', 'cars-b'], mapping, target='cars-a'))
    assert 'target_is_source' in [e.code for e in self_.errors]
    many = [f'p{i}' for i in range(9)]
    for slug in many:
        world.project(slug, ['car'])
    too = await _preview(world, world.request(many, {}))
    assert 'too_many_sources' in [e.code for e in too.errors]


@pytest.mark.asyncio
async def test_a_busy_source_is_refused(world: World) -> None:
    _two_sources(world)
    job = combine_base_dir() / 'cmb_20261001T000000_aaaaaaaa'
    job.mkdir(parents=True)
    (job / 'state.json').write_text(
        json.dumps({'status': 'running', 'target': 'other', 'sources': ['cars-a']})
    )
    result = await _preview(world, world.request(['cars-a', 'cars-b'], FULL_MAPPING))
    err = next(e for e in result.errors if e.code == 'source_busy')
    assert err.project == 'cars-a'
    assert any(j.startswith('combine:') for j in err.detail['jobs'])


@pytest.mark.asyncio
async def test_preview_sha_changes_when_a_source_changes(world: World) -> None:
    _two_sources(world)
    request = world.request(['cars-a', 'cars-b'], FULL_MAPPING)
    first = (await _preview(world, request)).preview_sha
    assert (await _preview(world, request)).preview_sha == first
    world.add_image('cars-a', items=[{'cls': 'car', 'bbox': BOX}])
    assert (await _preview(world, request)).preview_sha != first


@pytest.mark.asyncio
async def test_a_stale_sha_is_a_409_on_start(world: World) -> None:
    _two_sources(world)
    request = world.request(['cars-a', 'cars-b'], FULL_MAPPING)
    stale = CombineStartRequest(**request.model_dump(), expected_preview_sha='0' * 64)
    with pytest.raises(HTTPException) as exc:
        await service.start(world.fake, stale)
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'preview_stale'
    assert not combine_base_dir().exists()  # nothing was created


@pytest.mark.asyncio
async def test_start_refuses_an_invalid_request_as_422(world: World) -> None:
    _two_sources(world)
    request = world.request(['cars-a', 'cars-b'], {})
    body = CombineStartRequest(**request.model_dump(), expected_preview_sha='x')
    with pytest.raises(HTTPException) as exc:
        await service.start(world.fake, body)
    assert exc.value.status_code == 422
    assert exc.value.detail['error'] == 'combine_invalid'
    assert exc.value.detail['report']['errors']


class _WriteRecorder(QueryFakeOpenSearch):
    """Fails the test on any call that could change an index."""

    def __getattribute__(self, name: str) -> Any:
        if name in {'bulk', 'index', 'update', 'delete', 'delete_by_query', 'update_by_query'}:
            raise AssertionError(f'preview wrote through {name}')
        return super().__getattribute__(name)


@pytest.mark.asyncio
async def test_preview_writes_nothing(world: World, tmp_path) -> None:
    _two_sources(world)
    before = snapshot_indexes(world, ['cars-a', 'cars-b'])
    registry_before = {
        slug: r.resources.class_registry_path.read_text() for slug, r in world.records.items()
    }
    files_before = sorted(str(p) for p in tmp_path.rglob('*'))
    recorder = _WriteRecorder()
    recorder.store = world.fake.store
    recorder.seq = world.fake.seq
    result, _ = await service.preview(recorder, world.request(['cars-a', 'cars-b'], FULL_MAPPING))
    assert result.ok
    assert snapshot_indexes(world, ['cars-a', 'cars-b']) == before
    assert {
        slug: r.resources.class_registry_path.read_text() for slug, r in world.records.items()
    } == registry_before
    assert sorted(str(p) for p in tmp_path.rglob('*')) == files_before
    assert 'combined' not in world.records
