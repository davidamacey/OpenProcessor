"""The combine job (projects plan section 6): copy, class identity by name,
dedup across sources, provenance, hard links, and sources left untouched."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.services.detection.geometry import crop_id as make_crop_id
from src.services.projects.combine.store import ACTIVE_STATUSES

from .world import DIM, World, run_job, snapshot_indexes, tree_hash


S1 = [0.1, 0.1, 0.5, 0.5]
S2 = [0.6, 0.6, 0.9, 0.9]
SHARED_SEED = 500

MAPPING = {
    'cars-a': [
        {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'},
        {'dataset_class': 'truck', 'action': 'create', 'new_class_name': 'truck'},
    ],
    'cars-b': [
        {'dataset_class': 'sedan', 'action': 'map', 'new_class_name': 'car'},
        {'dataset_class': 'lorry', 'action': 'map', 'new_class_name': 'truck'},
    ],
}


@dataclass
class Scenario:
    a1: str
    a2: str
    a_shared: str
    b1: str
    b2: str
    b_shared: str
    crops: dict[str, str]


def build(world: World) -> Scenario:
    """Two sources of three images; one image is the same file in both. A
    labels the shared frame's two boxes as detector guesses; B has a human
    label on the first (same target class: a merge the human label wins) and
    a different class on the second (a conflict)."""
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    human = {
        'class_source': 'human',
        'label_source': 'human',
        'class_validated': True,
    }
    a1, (a1_car,) = world.add_image(
        'cars-a', items=[{'cls': 'car', 'bbox': S1, **human}], vector=True
    )
    a2, (a2_truck,) = world.add_image('cars-a', items=[{'cls': 'truck', 'bbox': S2}])
    a_shared, (a_s1, a_s2) = world.add_image(
        'cars-a',
        seed=SHARED_SEED,
        items=[{'cls': 'car', 'bbox': S1}, {'cls': 'car', 'bbox': S2}],
    )
    b1, (b1_sedan,) = world.add_image('cars-b', items=[{'cls': 'sedan', 'bbox': S1, **human}])
    b2, _ = world.add_image('cars-b')
    b_shared, (b_s1, b_s2) = world.add_image(
        'cars-b',
        seed=SHARED_SEED,
        items=[{'cls': 'sedan', 'bbox': S1, **human}, {'cls': 'lorry', 'bbox': S2}],
    )
    return Scenario(
        a1, a2, a_shared, b1, b2, b_shared,
        {
            'a1_car': a1_car, 'a2_truck': a2_truck, 'a_s1': a_s1, 'a_s2': a_s2,
            'b1_sedan': b1_sedan, 'b_s1': b_s1, 'b_s2': b_s2,
        },
    )  # fmt: skip


def target_item(world: World, scenario: Scenario, origin_crop: str, project: str) -> dict[str, Any]:
    matches = [
        d
        for d in world.items('combined').values()
        if d.get('origin_project') == project and d.get('origin_item_id') == origin_crop
    ]
    assert len(matches) == 1, (origin_crop, matches)
    return matches[0]


@pytest.mark.asyncio
async def test_two_sources_combine_into_one_project(world: World) -> None:
    sc = build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    store, _target = await run_job(world, request)

    state = store.job.read()
    assert state['status'] == 'completed', state.get('error')
    # 3 + 3 images, the shared one once.
    assert len(world.images('combined')) == 5
    items = world.items('combined')
    assert len(items) == 5  # a: 1+1+2, b: 1 + (shared: one merged, one conflict)
    report = state['report']
    assert report['images_copied'] == 5
    assert report['images_duplicate'] == 1
    assert report['items_merged'] == 1
    assert report['items_conflicts'] == 1

    ids = world.registry_ids('combined')
    assert ids == {'car': 0, 'truck': 1}
    for doc in items.values():
        assert doc['class_id'] == ids[doc['class_name']]
        assert doc['import_ids'] == [store.import_id]
        assert doc['origin_project'] in ('cars-a', 'cars-b')
        assert 'cluster_id' not in doc
        assert 'class_id_history' not in doc  # snapshots of source numbering
    # Class identity: B's "sedan" and "lorry" arrive as car and truck.
    assert target_item(world, sc, sc.crops['b1_sedan'], 'cars-b')['class_name'] == 'car'
    assert target_item(world, sc, sc.crops['a2_truck'], 'cars-a')['class_name'] == 'truck'


@pytest.mark.asyncio
async def test_provenance_and_label_source_are_preserved(world: World) -> None:
    sc = build(world)
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    human = target_item(world, sc, sc.crops['a1_car'], 'cars-a')
    assert human['label_source'] == 'human'
    assert human['class_validated'] is True
    assert human['origin_image_id'] == sc.a1
    detector = target_item(world, sc, sc.crops['a2_truck'], 'cars-a')
    assert detector['label_source'] == 'detector'
    assert detector['class_validated'] is False
    # The images doc carries the same provenance.
    image = next(d for d in world.images('combined').values() if d['origin_image_id'] == sc.b1)
    assert (image['origin_project'], image['import_ids']) == ('cars-b', [store.import_id])
    # The crop id is recomputed from the TARGET image id.
    for crop_id, doc in world.items('combined').items():
        assert crop_id == make_crop_id(doc['image_id'], doc['bbox_norm'])
        assert doc['image_id'] in world.images('combined')


@pytest.mark.asyncio
async def test_a_duplicate_merges_keeping_the_more_trusted_label(world: World) -> None:
    sc = build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    merged = target_item(world, sc, sc.crops['b_s1'], 'cars-b')  # B's human label won
    assert merged['label_source'] == 'human'
    assert merged['class_validated'] is True
    assert merged['class_name'] == 'car'
    assert merged['crop_id'] == make_crop_id(merged['image_id'], S1)
    assert merged['combine_merged_origins'] == sorted(
        [f'cars-a:{sc.crops["a_s1"]}', f'cars-b:{sc.crops["b_s1"]}']
    )
    assert not merged.get('combine_conflict')
    # A's detector copy of that box is gone: one item per box.
    assert not [
        d for d in world.items('combined').values() if d.get('origin_item_id') == sc.crops['a_s1']
    ]


@pytest.mark.asyncio
async def test_a_conflicting_box_keeps_the_priority_label_and_is_flagged(world: World) -> None:
    sc = build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    conflict = target_item(world, sc, sc.crops['a_s2'], 'cars-a')
    assert conflict['class_name'] == 'car'  # A is listed first; B said truck
    assert conflict['combine_conflict'] is True
    assert conflict['combine_conflict_origins'] == sorted(
        [f'cars-a:{sc.crops["a_s2"]}', f'cars-b:{sc.crops["b_s2"]}']
    )
    assert conflict['class_validated'] is False  # the combine never validates
    flagged = [d for d in world.items('combined').values() if d.get('combine_conflict')]
    assert len(flagged) == 1


@pytest.mark.asyncio
async def test_priority_follows_list_order(world: World) -> None:
    sc = build(world)
    swapped = {'cars-b': MAPPING['cars-b'], 'cars-a': MAPPING['cars-a']}
    await run_job(world, world.request(['cars-b', 'cars-a'], swapped, target='flipped'))
    items = world.items('flipped')
    conflict = next(d for d in items.values() if d.get('combine_conflict'))
    assert conflict['class_name'] == 'truck'  # B first: its "lorry" wins
    assert {conflict['origin_item_id']} == {sc.crops['b_s2']}


@pytest.mark.asyncio
async def test_target_classes_set_the_registry_order(world: World) -> None:
    build(world)
    await run_job(
        world,
        world.request(['cars-a', 'cars-b'], MAPPING, target_classes=['truck', 'car']),
    )
    assert world.registry_ids('combined') == {'truck': 0, 'car': 1}
    for doc in world.items('combined').values():
        assert doc['class_id'] == {'truck': 0, 'car': 1}[doc['class_name']]


@pytest.mark.asyncio
async def test_uploads_are_hard_linked_into_the_target(world: World) -> None:
    sc = build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    target_root = world.records['combined'].resources.upload_root
    for image in world.images('combined').values():
        path = Path(image['image_path'])
        assert path.is_relative_to(target_root), path
        assert path.exists()
    a1_src = Path(world.images('cars-a')[sc.a1]['image_path'])
    a1_dst = next(
        Path(d['image_path'])
        for d in world.images('combined').values()
        if d['origin_image_id'] == sc.a1
    )
    assert a1_dst.stat().st_ino == a1_src.stat().st_ino
    assert a1_src.stat().st_nlink >= 2


@pytest.mark.asyncio
async def test_the_sources_are_unchanged(world: World) -> None:
    build(world)
    slugs = ['cars-a', 'cars-b']
    before = snapshot_indexes(world, slugs)
    dirs = {s: tree_hash(world.records[s].resources.upload_root) for s in slugs}
    registries = {s: world.records[s].resources.class_registry_path.read_bytes() for s in slugs}
    await run_job(world, world.request(slugs, MAPPING))
    assert snapshot_indexes(world, slugs) == before  # docs and per-doc versions
    assert {s: tree_hash(world.records[s].resources.upload_root) for s in slugs} == dirs
    assert {
        s: world.records[s].resources.class_registry_path.read_bytes() for s in slugs
    } == registries


@pytest.mark.asyncio
async def test_validated_only_copies_validated_items_and_their_frames(world: World) -> None:
    sc = build(world)
    await run_job(
        world,
        world.request(
            ['cars-a', 'cars-b'], MAPPING, include={'cars-a': 'validated_only'}, dedup='none'
        ),
    )
    origins = {d['origin_item_id'] for d in world.items('combined').values()}
    assert sc.crops['a1_car'] in origins  # human
    assert sc.crops['a2_truck'] not in origins  # detector guess, excluded
    assert sc.crops['a_s1'] not in origins
    from_a = {
        d['origin_image_id']
        for d in world.images('combined').values()
        if d['origin_project'] == 'cars-a'
    }
    assert from_a == {sc.a1}


@pytest.mark.asyncio
async def test_dedup_none_keeps_both_copies(world: World) -> None:
    build(world)
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING, dedup='none'))
    assert len(world.images('combined')) == 6
    assert store.job.read()['report'].get('items_merged', 0) == 0


@pytest.mark.asyncio
async def test_skipped_class_is_dropped_and_embeddings_survive_by_dimension(
    world: World,
) -> None:
    world.project('p', ['car', 'bus'])
    _image, (car, bus) = world.add_image(
        'p', items=[{'cls': 'car', 'bbox': S1}, {'cls': 'bus', 'bbox': S2}], vector=True
    )
    mapping = {
        'p': [
            {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'},
            {'dataset_class': 'bus', 'action': 'skip'},
        ]
    }
    store, _ = await run_job(world, world.request(['p'], mapping))
    (doc,) = world.items('combined').values()
    assert doc['origin_item_id'] == car
    assert doc['pe_embedding'] == [0.2] * DIM
    assert store.job.read()['report']['items_skipped'] == 1
    assert world.items_index('combined') != world.items_index('p')
    assert bus  # silence unused


@pytest.mark.asyncio
async def test_job_status_is_terminal_and_not_active(world: World) -> None:
    build(world)
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    state = store.job.read()
    assert state['status'] not in ACTIVE_STATUSES
    assert state['phase'] == 'done'
    assert state['next_steps'][0]['action'] == 'recluster'
