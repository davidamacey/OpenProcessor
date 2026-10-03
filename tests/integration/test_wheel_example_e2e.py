"""The wheel example, offline, inside a REAL project (W6).

create project (lifecycle route) -> import the fixture (``/datasets/imports``)
-> the primary detector finds vehicles -> the ``vehicle_wheel`` region
profile finds wheels -> export. Fakes sit only at the boundary: OpenSearch,
Triton, the segmenter and the VLM. Everything between is production code,
reached through the HTTP routes wherever a route exists.

Two invariants are asserted at every hop:

* a sibling project (``default``) is never touched: no write to any of its
  indexes at any point, no read of them while the API runs, and its seeded
  document is byte-identical at the end;
* class identity is the NAME: no hop carries a raw class index across a
  boundary, so a box keeps its name through import, detection, region
  processing and export. The fixture's YAML numbering, the project
  registry's ids and the export's dense ids are all different on purpose.

The fixture is built by ``scripts/datasets/build_import_fixture.py`` from a
tiny synthetic COCO-shaped dataset (no network, no COCO pixels).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from curation.config_test_stack import SEGMENTER_URL, Upstream
from PIL import Image
from projects.conftest import fake_ensure_indexes

from integration.ingest_fakes import FakeTritonPool, curation_app
from integration.wheel_worker import run_region_worker
from integration.wheel_world import WRITE_OPS, RoutedOpenSearch
from scripts.datasets import build_import_fixture as fx
from src.config.curation import IndexRole, base_curation_config
from src.config.projects import new_project_record
from src.routers.curation import projects as projects_router
from src.services.detection.cascade_detect import RegionCandidate
from src.services.projects.registry import (
    ProjectRegistry,
    get_project_registry,
    set_project_registry,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

SLUG = 'cars'
API = f'/curation/projects/{SLUG}'
ITEMS_INDEX = f'op_prj_{SLUG}__items'
EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'
# Created in this order on purpose: no registry id equals the id the fixture's
# data.yaml gives the same name (truck 0, Car 1, automobile 2, bus 3).
CLASS_ORDER = ('bus', 'wheel', 'truck', 'car')
# data.yaml name -> registry class name (``Car`` and ``automobile`` merge).
DATASET_TO_NAME = {'Car': 'car', 'automobile': 'car', 'truck': 'truck', 'bus': 'bus'}


class World:
    def __init__(self, client: TestClient, cluster: RoutedOpenSearch, tmp: Path) -> None:
        self.client = client
        self.cluster = cluster
        self.tmp = tmp
        self.default_indexes = {
            name
            for role, name in new_project_record(
                'default', base_curation_config()
            ).resources.indexes.items()
            if role in (IndexRole.ITEMS, IndexRole.IMAGES, IndexRole.LABELS_CONFIRMED)
        }
        self.default_items = next(i for i in self.default_indexes if i.endswith('__items'))
        self.sentinel = {'crop_id': 'sentinel', 'class_name': 'car', 'class_id': 0}
        cluster.data.docs(self.default_items)['sentinel'] = dict(self.sentinel)
        #: Audit position after set-up: everything the test does is after it.
        self.mark = len(cluster.audit)

    def default_traffic(self, *, since: int, writes_only: bool) -> list[tuple[str, str]]:
        return [
            (op, index)
            for op, index in self.cluster.audit[since:]
            if index in self.default_indexes and (not writes_only or op in WRITE_OPS)
        ]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    import src.config.curation as curation_mod
    from src.routers.curation import _common
    from src.services.projects import capacity as capacity_mod
    from src.services.projects.bootstrap import bootstrap_default_project

    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_DATASET_IMPORTS_DIR', str(tmp_path / 'imports'))
    monkeypatch.setattr(curation_mod, '_default_curation_config', None)
    capacity_mod._cache = None
    _common._INDEXES_BOOTSTRAPPED.clear()
    cluster = RoutedOpenSearch()

    async def _make() -> RoutedOpenSearch:
        return cluster

    monkeypatch.setattr(projects_router, 'make_curation_opensearch', _make)
    upstream = Upstream()

    async def serve(_transport: Any, request: httpx.Request) -> httpx.Response:
        return await upstream.handle(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', serve)
    monkeypatch.setenv('OP_SEGMENTER_URL', SEGMENTER_URL)
    with (
        patch(
            'src.routers.curation._common._ensure_indexes',
            new=AsyncMock(side_effect=fake_ensure_indexes),
        ),
        curation_app(
            cluster,
            FakeTritonPool(),
            monkeypatch,
            registry='project',
            state_dir=tmp_path / 'state',
        ) as client,
    ):
        monkeypatch.setattr(curation_mod, '_default_curation_config', None)
        registry = ProjectRegistry(lambda: cluster)
        set_project_registry(registry)
        asyncio.run(bootstrap_default_project(cluster))
        asyncio.run(registry.ensure_fresh())
        yield World(client, cluster, tmp_path)
    set_project_registry(None)
    _common._INDEXES_BOOTSTRAPPED.clear()
    capacity_mod._cache = None


# ----------------------------------------------------------------- the fixture


def synthetic_coco() -> tuple[dict[str, Any], dict[str, Any]]:
    """Faked COCO annotations of the shape ``fetch_coco_subset.py`` writes:
    cars, trucks and buses, a crowd box, and two frames with no vehicle."""
    images, anns = [], []
    for i in range(1, 11):
        images.append({'id': i, 'file_name': f'{i:012d}.jpg', 'width': 320, 'height': 240})
        cat, box = {0: (8, [20, 40, 200, 150]), 1: (6, [30, 30, 240, 160])}.get(
            i % 4, (3, [40 + i, 60, 180, 130])
        )
        anns.append({'id': i, 'image_id': i, 'category_id': cat, 'bbox': box, 'iscrowd': 0})
    anns.append({'id': 99, 'image_id': 2, 'category_id': 3, 'bbox': [0, 0, 30, 30], 'iscrowd': 1})
    images.extend(
        {'id': j, 'file_name': f'{j:012d}.jpg', 'width': 320, 'height': 240} for j in (101, 102)
    )
    categories = [{'id': 3, 'name': 'car'}, {'id': 6, 'name': 'bus'}, {'id': 8, 'name': 'truck'}]
    return {'images': images, 'annotations': anns, 'categories': categories}, {
        'negative_image_ids': [101, 102],
        'seed': 1,
    }


def write_jpeg(name: str, dest: Path, width: int, height: int) -> None:
    """A decodable JPEG whose bytes differ per image (identical bytes would be
    deduplicated into one image by the importer)."""
    n = int(name.split('.')[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (width, height), (n * 37 % 256, n * 91 % 256, n * 13 % 256)).save(
        dest, format='JPEG'
    )


def build_fixture(tmp: Path) -> dict[str, Any]:
    annotations, meta = synthetic_coco()
    return fx.build_variants(annotations, meta, tmp / 'fixture', write_jpeg)


def dataset_boxes(root: Path, names: dict[str, str]) -> list[tuple[str, str, tuple[float, ...]]]:
    """``(image file, dataset class name, xyxy)`` for every valid label row,
    read straight from the label files, independent of the importer."""
    rows: list[tuple[str, str, tuple[float, ...]]] = []
    for txt in sorted((root / 'labels').rglob('*.txt')):
        image = next((root / 'images').rglob(f'{txt.stem}.jpg'), None)
        if image is None:
            continue
        for line in txt.read_text().splitlines():
            parts = line.split()
            if parts[0] not in names:
                continue
            coords = [float(v) for v in parts[1:]]
            if len(parts) == 5:
                cx, cy, w, h = coords
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            elif len(parts) >= 7 and len(parts) % 2 == 1:  # a polygon row is its bounding box
                box = (min(coords[0::2]), min(coords[1::2]), max(coords[0::2]), max(coords[1::2]))
            else:
                continue
            if min(box) < 0 or max(box) > 1:
                continue
            rows.append((image.name, names[parts[0]], box))
    return rows


# --------------------------------------------------------- segmenter + VLM fakes

# One car crop's segmenter candidates, in the item-crop frame: two wheels, a
# roof (no wheel: the VLM stand-in rejects it), a third wheel and a weak
# bumper that ``max_regions_per_item`` (4) must cut.
WHEEL_L = (0.10, 0.60, 0.30, 0.95)
WHEEL_R = (0.65, 0.60, 0.90, 0.95)
ROOF = (0.30, 0.05, 0.70, 0.25)
WHEEL_3 = (0.40, 0.70, 0.50, 0.90)
BUMPER = (0.45, 0.55, 0.55, 0.65)
SCORED = [(WHEEL_L, 0.9), (WHEEL_R, 0.85), (ROOF, 0.6), (WHEEL_3, 0.5), (BUMPER, 0.2)]


def candidates() -> list[RegionCandidate]:
    return [RegionCandidate(bbox_norm=b, score=sc, source='sam3') for b, sc in SCORED]


def to_frame(parent: list[float], crop_box: tuple[float, ...]) -> tuple[float, float, float, float]:
    """A crop-frame box mapped into the source frame (independent of the worker)."""
    px1, py1, px2, py2 = parent
    w, h = px2 - px1, py2 - py1
    return (
        px1 + crop_box[0] * w,
        py1 + crop_box[1] * h,
        px1 + crop_box[2] * w,
        py1 + crop_box[3] * h,
    )


def example_body(kind: str) -> dict[str, Any]:
    raw = json.loads((EXAMPLES / kind / 'vehicle_wheel.json').read_text())
    return {k: v for k, v in raw.items() if k not in ('_comment', 'name')}


def wait_import(world: World, import_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    job: dict[str, Any] = {}
    while time.time() < deadline:
        job = world.client.get(f'{API}/datasets/imports/{import_id}').json()
        if job['status'] in {'completed', 'completed_with_errors', 'failed'}:
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def create_project_with_classes(world: World) -> dict[str, int]:
    r = world.client.post('/curation/projects', json={'slug': SLUG, 'display_name': 'Cars'})
    assert r.status_code == 201, r.text
    for name in CLASS_ORDER:
        assert world.client.post(f'{API}/classes', json={'name': name}).status_code == 201
    listed = world.client.get(f'{API}/classes').json()['classes']
    return {c['class_name']: c['class_id'] for c in listed}


def close_to(a: Any, b: Any) -> bool:
    return all(abs(x - y) < 1e-4 for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------- the tests


def test_wheel_example_end_to_end(
    world: World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, cluster = world.client, world.cluster
    items = cluster.data.docs(ITEMS_INDEX)

    # -- hop 1: create the project through the lifecycle route ---------------
    ids = create_project_with_classes(world)
    assert set(ids) == set(CLASS_ORDER)
    record = get_project_registry().get(SLUG)
    assert record is not None
    assert record.resources.indexes[IndexRole.ITEMS] == ITEMS_INDEX
    assert world.default_traffic(since=world.mark, writes_only=False) == []

    # -- hop 2: import the fixture; labels resolve by NAME -------------------
    fixture = build_fixture(tmp_path)
    spec = fixture['variants']['yolo']
    yaml_names = dict(spec['names'])
    for dataset_name, name in DATASET_TO_NAME.items():
        yaml_index = next(i for i, n in yaml_names.items() if n == dataset_name)
        assert int(yaml_index) != ids[name], 'the fixture must not number classes like the project'
    root = tmp_path / 'fixture' / 'yolo'
    mapping = [
        {'dataset_class': d, 'action': 'map', 'class_id': ids[n]}
        for d, n in DATASET_TO_NAME.items()
    ]
    started = client.post(
        f'{API}/datasets/imports',
        json={
            'source': {'path': str(root)},
            'mapping': mapping,
            'options': {'processing': 'propose'},
        },
    )
    assert started.status_code == 202, started.text
    job = wait_import(world, started.json()['import_id'])
    assert job['status'] == 'completed', job
    report = job['report']
    expected = spec['expected_report']
    assert report['negatives'] == expected['label_states']['negative']
    assert report['unlabeled'] == expected['label_states']['unlabeled']
    assert report['items_created'] == sum(expected['boxes_per_class'].values())

    imported = [d for d in items.values() if d.get('class_source') == 'external_label']
    by_box = {
        (Path(d['image_path']).name, tuple(round(v, 4) for v in d['bbox_norm'])): d
        for d in imported
    }
    truth = dataset_boxes(root, yaml_names)
    assert len(truth) == len(imported)
    for image, dataset_name, xyxy in truth:
        item = by_box[(image, tuple(round(v, 4) for v in xyxy))]
        name = DATASET_TO_NAME[dataset_name]
        assert (item['class_name'], item['class_id']) == (name, ids[name]), (image, dataset_name)
        assert item['class_validated'] is True
    per_name = {
        n: sum(1 for d in imported if d['class_name'] == n) for n in ('car', 'truck', 'bus')
    }
    assert (
        per_name['car']
        == expected['boxes_per_class']['Car'] + expected['boxes_per_class']['automobile']
    )

    # -- hop 3: the primary detector's proposals never take a class by index --
    proposals = [d for d in items.values() if d.get('proposed_by_import')]
    assert len(proposals) == report['proposals_created'] == 12
    assert {d.get('class_name') for d in proposals} == {None}
    assert {d.get('class_id') for d in proposals} == {None}
    assert {d['class_source'] for d in proposals} == {'item_proposal'}
    assert not {d.get('proposal_name') for d in proposals} & set(ids)

    # -- hop 4: the example profile and pack, through the config routes ------
    for kind in ('region_profiles', 'prompt_packs'):
        body = {'name': 'wheel_example', 'body': example_body(kind)}
        assert client.post(f'{API}/{kind}', json=body).status_code == 201
        act = client.post(f'{API}/{kind}/wheel_example/activate', json={'expected_active': None})
        assert act.status_code == 200, act.text
        assert act.json()['active'] == {'name': 'wheel_example', 'revision': 1}
    dry = client.post(
        f'{API}/reprocess',
        json={
            'targets': {'filter': {'missing_status': True}},
            'scopes': ['region'],
            'dry_run': True,
        },
    )
    assert dry.status_code == 200
    assert dry.json()['scopes'][0]['queued'] == 0
    go = client.post(
        f'{API}/reprocess',
        json={
            'targets': {'filter': {'missing_status': True}},
            'scopes': ['region'],
            'dry_run': False,
        },
    )
    assert go.status_code == 200, go.text
    # Only the profile's parent class (car) is queued; truck/bus items are not seeded.
    in_scope = {
        c
        for c, d in items.items()
        if 'car' in {(d.get('class_name') or '').lower(), (d.get('proposal_name') or '').lower()}
    }
    assert in_scope
    assert {
        c for c, d in items.items() if d.get('region_status') == 'pending_detection'
    } == in_scope
    assert world.default_traffic(since=world.mark, writes_only=False) == []

    # -- hop 5: the worker finds wheels, on car items only (by name) ---------
    cars = {k: d for k, d in items.items() if d.get('class_name') == 'car'}
    assert len(cars) == per_name['car']
    before_worker = len(cluster.audit)

    def done() -> bool:
        return all(items[k].get('region_status') != 'pending_detection' for k in cars)

    mocks = asyncio.run(
        run_region_worker(
            tmp_path,
            monkeypatch,
            cluster=cluster,
            records=[get_project_registry().get(SLUG), get_project_registry().get('default')],  # type: ignore[list-item]
            segmenter=candidates,
            done=done,
        )
    )
    assert len(mocks.segment_calls) == len(cars)
    assert sorted(c for batch in mocks.vlm_batches for c in batch) == sorted(cars)
    for item in cars.values():
        assert item['class_name'] == 'car'  # the region stage never rewrites the class
        assert item['region_status'] == 'detected'
        assert item['region_verifier'] == 'fake-vlm'
        boxes = item['region_boxes']
        assert [b['state'] for b in boxes] == ['accepted', 'accepted', 'rejected', 'accepted']
        assert [b['score'] for b in boxes] == [0.9, 0.85, 0.6, 0.5]  # the 0.2 bumper was cut
        for box, (crop_box, _score) in zip(boxes, SCORED, strict=False):
            assert close_to(box['bbox_norm'], to_frame(item['bbox_norm'], crop_box))
        assert (item['region_count'], item['region_rejected_count']) == (3, 1)
        # provenance: the profile and pack that produced it, the endpoint that verified it
        assert (item['region_profile'], item['region_profile_revision']) == ('wheel_example', 1)
        assert item['vlm_prompt_pack'] == 'wheel_example@1'
        assert (item['vlm_endpoint'], item['vlm_model']) == ('fake-vlm@1', 'fake-vlm')
        assert item['region_detector_chain'] == [
            'vlm_visible:yes',
            'sam3:hit',
            'sam3:combined_verify_ok',
        ]
        for box in boxes:
            assert (box['detector'], box['source']) == ('sam3', 'segmenter')
            # a text-free profile: every text field of every box is empty
            assert {k: v for k, v in box.items() if k.startswith('text') and v} == {}
    untouched = {k: d for k, d in items.items() if k not in cars}
    assert untouched
    for k, d in untouched.items():
        # in-scope proposals stay queued; items outside the parent class never were
        assert d.get('region_status') == ('pending_detection' if k in in_scope else None)
        assert not d.get('region_boxes')
    assert [
        e
        for e in cluster.audit[before_worker:]
        if e[1] in world.default_indexes and e[0] != 'search'
    ] == []
    assert world.default_traffic(since=world.mark, writes_only=True) == []

    # -- hop 5b: a human adds a missed wheel to one car through the route ----
    edited_id = sorted(cars)[0]
    edited = items[edited_id]
    missed = (0.02, 0.70, 0.08, 0.90)
    put = client.put(
        f'{API}/crops/{edited_id}/regions',
        json={
            'boxes': [{'box_id': b['box_id']} for b in edited['region_boxes']]
            + [{'box_id': None, 'bbox_norm': list(to_frame(edited['bbox_norm'], missed))}],
            'frame': 'source',
            'expected_region_revision': edited['region_revision'],
        },
    )
    assert put.status_code == 200, put.text
    edited = items[edited_id]
    assert [b['state'] for b in edited['region_boxes']] == ['accepted'] * 2 + ['rejected'] + [
        'accepted'
    ] * 2
    assert (edited['region_count'], edited['region_rejected_count']) == (4, 1)
    assert edited['region_validated'] is True
    assert edited['region_label_source'] == 'human'
    assert edited['region_boxes'][2]['rejection_reason'] == 'region_visible_elsewhere'
    assert world.default_traffic(since=world.mark, writes_only=True) == []

    # -- hop 6: export wheels from the car crops -----------------------------
    exp = client.post(
        f'{API}/export/single_class',
        json={
            'profile_name': 'wheels',
            'class_ids': [ids['car']],
            'box_source': 'region',
            'region_class_name': 'wheel',
            'image_mode': 'item_crop',
            'seed': 7,
            'empty_bg_ratio': 0.0,
        },
    )
    assert exp.status_code == 200, exp.text
    out = Path(exp.json()['export_dir'])
    assert out.is_relative_to(record.resources.export_root)
    assert exp.json()['positive_images'] == len(cars)
    assert '  0: wheel\n' in (out / 'data.yaml').read_text()
    registry_doc = json.loads((out / 'class_registry.json').read_text())
    assert registry_doc['names'] == ['wheel']
    # the parent filter is named by the project's own registry, never by position
    assert registry_doc['classes'] == [{'class_id': ids['car'], 'class_name': 'car'}]
    wheel_rows = sorted(
        tuple(round(float(v), 4) for v in line.split()[1:])
        for crop_box in (WHEEL_L, WHEEL_R, WHEEL_3)
        for line in [
            '0 '
            + ' '.join(
                f'{v:.6f}'
                for v in (
                    (crop_box[0] + crop_box[2]) / 2,
                    (crop_box[1] + crop_box[3]) / 2,
                    crop_box[2] - crop_box[0],
                    crop_box[3] - crop_box[1],
                )
            )
        ]
    )
    label_files = sorted((out / 'labels').rglob('*.txt'))
    assert len(label_files) == len(cars)
    missed_row = (
        round((missed[0] + missed[2]) / 2, 4),
        round((missed[1] + missed[3]) / 2, 4),
        round(missed[2] - missed[0], 4),
        round(missed[3] - missed[1], 4),
    )
    edited_labels = [p for p in label_files if edited_id in p.stem]
    assert len(edited_labels) == 1, 'the human-edited car is exported under its own crop id'
    for path in label_files:
        rows = [ln.split() for ln in path.read_text().splitlines()]
        assert {r[0] for r in rows} == {'0'}  # the one dense id: wheel
        got = sorted(tuple(round(float(v), 4) for v in r[1:]) for r in rows)
        # three machine wheels per car, plus the one the human added to the edited car
        assert got == (sorted([*wheel_rows, missed_row]) if path in edited_labels else wheel_rows)
    by_split = {s: len(list((out / 'labels' / s).glob('*.txt'))) for s in ('train', 'val')}
    for split in ('train', 'val'):
        assert by_split[split] == sum(1 for d in cars.values() if d['dataset_split'] == split)

    pre = client.post(f'{API}/train/preflight', json={'dataset_export_dir': str(out)})
    assert pre.status_code == 200, pre.text
    checks = {c['name']: c['severity'] for c in pre.json()['checks']}
    # The export itself is sound for a single-class run: pairing, emptiness and
    # splits all pass. What blocks is only the environment (no trainer volume
    # in this process) and the 5-frame size of the toy dataset.
    for name in ('region_pairing', 'empty_labels', 'export_not_empty', 'export_splits_nonempty'):
        assert checks[name] == 'ok', (name, pre.json())
    assert {n for n, sev in checks.items() if sev == 'block'} == {
        'free_disk',
        'class_balance',
        'test_holdout',
    }

    # -- the sibling project, end to end --------------------------------------
    assert world.default_traffic(since=world.mark, writes_only=True) == []
    assert cluster.data.docs(world.default_items) == {'sentinel': world.sentinel}


def test_audit_catches_a_write_to_the_sibling_project(world: World) -> None:
    """The ``default``-untouched assertions above are only worth something if
    the audit would notice a write: prove it does, for every write shape."""
    cluster = world.cluster
    asyncio.run(cluster.index(index=world.default_items, id='x', body={'a': 1}))
    asyncio.run(cluster.update(index=world.default_items, id='x', body={'doc': {'a': 2}}))
    asyncio.run(
        cluster.bulk(
            body=[{'index': {'_index': world.default_items, '_id': 'y'}}, {'a': 3}], refresh=False
        )
    )
    ops = {op for op, _ in world.default_traffic(since=world.mark, writes_only=True)}
    assert {'index', 'update', 'bulk'} <= ops


def test_coco_variant_is_detected_and_imports(world: World) -> None:
    """The ``coco/`` layout of the same fixture (``images/`` next to
    ``annotations/``) is detected as COCO without naming the format."""
    ids = create_project_with_classes(world)
    fixture = build_fixture(world.tmp)
    spec = fixture['variants']['coco']
    root = world.tmp / 'fixture' / 'coco'
    body = {'source': {'path': str(root)}}
    pre = world.client.post(f'{API}/datasets/preview', json=body)
    assert pre.status_code == 200, pre.text
    assert pre.json()['format'] == 'coco'
    issues = {i['code']: i['count'] for i in pre.json()['issues']}
    assert issues.get('coco_crowd_skipped', 0) == spec['expected_issues']['coco_crowd_skipped']
    mapping = [
        {'dataset_class': n, 'action': 'map', 'class_id': ids[n]} for n in ('car', 'truck', 'bus')
    ]
    started = world.client.post(f'{API}/datasets/imports', json={**body, 'mapping': mapping})
    assert started.status_code == 202, started.text
    job = wait_import(world, started.json()['import_id'])
    assert job['status'] == 'completed', job
    items = world.cluster.data.docs(ITEMS_INDEX)
    got = {n: sum(1 for d in items.values() if d.get('class_name') == n) for n in ids}
    assert got['car'] == spec['expected_report']['boxes_per_class']['car']
    assert got['truck'] == spec['expected_report']['boxes_per_class']['truck']
    assert got['bus'] == spec['expected_report']['boxes_per_class']['bus']
    assert job['report']['negatives'] == spec['expected_report']['label_states']['negative']
    assert world.default_traffic(since=world.mark, writes_only=False) == []
