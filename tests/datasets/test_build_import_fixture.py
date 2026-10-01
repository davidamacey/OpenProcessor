"""build_import_fixture.build_variants against faked COCO annotations.

The strongest check here is not the builder's own arithmetic: every layout it
writes is scanned by the REAL dataset scanner, and the scanner's issues,
label states and per-class box counts must equal the ``FIXTURE.json``
expectations the builder derived independently.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from scripts.datasets import build_import_fixture as b
from src.services.curation.dataset_import.options import DatasetSource
from src.services.curation.dataset_import.regions import ParentCandidate, attach_region_boxes
from src.services.curation.dataset_import.scan import scan_dataset


CATS = [
    {'id': 1, 'name': 'person'},
    {'id': 3, 'name': 'car'},
    {'id': 6, 'name': 'bus'},
    {'id': 8, 'name': 'truck'},
]
N_POSITIVE = 40
N_NEGATIVE = 6


def _annotations() -> tuple[dict, dict]:
    images, anns = [], []
    aid = 1

    def add(image_id: int, cat: int, bbox: list[float], crowd: int = 0) -> None:
        nonlocal aid
        anns.append(
            {
                'id': aid,
                'image_id': image_id,
                'category_id': cat,
                'bbox': bbox,
                'iscrowd': crowd,
            }
        )
        aid += 1

    for i in range(1, N_POSITIVE + 1):
        images.append({'id': i, 'file_name': f'{i:012d}.jpg', 'width': 200, 'height': 160})
        if i % 4 == 1:
            add(i, 8, [10, 20, 120, 100])  # truck
        elif i % 4 == 2:
            add(i, 6, [20, 30, 150, 90])  # bus
        else:
            add(i, 3, [20, 40, 100, 90])  # car (tall enough for wheels)
            if i % 4 == 3:
                add(i, 3, [130, 10, 60, 30])  # a short car: no wheels
        if i % 7 == 0:
            add(i, 3, [0, 0, 30, 30], crowd=1)
        add(i, 1, [150, 120, 20, 30])  # a non-target box must never be imported
    negatives = []
    for j in range(N_NEGATIVE):
        image_id = 1000 + j
        images.append(
            {'id': image_id, 'file_name': f'{image_id:012d}.jpg', 'width': 200, 'height': 160}
        )
        negatives.append(image_id)
        add(image_id, 1, [10, 10, 40, 60])  # kept in the GT, still a negative for the 3 classes
    return {'images': images, 'annotations': anns, 'categories': CATS}, {
        'negative_image_ids': negatives,
        'seed': 1,
    }


def _writer(file_name: str, dest: Path, width: int, height: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (width, height), 'gray').save(dest, format='JPEG')


@pytest.fixture(scope='module')
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp('fixture')
    annotations, meta = _annotations()
    return out, b.build_variants(annotations, meta, out, _writer)


def _scan(out: Path, variant: str):
    fmt = 'coco' if variant == 'coco' else 'auto'
    return scan_dataset(
        DatasetSource(path=str(out / variant), format=fmt), path_guard=lambda _p: True
    )


@pytest.mark.parametrize('variant', ['yolo', 'coco', 'yolo_region', 'yolo_region_only'])
def test_scanner_agrees_with_fixture_expectations(built: tuple[Path, dict], variant: str) -> None:
    out, fixture = built
    spec = fixture['variants'][variant]
    scan = _scan(out, variant)
    issues = {i.code: i.count for i in scan.issues.issues()}
    expected = {c: n for c, n in spec['expected_issues'].items() if n}
    assert issues == expected
    states: Counter[str] = Counter(e.label_state for e in scan.entries)
    assert dict(states) == {k: v for k, v in spec['expected_report']['label_states'].items() if v}
    per_split = Counter(e.split for e in scan.entries)
    assert dict(per_split) == {
        k: v for k, v in spec['expected_report']['images_per_split'].items() if v
    }
    assert scan.class_box_counts == spec['expected_report']['boxes_per_class']


def test_yolo_names_are_not_registry_order_and_carry_a_synonym(built: tuple[Path, dict]) -> None:
    out, fixture = built
    assert fixture['variants']['yolo']['names'] == {
        '0': 'truck',
        '1': 'Car',
        '2': 'automobile',
        '3': 'bus',
    }
    counts = _scan(out, 'yolo').class_box_counts
    # both spellings occur: odd COCO ids become 'automobile', even ones 'Car'
    assert counts['Car'] > 0
    assert counts['automobile'] > 0


def test_all_six_injected_problems_present_once(built: tuple[Path, dict]) -> None:
    _, fixture = built
    assert set(fixture['variants']['yolo']['expected_issues'].values()) == {1}
    assert len(fixture['variants']['yolo']['expected_issues']) == 6


def test_split_is_stable_70_15_15_by_sha1() -> None:
    assert b.split_of(7) == b.split_of(7)
    counts = Counter(b.split_of(i) for i in range(1, 2001))
    assert 1300 < counts['train'] < 1500
    assert 200 < counts['val'] < 400
    assert 200 < counts['test'] < 400


def test_non_target_boxes_never_leak_into_any_variant(built: tuple[Path, dict]) -> None:
    out, _ = built
    assert 'person' not in _scan(out, 'coco').class_box_counts
    gt = [json.loads(p.read_text()) for p in (out / 'coco/annotations').glob('*.json')]
    assert all(c['name'] != 'person' for g in gt for c in g['categories'])


def test_region_variant_wheels_attach_by_containment(built: tuple[Path, dict]) -> None:
    out, fixture = built
    spec = fixture['variants']['yolo_region']
    scan = _scan(out, 'yolo_region')
    attached = standalone = 0
    for entry in scan.entries:
        parents = [
            ParentCandidate(key=str(i), bbox_norm=bx.bbox_norm, class_name='car')
            for i, bx in enumerate(entry.boxes)
            if bx.dataset_class == 'car'
        ]
        wheels = [bx for bx in entry.boxes if bx.dataset_class == 'wheel']
        res = attach_region_boxes(parents, wheels)
        attached += len(res.attachments)
        standalone += len(res.standalone)
    assert (attached, standalone) == (spec['attachments'], spec['standalone'])
    assert attached > 0
    assert standalone == 1
    assert 'SYNTHETIC' in spec['synthetic_geometry']


def test_region_only_variant_has_no_car_rows(built: tuple[Path, dict]) -> None:
    out, fixture = built
    assert 'car' not in _scan(out, 'yolo_region_only').class_box_counts
    assert _scan(out, 'yolo_region').class_box_counts['car'] > 0
    a = fixture['variants']['yolo_region']['expected_report']['boxes_per_class']['wheel']
    b_ = fixture['variants']['yolo_region_only']['expected_report']['boxes_per_class']['wheel']
    assert a == b_


def test_negatives_have_empty_labels_in_every_yolo_layout(built: tuple[Path, dict]) -> None:
    out, _ = built
    for variant in ('yolo', 'yolo_region', 'yolo_region_only'):
        negs = [e for e in _scan(out, variant).entries if e.source_stem.startswith('0000000010')]
        assert len(negs) == N_NEGATIVE
        assert all(e.label_state == 'negative' for e in negs)


def test_build_is_deterministic(tmp_path: Path) -> None:
    annotations, meta = _annotations()
    a = b.build_variants(annotations, meta, tmp_path / 'a', _writer)
    c = b.build_variants(annotations, meta, tmp_path / 'c', _writer)
    assert a == c
    la = sorted(p.relative_to(tmp_path / 'a').as_posix() for p in (tmp_path / 'a').rglob('*.txt'))
    lc = sorted(p.relative_to(tmp_path / 'c').as_posix() for p in (tmp_path / 'c').rglob('*.txt'))
    assert la == lc
    assert all((tmp_path / 'a' / p).read_text() == (tmp_path / 'c' / p).read_text() for p in la)


def test_only_tall_cars_get_wheels_and_wheel_count_is_exact(built: tuple[Path, dict]) -> None:
    _, fixture = built
    annotations, _ = _annotations()
    tall = [
        a
        for a in annotations['annotations']
        if a['category_id'] == 3 and not a['iscrowd'] and a['bbox'][3] >= b.MIN_CAR_HEIGHT_PX
    ]
    short = [a for a in annotations['annotations'] if a['category_id'] == 3 and a['bbox'][3] < 48]
    assert tall
    assert short  # the fixture does contain cars that must not get wheels
    spec = fixture['variants']['yolo_region']
    assert spec['attachments'] == 2 * len(tall)
    assert spec['expected_report']['boxes_per_class']['wheel'] == 2 * len(tall) + spec['standalone']


#: What a COCO category must be called in the YOLO variant, written out here
#: rather than derived from the builder's own name table.
EXPECTED_YOLO_NAMES = {'truck': {'truck'}, 'bus': {'bus'}, 'car': {'Car', 'automobile'}}


def test_coco_class_names_land_on_the_matching_yolo_names(built: tuple[Path, dict]) -> None:
    out, fixture = built
    names: dict[int, str] = {}
    for line in (out / 'yolo/data.yaml').read_text().splitlines():
        key, _, value = line.strip().partition(':')
        if key.isdigit():
            names[int(key)] = value.strip().strip('\'"')
    assert set(names.values()) == {'truck', 'Car', 'automobile', 'bus'}
    damaged = {
        Path(v).stem
        for k, v in fixture['variants']['yolo']['injected'].items()
        if k != 'orphan_label'
    }
    coco_name = {1: 'truck', 2: 'bus'}  # image id % 4; every other id carries only cars
    checked: Counter[str] = Counter()
    for image_id in range(1, N_POSITIVE + 1):
        stem = f'{image_id:012d}'
        if stem in damaged:
            continue
        label = next((out / 'yolo/labels').glob(f'*/{stem}.txt'))
        wanted = EXPECTED_YOLO_NAMES[coco_name.get(image_id % 4, 'car')]
        for row in label.read_text().splitlines():
            assert names[int(row.split()[0])] in wanted, (image_id, row)
            checked[coco_name.get(image_id % 4, 'car')] += 1
    assert all(checked[c] > 0 for c in EXPECTED_YOLO_NAMES), checked


def test_a_file_name_with_a_path_is_refused(tmp_path: Path) -> None:
    annotations, meta = _annotations()
    annotations['images'][0]['file_name'] = '../../evil.jpg'
    written: list[Path] = []

    def writer(_name: str, dest: Path, _w: int, _h: int) -> None:
        written.append(dest)

    from scripts.datasets._common import FetchError

    with pytest.raises(FetchError, match='non-basename'):
        b.build_variants(annotations, meta, tmp_path / 'out', writer)
    assert written == []
