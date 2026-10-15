"""``fetch_coco_subset.run`` end to end on a synthetic COCO split: every file it
writes (selection summary, ground truth, attribution, side sets, pin) is pinned,
so the orchestration can be restructured without drifting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.datasets import coco_fetch, fetch_coco_subset as f


LICENSES: list[dict[str, Any]] = [
    {'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'},
    {'id': 4, 'name': 'Attribution License'},
    {'id': 5, 'name': 'Attribution-ShareAlike License'},
]
CATEGORIES: list[dict[str, Any]] = [
    {'id': 1, 'name': 'person'},
    {'id': 3, 'name': 'car'},
    {'id': 18, 'name': 'dog'},
]


def _image(image_id: int, license_id: int) -> dict[str, Any]:
    return {
        'id': image_id,
        'file_name': f'{image_id:012d}.jpg',
        'license': license_id,
        'width': 640,
        'height': 480,
        'flickr_url': f'http://example.invalid/{image_id}',
        'coco_url': f'http://example.invalid/coco/{image_id}',
    }


def _ann(ann_id: int, image_id: int, category_id: int, *, crowd: int = 0) -> dict[str, Any]:
    return {
        'id': ann_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': [10, 10, 300, 300],
        'iscrowd': crowd,
    }


def _split(first: int) -> dict[str, Any]:
    """Six cars, six dogs, two person-only, four empty, one NC car; one crowd car."""
    images, anns = [], []
    nxt, ann_id = first, first * 10
    for category_id, count in ((3, 6), (18, 6), (1, 2)):
        for _ in range(count):
            images.append(_image(nxt, 4 if nxt % 2 else 5))
            anns.append(_ann(ann_id, nxt, category_id))
            nxt, ann_id = nxt + 1, ann_id + 1
    images.append(_image(nxt, 1))  # disallowed license, car
    anns.append(_ann(ann_id, nxt, 3))
    nxt, ann_id = nxt + 1, ann_id + 1
    anns.append(_ann(ann_id, first, 3, crowd=1))
    for _ in range(4):
        images.append(_image(nxt, 4))
        nxt += 1
    return {'images': images, 'annotations': anns, 'categories': CATEGORIES, 'licenses': LICENSES}


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    val, train = _split(100), _split(1000)
    monkeypatch.setattr(f, 'ensure_annotations', lambda _cache: (val, train))
    downloaded: list[str] = []

    def _fake_download(url: str, dest: Path, **_kw: Any) -> None:
        downloaded.append(url.rsplit('/', 2)[-2] + '/' + dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(dest.name.encode())

    monkeypatch.setattr(coco_fetch, 'download', _fake_download)
    return downloaded


def _tree(out: Path) -> dict[str, str]:
    return {
        str(p.relative_to(out)): p.read_text() if p.suffix in {'.json', '.csv'} else '<bytes>'
        for p in sorted(out.rglob('*'))
        if p.is_file()
    }


def _run(tmp_path: Path, *argv: str) -> tuple[dict[str, Any], dict[str, str]]:
    args = f.build_parser().parse_args(
        ['--out', str(tmp_path / 'out'), '--cache-dir', str(tmp_path / 'cache'), *argv]
    )
    summary = f.run(args)
    return summary, _tree(tmp_path / 'out')


SCENARIOS: dict[str, list[str]] = {
    'full': [
        '--classes', 'car,dog', '--per-class', '4', '--seed', '11', '--licenses', 'by,by-sa',
        '--side-sets', 'upload=2,later=3', '--negatives', '2', '--manifest', '{pin}',
    ],
    'val_only_n': ['--classes', 'car,dog', '--n', '6', '--seed', '3', '--val-only', '--licenses', 'by'],
    'negatives_keep_other_annotations': [
        '--classes', 'car', '--per-class', '2', '--val-only', '--licenses', 'by,by-sa',
        '--negatives', '8', '--seed', '5',
    ],
    'skip_download': ['--classes', 'dog', '--per-class', '3', '--licenses', 'by', '--skip-download'],
}  # fmt: skip
GOLDEN = Path(__file__).parent / 'golden' / 'fetch_coco_run.json'


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_run_outputs_match_the_pinned_golden(
    scenario: str, tmp_path: Path, world: list[str]
) -> None:
    argv = [a.format(pin=tmp_path / 'pin.json') for a in SCENARIOS[scenario]]
    summary, tree = _run(tmp_path, *argv)
    pin = tmp_path / 'pin.json'
    actual = {
        'summary': summary,
        'tree': tree,
        'downloaded': sorted(world),
        'pin': json.loads(pin.read_text()) if pin.exists() else None,
    }
    golden = json.loads(GOLDEN.read_text())[scenario]
    assert actual == golden
