"""cluster_eval_set: COCO-derived labels, strata and protected flags from a fake project API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from scripts.bench import cluster_eval_set as ces


if TYPE_CHECKING:
    from pathlib import Path

W, H = 640, 480


def _coco() -> dict[str, Any]:
    cats = [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'cat'}, {'id': 3, 'name': 'cow'}]
    images = [
        {'id': 10, 'file_name': 'a.jpg', 'width': W, 'height': H},
        {'id': 11, 'file_name': 'b.jpg', 'width': W, 'height': H},
    ]
    ann = [
        {'id': 1, 'image_id': 10, 'category_id': 1, 'bbox': [100, 100, 200, 100], 'iscrowd': 0},
        {'id': 2, 'image_id': 10, 'category_id': 2, 'bbox': [400, 300, 20, 20], 'iscrowd': 0},
        {'id': 3, 'image_id': 10, 'category_id': 3, 'bbox': [0, 0, 50, 50], 'iscrowd': 1},
        {'id': 4, 'image_id': 11, 'category_id': 3, 'bbox': [10, 10, 100, 100], 'iscrowd': 0},
        {'id': 5, 'image_id': 11, 'category_id': 1, 'bbox': [300, 10, 100, 100], 'iscrowd': 0},
    ]
    return {'categories': cats, 'images': images, 'annotations': ann}


def _crop(cid: str, path: str, px: tuple[float, float, float, float], **kw: Any) -> dict[str, Any]:
    x1, y1, x2, y2 = px
    return {
        'crop_id': cid,
        'image_id': path,
        'image_path': f'/data/{path}',
        'bbox_norm': [x1 / W, y1 / H, x2 / W, y2 / H],
        'cluster_id': 1,
        **kw,
    }


@pytest.fixture
def coco(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / 'gt.json'
    path.write_text(json.dumps(_coco()))
    return ces.load_coco(path)


def test_match_uses_the_best_overlapping_annotation_and_ignores_crowd(
    coco: dict[str, Any],
) -> None:
    name, iou, side = ces.match_class(_crop('c', 'a.jpg', (105, 100, 300, 200)), coco, 0.5)  # type: ignore[misc]
    assert name == 'car'
    assert iou > 0.9
    assert side == pytest.approx(100.0)
    assert ces.match_class(_crop('c', 'a.jpg', (0, 0, 50, 50)), coco, 0.5) is None  # crowd only
    assert ces.match_class(_crop('c', 'a.jpg', (500, 400, 600, 470)), coco, 0.5) is None
    assert ces.match_class(_crop('c', 'zzz.jpg', (105, 100, 300, 200)), coco, 0.5) is None


def test_build_labels_only_matched_crops_and_never_invents_a_label(coco: dict[str, Any]) -> None:
    crops = [
        _crop('car1', 'a.jpg', (100, 100, 300, 200)),
        _crop('person', 'a.jpg', (500, 400, 600, 470)),
        _crop('cow1', 'b.jpg', (10, 10, 110, 110), cluster_id=-1),
    ]
    labels, partition, manifest = ces.build(crops, coco)
    assert {r['item_id']: r['label'] for r in labels} == {'car1': 'car', 'cow1': 'cow'}
    assert [p['item_id'] for p in partition] == ['car1', 'person', 'cow1']
    assert partition[2]['cluster_id'] is None
    assert manifest['crops_without_annotation'] == 1
    assert manifest['unassigned_in_project'] == 1


def test_strata_priority_low_res_then_rare_then_similar(coco: dict[str, Any]) -> None:
    crops = [
        _crop('tiny', 'a.jpg', (400, 300, 420, 320)),  # cat, 20 px
        _crop('car1', 'a.jpg', (100, 100, 300, 200)),
        _crop('car2', 'b.jpg', (300, 10, 400, 110)),
        _crop('cow1', 'b.jpg', (10, 10, 110, 110)),
    ]
    labels, _, manifest = ces.build(crops, coco)
    by = {r['item_id']: r['stratum'] for r in labels}
    # classes by count: cat 1, cow 1, car 2 -> rarest third (1 class) is cat
    assert manifest['rare_classes'] == ['cat']
    assert by['tiny'] == 'low_res'
    assert by['car1'] == by['car2'] == 'similar_classes'
    assert by['cow1'] == 'similar_classes'
    assert ces.stratum('person', 80.0, set(), 2) == 'mixed_scene'
    assert ces.stratum('person', 80.0, set(), 1) == 'other'
    assert ces.stratum('cat', 80.0, {'cat'}, 1) == 'rare'


@pytest.mark.parametrize(
    'flag', ['test_holdout', 'label_locked', 'label_validated', 'class_validated']
)
def test_protected_follows_the_project_flags(flag: str) -> None:
    assert ces.is_protected({flag: True}) is True
    assert ces.is_protected({'import_ids': ['imp1']}) is True
    assert ces.is_protected({flag: False, 'import_ids': []}) is False


def test_no_matching_crop_refuses(coco: dict[str, Any]) -> None:
    with pytest.raises(ces.EvalSetError, match='refusing to invent'):
        ces.build([_crop('x', 'a.jpg', (500, 400, 600, 470))], coco)


def test_cmd_build_pages_through_the_api_read_only(tmp_path: Path) -> None:
    gt = tmp_path / 'gt.json'
    gt.write_text(json.dumps(_coco()))
    crops = [_crop(f'c{i}', 'a.jpg', (100, 100, 300, 200)) for i in range(ces.PAGE_SIZE + 3)]
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.params.get('page', '')))
        page = int(request.url.params['page'])
        chunk = crops[(page - 1) * ces.PAGE_SIZE : page * ces.PAGE_SIZE]
        return httpx.Response(200, json={'total': len(crops), 'crops': chunk})

    args = ces.build_parser().parse_args(
        [
            'build',
            '--api',
            'http://x',
            '--project',
            'p',
            '--coco-gt',
            str(gt),
            '--out-dir',
            str(tmp_path / 'o'),
        ]
    )
    assert ces.cmd_build(args, transport=httpx.MockTransport(handler)) == 0
    assert seen == [('GET', '1'), ('GET', '2')]
    lines = (tmp_path / 'o' / 'labels.jsonl').read_text().splitlines()
    assert len(lines) == ces.PAGE_SIZE + 3
    manifest = json.loads((tmp_path / 'o' / 'manifest.json').read_text())
    assert manifest['project'] == 'p'
    assert manifest['crops_labelled'] == ces.PAGE_SIZE + 3
