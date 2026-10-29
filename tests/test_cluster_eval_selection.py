"""Selection, jitter and pin logic of the cluster evaluation set (no network, no pixels)."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.datasets import cluster_eval_selection as sel, fetch_cluster_eval as fce


def _coco_data() -> dict[str, Any]:
    cats = [
        {'id': i, 'name': n} for i, n in enumerate(['car', 'cat', 'person', 'toaster', 'cow'], 1)
    ]
    images, anns = [], []
    for i in range(1, 61):
        images.append(
            {
                'id': i,
                'file_name': f'{i:012d}.jpg',
                'width': 640,
                'height': 480,
                'license': 1 + i % 2,
                'flickr_url': f'http://f/{i}',
            }
        )
        cls = [1, 2, 4, 5][i % 4]
        anns.append(
            {'id': i, 'image_id': i, 'category_id': cls, 'bbox': [10, 10, 200, 150], 'iscrowd': 0}
        )
        if i % 3 == 0:
            anns.append(
                {
                    'id': 100 + i,
                    'image_id': i,
                    'category_id': 3,
                    'bbox': [300, 10, 20, 20],
                    'iscrowd': 0,
                }
            )
        if i % 5 == 0:
            anns.append(
                {'id': 200 + i, 'image_id': i, 'category_id': 1, 'bbox': [0, 0, 5, 5], 'iscrowd': 1}
            )
    return {
        'images': images,
        'annotations': anns,
        'categories': cats,
        'licenses': [{'id': 1, 'name': 'Attribution License'}, {'id': 2, 'name': 'Non-commercial'}],
    }


def test_coco_images_keeps_only_allowed_licences_and_ignores_crowd() -> None:
    rows = sel.coco_images(_coco_data(), {'Attribution License'})
    assert rows
    assert all(r['license'] == 'Attribution License' for r in rows)
    assert all(r['image_id'] % 2 == 0 for r in rows)  # odd ids carry the NC licence (1 + i % 2)
    assert 'car' not in next(r for r in rows if r['image_id'] == 10)['classes']  # crowd box
    assert next(r for r in rows if r['image_id'] == 6)['small'] == 1  # the 20 px person


def test_rarest_coco_classes_by_train_instances() -> None:
    train = _coco_data()
    rare = sel.rarest_coco_classes(train, 2)
    assert [(r['class'], r['train2017_instances']) for r in rare] == [('car', 15), ('cat', 15)]


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coco = sel.coco_images(_coco_data(), {'Attribution License', 'Non-commercial'})
    boxes = {
        f'oi{i}': [
            {
                'name': 'cat' if i % 2 else 'lion',
                'x_min': 0.1,
                'x_max': 0.6,
                'y_min': 0.1,
                'y_max': 0.6,
            },
            {'name': 'person', 'x_min': 0.7, 'x_max': 0.9, 'y_min': 0.1, 'y_max': 0.4},
        ]
        for i in range(30)
    }
    meta = {
        k: {'split': 'test', 'license': 'by', 'author': 'a', 'original_url': 'u'} for k in boxes
    }
    return coco, sel.openimages_images(boxes, meta)


def test_selection_is_seeded_disjoint_and_grouped() -> None:
    coco, oi = _rows()
    a = sel.select_base(coco, oi, {'toaster'}, seed=1, scale=0.1)
    b = sel.select_base(coco, oi, {'toaster'}, seed=1, scale=0.1)
    c = sel.select_base(coco, oi, {'toaster'}, seed=2, scale=0.1)
    names = [r['file_name'] for r in a]
    assert names == [r['file_name'] for r in b]
    assert names != [r['file_name'] for r in c]
    assert len(names) == len(set(names))
    assert {r['group'] for r in a} >= {'similar', 'rare'}
    assert all(r['primary'] in sel.SIMILAR for r in a if r['group'] == 'similar')
    assert all(
        'toaster' in r['classes'] for r in a if r['group'] == 'rare' and r['source'] == 'coco'
    )
    assert all(r['file_name'].startswith('oi_') for r in a if r['source'] == 'openimages')


def test_near_duplicate_plan_is_reproducible_and_bounded() -> None:
    coco, oi = _rows()
    base = sel.select_base(coco, oi, set(), seed=3, scale=0.2)
    by_name = {r['file_name']: r for r in base}
    plan = sel.select_near_duplicates(base, seed=3, sources=5)
    assert plan == sel.select_near_duplicates(base, seed=3, sources=5)
    assert len(plan) == 5
    for nd in plan:
        assert sum(by_name[nd['parent']]['classes'].values()) <= sel.MAX_SOURCE_BOXES
        assert len(nd['variants']) == sel.VARIANTS
        for v in nd['variants']:
            assert v['file_name'].startswith('nd_')
            assert all(0 <= m <= sel.MAX_MARGIN for m in v['margins'])
            assert sel.QUALITY_RANGE[0] <= v['quality'] <= sel.QUALITY_RANGE[1]


def test_jitter_window_and_annotation_shift() -> None:
    window = sel.jitter_window(1000, 500, [0.05, 0.1, 0.02, 0.0])
    assert window == (50, 50, 980, 500)
    anns = [
        {'bbox': [100, 100, 200, 100], 'iscrowd': 0},  # inside
        {'bbox': [0, 0, 100, 100], 'iscrowd': 0},  # mostly cut away by the left/top margin
        {'bbox': [900, 300, 100, 100], 'iscrowd': 0},  # clipped on the right, 80 % kept
    ]
    out = sel.shift_annotations(anns, window)
    assert [a['bbox'] for a in out] == [[50, 50, 200, 100], [850, 250, 80, 100]]


def test_pin_hash_covers_content() -> None:
    row = {'source': 'coco', 'image_id': 1, 'file_name': 'a.jpg', 'sha256': 'x'}
    pin = sel.build_pin([row], [], [{'class': 'toaster', 'train2017_instances': 3}], 1, 1.0, 'h')
    assert pin['manifest_sha256'] == sel.pin_hash(pin)
    assert 'toaster' in pin['rare_classes']
    assert 'lion' in pin['rare_classes']
    other = sel.build_pin([{**row, 'sha256': 'y'}], [], [], 1, 1.0, 'h')
    assert other['manifest_sha256'] != pin['manifest_sha256']


def test_open_images_parsing_filters_group_depiction_licence_and_rotation() -> None:
    desc = 'LabelName,DisplayName\n' + ''.join(f'/m/{n},{n}\n' for n in sel.OI_CLASSES)
    mids = fce.oi_class_ids(desc)
    assert mids['/m/Cattle'] == 'cow'
    head = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'
    rows = (
        'a,x,/m/Cat,1,0,.5,0,.5,0,0,0,0,0\n'
        'a,x,/m/Cat,1,0,.5,0,.5,0,0,1,0,0\n'
        'b,x,/m/Cat,1,0,.5,0,.5,0,0,0,1,0\n'
        'c,x,/m/Unknown,1,0,.5,0,.5,0,0,0,0,0\n'
    )
    boxes = fce.oi_boxes(head + rows, mids)
    assert list(boxes) == ['a']
    assert len(boxes['a']) == 1
    images = (
        'ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Rotation\n'
        f'a,t,u,l,{fce.ALLOWED_LICENSE_URL},p,Ann,\n'
        f'b,t,u,l,{fce.ALLOWED_LICENSE_URL},p,Bob,90.0\n'
        'c,t,u,l,https://creativecommons.org/licenses/by-nc/2.0/,p,Cy,\n'
    )
    assert list(fce.oi_metadata(images, 'test')) == ['a']


def test_missing_oi_class_name_refuses() -> None:
    with pytest.raises(fce.FetchError, match='not found'):
        fce.oi_class_ids('LabelName,DisplayName\n/m/1,Car\n')


def test_excluded_names_are_never_drawn() -> None:
    coco, oi = _rows()
    first = sel.select_base(coco, oi, {'toaster'}, seed=1, scale=0.2)
    banned = frozenset(r['file_name'] for r in first)
    second = sel.select_base(coco, oi, {'toaster'}, seed=1, scale=0.2, exclude=banned)
    assert second
    assert not banned & {r['file_name'] for r in second}


def test_backfill_hands_a_coco_shortfall_to_open_images_of_the_same_group() -> None:
    coco, oi = _rows()
    all_coco_similar = frozenset(
        r['file_name'] for r in coco if r['primary'] in sel.SIMILAR
    )  # COCO has nothing left to draw for the similar group
    plain = sel.select_base(coco, oi, {'toaster'}, seed=1, scale=0.1, exclude=all_coco_similar)
    topped = sel.select_base(
        coco, oi, {'toaster'}, seed=1, scale=0.1, exclude=all_coco_similar, backfill=True
    )

    def similar_oi(rows: list[dict[str, Any]]) -> int:
        return sum(r['source'] == 'openimages' and r['group'] == 'similar' for r in rows)

    assert similar_oi(topped) > similar_oi(plain)
    assert not any(r['source'] == 'coco' and r['group'] == 'similar' for r in topped)


def test_excluded_names_reads_pins_lists_and_split_manifests(tmp_path: Any) -> None:
    import json

    pin = tmp_path / 'pin.json'
    pin.write_text(
        json.dumps(
            {
                'images': [{'file_name': 'a.jpg'}],
                'near_duplicates': [{'parent': 'a.jpg', 'variants': [{'file_name': 'nd_a_1.jpg'}]}],
            }
        )
    )
    rows = tmp_path / 'rows.json'
    rows.write_text(json.dumps([{'file_name': 'b.jpg'}]))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'calibration': ['c.jpg'], 'heldout': ['d.jpg']}))
    assert fce.excluded_names([pin, rows, split]) == {
        'a.jpg',
        'nd_a_1.jpg',
        'b.jpg',
        'c.jpg',
        'd.jpg',
    }
