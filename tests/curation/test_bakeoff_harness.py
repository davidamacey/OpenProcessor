"""Tests for the genericized bake-off harness (BakeoffProfile, converters, runner).

Covers the parts of ``scripts/curation/bakeoff/`` that decide *what* is
measured: profile resolution, CLI-vs-profile precedence in ``run``, the
converter registry + generic YOLO writer in ``datasets``, ranking in
``compare`` and profile propagation in ``bakeoff_runner``. Also guards the
harness core against domain vocabulary creeping back in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scripts.curation.bakeoff import (
    bakeoff_runner,
    class_map,
    compare,
    datasets,
    metrics,
    run,
    scoring,
)
from scripts.curation.bakeoff.backends import registry
from scripts.curation.bakeoff.backends.base import Detection
from scripts.curation.bakeoff.backends.yolo_post import decode_yolo26_e2e, decode_yolo_v11, nms
from scripts.curation.bakeoff.dataset import YoloTestSet
from scripts.curation.bakeoff.freeze import test_sha as label_sha
from scripts.curation.bakeoff.profile import (
    GENERIC_PROFILE,
    BakeoffProfile,
    resolve_baselines_path,
    resolve_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / 'scripts/curation/bakeoff'


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith('OP_BAKEOFF_PROFILE'):
            monkeypatch.delenv(key)


def _img(path: Path, w: int = 100, h: int = 50) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
    return path


# --- BakeoffProfile -----------------------------------------------------------


def test_generic_profile_is_domain_neutral() -> None:
    p = resolve_profile(None)
    assert p == GENERIC_PROFILE
    assert p.description == 'All classes present in the eval split'
    assert p.class_filter == ()
    assert p.class_names == ()
    assert p.imgsz == 640
    assert p.rank_metric == 'map_50_95'
    assert p.converter_modules == ()
    assert p.backend_modules == ()
    assert p.context_class_ids == ()
    assert p.triton_model == ''
    assert not hasattr(p, 'target_class_id')
    assert not hasattr(p, 'target_class_name')


def test_profile_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CLASS_FILTER', 'box, label')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CLASS_NAMES', 'pallet,box,label')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '0, 1')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_OP_CONF', '0.4')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_BACKEND_MODULES', 'my.backends')
    p = resolve_profile(None)
    assert p.class_filter == ('box', 'label')
    assert p.class_names == ('pallet', 'box', 'label')
    assert p.context_class_ids == (0, 1)
    assert p.op_conf == pytest.approx(0.4)
    assert p.backend_modules == ('my.backends',)


def test_profile_json_path_and_unknown_name(tmp_path: Path) -> None:
    f = tmp_path / 'mine.json'
    f.write_text(json.dumps({'name': 'widgets', 'class_filter': ['widget']}))
    assert resolve_profile(str(f)).class_filter == ('widget',)
    with pytest.raises(ValueError, match='unknown bake-off profile'):
        resolve_profile('no_such_profile')
    with pytest.raises(ValueError, match='unknown BakeoffProfile field'):
        BakeoffProfile.from_dict({'name': 'x', 'target_class_id': 0})
    with pytest.raises(ValueError, match='rank_metric'):
        BakeoffProfile(name='x', rank_metric='ap_small')


def test_profile_baselines_path_resolves_against_repo() -> None:
    default = Path('/nonexistent/default.json')
    assert resolve_baselines_path(GENERIC_PROFILE, default) == default
    rel = BakeoffProfile(name='x', baselines_path='scripts/curation/bakeoff/baselines.json')
    assert resolve_baselines_path(rel, default) == HARNESS / 'baselines.json'


def test_default_baseline_registry_is_empty() -> None:
    reg = json.loads((HARNESS / 'baselines.json').read_text())
    assert reg['baselines'] == []  # previous runs are the baselines; quant variants are produced


# --- run.py: profile vs CLI precedence ------------------------------------------


def _resolve(*argv: str):
    return run.resolve_args(run.build_parser().parse_args(list(argv)))


def test_run_generic_profile_has_no_hardcoded_context_classes() -> None:
    args, profile = _resolve('--dataset', '/x')
    assert profile.name == 'generic'
    assert args.primary_classes == ''
    assert run.parse_class_ids(args.primary_classes) == ()
    assert args.backend == 'ultralytics'
    assert args.imgsz == 640
    assert args.backend_options == {}
    assert args.class_map is None


def test_run_v2_flags_parse_json() -> None:
    args, _ = _resolve(
        '--dataset',
        '/x',
        '--class-map-json',
        '{"0": 37, "1": 38}',
        '--backend-options-json',
        '{"providers": "CPUExecutionProvider", "variant": "x"}',
        '--train-test-overlap-json',
        '{"n_images": 2, "fraction": 0.5}',
        '--model-key',
        'run:abc',
    )
    assert args.class_map == {0: 37, 1: 38}
    assert args.ort_providers == 'CPUExecutionProvider'  # known option applied to its flag
    assert args.backend_options == {'providers': 'CPUExecutionProvider', 'variant': 'x'}
    assert args.train_test_overlap == {'n_images': 2, 'fraction': 0.5}
    assert args.display_name == 'run:abc'  # defaults to the model key
    args, _ = _resolve('--dataset', '/x', '--class-map-json', 'null')
    assert args.class_map is None
    for removed in ('--gt-class-id', '--gt-class-name', '--pred-class-id', '--lpdnet-variant'):
        with pytest.raises(SystemExit):
            _resolve('--dataset', '/x', removed, '0')


def test_run_triton_requires_a_model() -> None:
    with pytest.raises(SystemExit, match='triton'):
        _resolve('--backend', 'triton', '--dataset', '/x')
    args, _ = _resolve('--backend', 'triton', '--triton-model', 'my_det')
    assert args.triton_model == 'my_det'


def test_run_unknown_profile_exits() -> None:
    with pytest.raises(SystemExit, match='unknown bake-off profile'):
        _resolve('--profile', 'nope')


# --- datasets.py: registry + generic writer ------------------------------------


def test_datasets_has_no_hardcoded_class_constant() -> None:
    assert not hasattr(datasets, 'LPR_CLASS_ID')
    src = (HARNESS / 'datasets.py').read_text()
    assert 'license_plate' not in src


def test_builtin_converters_are_generic_only() -> None:
    names = set(datasets.available_converters())
    assert {'voc', 'yolo'} <= names
    assert all(
        c.example_for is None
        for c in datasets.available_converters().values()
        if c.name in {'voc', 'yolo'}
    )


def test_writer_data_yaml_uses_profile_class_names(tmp_path: Path) -> None:
    prof = BakeoffProfile(name='shop', class_names=('pallet', 'box'))
    w = datasets.YoloWriter.for_profile(tmp_path, prof, split='test')
    w.write_data_yaml()
    yaml = (tmp_path / 'data.yaml').read_text()
    assert 'nc: 2\n' in yaml
    assert '  0: pallet\n  1: box\n' in yaml
    with pytest.raises(ValueError, match='outside nc'):
        w.write('a', _img(tmp_path / 'src/a.png'), [(5, 0.5, 0.5, 0.1, 0.1)])


def test_writer_single_class_collapses_to_zero(tmp_path: Path) -> None:
    w = datasets.YoloWriter(tmp_path, ('thing',))
    assert not hasattr(w, 'target_class_id')
    assert (w.map_class_id(7), w.map_class_name('anything')) == (0, 0)
    with pytest.raises(TypeError):
        datasets.YoloWriter(tmp_path, ('thing',), target_class_id=0)  # type: ignore[call-arg]


def _voc(path: Path, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    objs = ''.join(
        f'<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>'
        f'<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>'
        for n, b in objects
    )
    path.write_text(
        f'<annotation><size><width>100</width><height>50</height></size>{objs}</annotation>'
    )


def test_voc_converter_maps_names_multiclass(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    img = _img(src / 'f1.png')
    _voc(
        img.with_suffix('.xml'),
        [('Box', (0, 0, 50, 50)), ('pallet', (50, 0, 100, 50)), ('cat', (10, 10, 20, 20))],
    )
    prof = BakeoffProfile(name='shop', class_names=('pallet', 'box'))
    out = tmp_path / 'out'
    assert datasets.convert('voc', src, out, prof) == 1
    rows = sorted((out / 'labels/test/f1.txt').read_text().split('\n')[:-1])
    assert [r.split()[0] for r in rows] == ['0', '1']  # 'cat' dropped
    assert (out / 'images/test/f1.png').is_symlink()


def test_voc_converter_collapses_single_class(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    img = _img(src / 'f1.png')
    _voc(img.with_suffix('.xml'), [('anything', (0, 0, 50, 50))])
    out = tmp_path / 'out'
    datasets.convert('voc', src, out, GENERIC_PROFILE)
    line = (out / 'labels/test/f1.txt').read_text().split()
    assert line[0] == '0'
    assert [float(v) for v in line[1:]] == pytest.approx([0.25, 0.5, 0.5, 1.0])
    assert '  0: object\n' in (out / 'data.yaml').read_text()


def test_yolo_passthrough_keeps_or_drops_ids(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    _img(src / 'images/f1.png')
    (src / 'labels').mkdir(parents=True)
    (src / 'labels/f1.txt').write_text('1 0.5 0.5 0.2 0.2\n7 0.1 0.1 0.1 0.1\nbad line\n')
    prof = BakeoffProfile(name='two', class_names=('a', 'b'))
    out = tmp_path / 'out'
    datasets.convert('yolo', src, out, prof)
    assert (out / 'labels/test/f1.txt').read_text() == '1 0.500000 0.500000 0.200000 0.200000\n'


def test_unknown_format_error_names_registry() -> None:
    with pytest.raises(ValueError, match='unknown dataset format'):
        datasets.get_converter('no-such-format')


# --- multi-class decode / NMS ------------------------------------------------------

# A [1, 5, 6] single-class head and the exact (boxes, scores) the pre-multi-class
# decoder produced for it (captured before the change; scale 0.5, pad (10, 20)).
_NC1_HEAD = np.array(
    [
        [
            [0.5, 0.25, 0.75, 0.1, 0.9, 0.3],
            [0.5, 0.5, 0.25, 0.2, 0.1, 0.8],
            [0.2, 0.1, 0.3, 0.05, 0.4, 0.2],
            [0.2, 0.3, 0.1, 0.05, 0.2, 0.4],
            [0.9, 0.0005, 0.6, 0.3, 0.002, 0.7],
        ]
    ],
    dtype=np.float32,
)
_NC1_GOLDEN_BOXES = [
    [60.0, 40.0, 100.0, 80.0],
    [100.0, 0.0, 160.0, 20.0],
    [-5.0, -5.0, 5.0, 5.0],
    [120.0, -40.0, 200.0, 0.0],
    [20.0, 80.0, 60.0, 160.0],
]
_NC1_GOLDEN_SCORES = [0.9, 0.6, 0.3, 0.002, 0.7]


def _decode(head: np.ndarray):
    return decode_yolo_v11(
        head, scale=0.5, pad=(10, 20), input_size=100, conf_thresh=0.001, coords_normalized=True
    )


def test_yolo_decode_multiclass_head() -> None:
    boxes, scores, class_ids = _decode(_NC1_HEAD)
    np.testing.assert_allclose(boxes, _NC1_GOLDEN_BOXES, rtol=1e-6)
    np.testing.assert_allclose(scores, _NC1_GOLDEN_SCORES, rtol=1e-6)
    assert class_ids.tolist() == [0, 0, 0, 0, 0]
    # Same head transposed ([1, N, 5]) decodes identically.
    boxes_t, scores_t, _ = _decode(_NC1_HEAD.transpose(0, 2, 1))
    np.testing.assert_allclose(boxes_t, _NC1_GOLDEN_BOXES, rtol=1e-6)
    np.testing.assert_allclose(scores_t, _NC1_GOLDEN_SCORES, rtol=1e-6)

    # nc = 3: score = max over the class rows, class = argmax.
    cls_rows = np.array(
        [
            [0.1, 0.7, 0.0005, 0.2, 0.3, 0.1],
            [0.9, 0.2, 0.0001, 0.1, 0.6, 0.2],
            [0.3, 0.1, 0.0002, 0.8, 0.1, 0.95],
        ],
        dtype=np.float32,
    )
    head3 = np.concatenate([_NC1_HEAD[:, :4], cls_rows[None]], axis=1)  # [1, 7, 6]
    # Real heads have thousands of anchors (N >> 4 + nc); pad to 8 anchors with
    # sub-threshold ones so the channel axis is the shorter one, as in practice.
    head3 = np.concatenate([head3, np.zeros((1, 7, 2), dtype=np.float32)], axis=2)
    boxes, scores, class_ids = _decode(head3)
    assert class_ids.tolist() == [1, 0, 2, 1, 2]
    assert scores.tolist() == pytest.approx([0.9, 0.7, 0.8, 0.6, 0.95])
    assert boxes[0].tolist() == pytest.approx(_NC1_GOLDEN_BOXES[0])
    _, _, cls_t = _decode(head3.transpose(0, 2, 1))
    assert cls_t.tolist() == [1, 0, 2, 1, 2]
    # A known class count disambiguates a head with fewer anchors than channels.
    _, _, cls_n = decode_yolo_v11(
        head3[:, :, :6],
        scale=0.5,
        pad=(10, 20),
        input_size=100,
        conf_thresh=0.001,
        coords_normalized=True,
        num_classes=3,
    )
    assert cls_n.tolist() == [1, 0, 2, 1, 2]


def test_decode_yolo26_e2e_keeps_class_column() -> None:
    out = np.array(
        [[[10, 20, 30, 40, 0.9, 3], [0, 0, 5, 5, 0.0001, 1], [50, 60, 70, 80, 0.5, 7]]],
        dtype=np.float32,
    )
    boxes, scores, class_ids = decode_yolo26_e2e(out, scale=0.5, pad=(10, 0), conf_thresh=0.001)
    assert class_ids.tolist() == [3, 7]
    assert class_ids.dtype.kind == 'i'
    assert scores.tolist() == pytest.approx([0.9, 0.5])
    assert boxes[0].tolist() == pytest.approx([0.0, 40.0, 40.0, 80.0])


def test_nms_is_per_class() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    # Class-agnostic: the two lower-scored overlapping boxes are suppressed.
    assert nms(boxes, scores, 0.5) == [0]
    # Per class: box 1 is another class, so it survives; box 2 is suppressed by box 0.
    assert nms(boxes, scores, 0.5, class_ids=np.array([0, 1, 0])) == [0, 1]


# --- multi-class dataset, class mapping, metrics -----------------------------------------


def _mc_dataset(
    root: Path,
    labels: dict[str, list[tuple[int, float, float, float, float]]],
    names: list[str],
    *,
    fill: dict[str, int] | None = None,
) -> Path:
    """A tiny YOLO test split: 100x100 images, ``labels[stem]`` rows, ``data.yaml`` names."""
    for stem, rows in labels.items():
        img = root / 'images/test' / f'{stem}.png'
        img.parent.mkdir(parents=True, exist_ok=True)
        value = (fill or {}).get(stem, 0)
        cv2.imwrite(str(img), np.full((100, 100, 3), value, dtype=np.uint8))
        lbl = root / 'labels/test' / f'{stem}.txt'
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(''.join(f'{c} {cx} {cy} {w} {h}\n' for c, cx, cy, w, h in rows))
    (root / 'data.yaml').write_text(
        f'path: {root}\ntest: images/test\nnc: {len(names)}\nnames: {json.dumps(names)}\n'
    )
    return root


# GT box of every _mc_dataset row below, in absolute pixels: (20, 20, 60, 60).
_BOX = (0.4, 0.4, 0.4, 0.4)


def test_dataset_loads_all_classes_and_present_set(tmp_path: Path) -> None:
    names = [f'c{i}' for i in range(84)]
    root = _mc_dataset(
        tmp_path / 'ds',
        {'a': [(37, *_BOX), (43, 0.8, 0.8, 0.1, 0.1)], 'b': [(37, *_BOX)], 'bg': []},
        names,
    )
    ds = YoloTestSet.from_dataset_root(root)
    assert ds.present_class_ids == {37, 43}
    assert ds.n_gt_by_class == {37: 2, 43: 1}
    assert ds.class_names[37] == 'c37'
    assert (ds.n_positive_frames, ds.n_background_frames) == (2, 1)
    gt = ds.coco_gt([37, 43])
    assert gt['categories'] == [{'id': 38, 'name': 'c37'}, {'id': 44, 'name': 'c43'}]
    assert sorted(a['category_id'] for a in gt['annotations']) == [38, 38, 44]
    only37 = ds.coco_gt([37])
    assert [a['category_id'] for a in only37['annotations']] == [38, 38]

    # data.yaml names: inline JSON list (multi-class export) and block dict
    # (single-class export) both parse.
    assert class_map.read_names(root / 'data.yaml')[43] == 'c43'
    block = tmp_path / 'block.yaml'
    block.write_text("path: /x\nnc: 2\nnames:\n  0: Mini Cooper\n  1: 'vw'\n")
    assert class_map.read_names(block) == {0: 'Mini Cooper', 1: 'vw'}
    (root / 'class_registry.json').write_text(json.dumps({'export_id_map': {'38': 37}}))
    assert class_map.read_export_id_map(root) == {38: 37}
    assert class_map.read_export_id_map(tmp_path) == {}


def _det(
    cls: int, score: float = 0.9, box: tuple[float, float, float, float] = (20, 20, 60, 60)
) -> Detection:
    x1, y1, x2, y2 = (float(v) for v in box)
    return Detection(x1, y1, x2, y2, score, class_id=cls)


def test_metrics_per_class_perfect_and_miss(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': [(1, *_BOX)]}, ['A', 'B'])
    )
    a, b = ds.images
    dets = {a.image_id: [_det(0)], b.image_id: []}
    res = metrics.coco_eval(ds.coco_gt([0, 1]), metrics.detections_to_coco(dets, None), [0, 1])
    assert res.per_class[0].ap50 == pytest.approx(1.0)
    assert res.per_class[1].ap50 == pytest.approx(0.0)
    assert res.overall.map_50 == pytest.approx(0.5)
    op = metrics.operating_point(ds.images, dets, conf=0.25, iou=0.45, class_ids=[0, 1])
    assert (op.per_class[0].tp, op.per_class[0].fp, op.per_class[0].fn) == (1, 0, 0)
    assert (op.per_class[1].tp, op.per_class[1].fp, op.per_class[1].fn) == (0, 0, 1)
    assert (op.micro.tp, op.micro.fp, op.micro.fn) == (1, 0, 1)
    assert op.micro.precision == pytest.approx(1.0)
    assert op.micro.recall == pytest.approx(0.5)
    # A class-0 box on the class-1 frame is an FP for class 0, not a TP for class 1.
    op = metrics.operating_point(
        ds.images, {b.image_id: [_det(0)]}, conf=0.25, iou=0.45, class_ids=[0, 1]
    )
    assert (op.per_class[0].fp, op.per_class[1].tp) == (1, 0)


def test_overall_map_equals_cocoeval_stats(tmp_path: Path) -> None:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    labels = {
        'a': [(0, *_BOX), (1, 0.8, 0.8, 0.2, 0.2)],
        'b': [(1, *_BOX), (2, 0.2, 0.8, 0.2, 0.3)],
        'c': [(2, *_BOX), (0, 0.75, 0.25, 0.3, 0.3)],
        'd': [],
    }
    ds = YoloTestSet.from_dataset_root(_mc_dataset(tmp_path / 'ds', labels, ['x', 'y', 'z']))
    ids = {img.path.stem: img.image_id for img in ds.images}
    dets = {
        ids['a']: [_det(0, 0.9), _det(1, 0.4, (68, 68, 90, 92)), _det(2, 0.3)],
        ids['b']: [_det(1, 0.8, (22, 18, 61, 63)), _det(0, 0.6)],
        ids['c']: [_det(2, 0.7), _det(0, 0.5, (60, 10, 88, 40)), _det(1, 0.2, (5, 5, 30, 30))],
        ids['d']: [_det(0, 0.35)],
    }
    gt = ds.coco_gt([0, 1, 2])
    results = metrics.detections_to_coco(dets, None)
    res = metrics.coco_eval(gt, results, [0, 1, 2])

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(results), iouType='bbox')
        ev.params.catIds = [1, 2, 3]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert res.overall.map_50_95 == pytest.approx(ev.stats[0], abs=1e-9)
    assert res.overall.map_50 == pytest.approx(ev.stats[1], abs=1e-9)
    assert res.overall.map_75 == pytest.approx(ev.stats[2], abs=1e-9)
    per_class_mean = sum(res.per_class[k].ap50_95 for k in (0, 1, 2)) / 3
    assert per_class_mean == pytest.approx(ev.stats[0], abs=1e-9)
    assert 0.0 < res.overall.map_50 < 1.0  # a non-trivial scenario


def _score(ds: YoloTestSet, dets, mapping: class_map.ClassMapping) -> dict:
    return scoring.score(ds, dets, mapping, scored_class_ids=sorted(ds.present_class_ids))


def test_uncovered_class_reported_not_dropped(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': [(1, *_BOX)]}, ['A', 'B'])
    )
    model_names = {0: 'a'}
    mapping = class_map.finalize(
        'names',
        class_map.resolve_by_names(model_names, ds.class_names),
        model_names=model_names,
        eval_names=ds.class_names,
        scored_class_ids=[0, 1],
    )
    blocks = _score(ds, {ds.images[0].image_id: [_det(0)]}, mapping)
    rows = {r['eval_class_id']: r for r in blocks['per_class']}
    assert rows[0]['covered'] is True
    assert rows[0]['model_class_ids'] == [0]
    assert rows[0]['ap50'] == pytest.approx(1.0)
    assert rows[1]['covered'] is False
    assert rows[1]['model_class_ids'] == []
    assert rows[1]['n_gt'] == 1
    for key in ('ap50', 'ap50_95', 'ap75', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'):
        assert rows[1][key] is None, key
    assert blocks['coverage']['not_covered'] == [{'eval_class_id': 1, 'name': 'B'}]
    assert blocks['coverage']['n_covered'] == 1
    assert blocks['coverage']['n_eval_classes'] == 2
    assert blocks['overall']['n_classes'] == 1
    assert blocks['overall']['map_50'] == pytest.approx(1.0)  # B's miss is not charged
    assert blocks['overall']['fn'] == 0


def test_unmapped_model_class_counted(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': []}, ['A', 'B'])
    )
    model_names = {0: 'A', 1: 'zebra', 2: 'b'}
    mapping = class_map.finalize(
        'names',
        class_map.resolve_by_names(model_names, ds.class_names),
        model_names=model_names,
        eval_names=ds.class_names,
        scored_class_ids=[0],
    )
    a, bg = ds.images
    dets = {
        a.image_id: [_det(0), _det(1, 0.8), _det(1, 0.7)],
        bg.image_id: [_det(1, 0.6), _det(2, 0.9)],  # class 2 -> 'B', not scored
    }
    blocks = scoring.score(ds, dets, mapping, scored_class_ids=[0])
    cov = blocks['coverage']
    assert cov['unmapped_model_classes'] == [
        {'model_class_id': 1, 'name': 'zebra', 'n_predictions': 3}
    ]
    assert cov['predictions_outside_scored_classes'] == 1
    assert blocks['overall']['fp'] == 0  # unmapped/outside predictions are not scored
    assert blocks['overall']['map_50'] == pytest.approx(1.0)


# The live 5-class subset run: remap new->registry, eval export_id_map registry->export.
_REMAP = {0: 38, 1: 39, 2: 44, 3: 52, 4: 79}
_EVAL_ID_MAP = {38: 37, 39: 38, 44: 43, 52: 51, 79: 78}
_EVAL_NAMES = {37: 'miata', 38: 'minicooper', 43: 'mustang', 51: 'porsche', 78: 'vw'}
_MODEL_NAMES = {0: 'miata', 1: 'minicooper', 2: 'mustang', 3: 'porsche', 4: 'vw'}


def test_class_map_registry_subset_remap() -> None:
    m2e, warnings = class_map.resolve_by_registry(
        _REMAP, _EVAL_ID_MAP, model_names=_MODEL_NAMES, eval_names=_EVAL_NAMES
    )
    assert m2e == {0: 37, 1: 38, 2: 43, 3: 51, 4: 78}
    assert warnings == []
    mapping = class_map.finalize(
        'run_class_remap',
        m2e,
        model_names=_MODEL_NAMES,
        eval_names=_EVAL_NAMES,
        scored_class_ids=[37, 38, 43, 51, 78],
        warnings=warnings,
    )
    assert mapping.to_dict() == {
        'method': 'run_class_remap',
        'model_to_eval': {'0': 37, '1': 38, '2': 43, '3': 51, '4': 78},
        'unmapped_model_classes': [],
        'not_covered_eval_classes': [],
        'warnings': [],
    }


def test_class_map_full_export_identity() -> None:
    export_id_map = {38: 0, 39: 1, 44: 2}
    model_to_registry = class_map.invert_export_id_map(export_id_map)
    assert model_to_registry == {0: 38, 1: 39, 2: 44}
    m2e, _ = class_map.resolve_by_registry(model_to_registry, export_id_map)
    assert m2e == {0: 0, 1: 1, 2: 2}


def test_class_map_full_export_identity_across_exports() -> None:
    train = {38: 0, 39: 1, 44: 2, 60: 3}
    evalm = {38: 5, 39: 3, 44: 9}  # dense positions moved; registry 60 gone
    m2e, _ = class_map.resolve_by_registry(class_map.invert_export_id_map(train), evalm)
    assert m2e == {0: 5, 1: 3, 2: 9}
    mapping = class_map.finalize(
        'registry_ids',
        m2e,
        model_names={0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        eval_names={3: 'b', 5: 'a', 9: 'c', 4: 'e'},
        scored_class_ids=[3, 4, 5, 9],
    )
    assert mapping.unmapped_model_classes == [{'model_class_id': 3, 'name': 'd'}]
    assert mapping.not_covered_eval_classes == [{'eval_class_id': 4, 'name': 'e'}]


def test_class_map_names_fallback_normalizes() -> None:
    assert class_map.normalize_name('  Mini Cooper ') == 'mini_cooper'
    assert class_map.normalize_name('VW--Beetle') == 'vw_beetle'
    m2e = class_map.resolve_by_names(
        {0: 'Mini Cooper', 1: 'VW-Beetle', 2: 'truck'}, {0: 'mini_cooper', 1: 'vw_beetle', 2: 'bus'}
    )
    assert m2e == {0: 0, 1: 1}


def test_class_map_names_fallback_registry_name_mismatch_warns() -> None:
    m2e, warnings = class_map.resolve_by_registry(
        {3: 52}, {52: 51}, model_names={3: 'porsche'}, eval_names={51: 'porsche_911'}
    )
    assert m2e == {3: 51}
    assert warnings == [
        "model class 3 'porsche' maps to eval class 51 'porsche_911' by registry id"
    ]
    # Names that only differ by normalization are not a mismatch.
    _, warnings = class_map.resolve_by_registry(
        {3: 52}, {52: 51}, model_names={3: 'Porsche 911'}, eval_names={51: 'porsche_911'}
    )
    assert warnings == []


def test_class_map_explicit_names_and_merge_warning() -> None:
    m2e, warnings = class_map.resolve_explicit(
        {'0': 'Miata', '1': 'nope', '2': 'miata'}, _EVAL_NAMES
    )
    assert m2e == {0: 37, 2: 37}
    assert warnings == ["class_map name 'nope' (model class 1) is not an eval class"]
    mapping = class_map.finalize(
        'explicit', m2e, model_names=None, eval_names=_EVAL_NAMES, scored_class_ids=[37, 38]
    )
    assert 'model classes 0, 2 all map to eval class 37' in ' '.join(mapping.warnings)
    assert mapping.not_covered_eval_classes == [{'eval_class_id': 38, 'name': 'minicooper'}]


# --- compare.py -----------------------------------------------------------------------


def _v2_report(key: str, display: str, aps: dict[int, float | None], **extra) -> dict:
    """A per-model v2 report: ``aps[eval_id]`` = AP (None = class not covered)."""
    per_class: list[dict[str, Any]] = []
    for cid, ap in sorted(aps.items()):
        if ap is None:
            per_class.append(
                {'eval_class_id': cid, 'name': f'c{cid}', 'n_gt': 2, 'covered': False,
                 'model_class_ids': [], **dict.fromkeys(
                     ('ap50', 'ap50_95', 'ap75', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'))}
            )  # fmt: skip
        else:
            per_class.append(
                {'eval_class_id': cid, 'name': f'c{cid}', 'n_gt': 2, 'covered': True,
                 'model_class_ids': [cid], 'ap50': ap, 'ap50_95': ap, 'ap75': ap,
                 'precision': 1.0, 'recall': 0.5, 'f1': 2 / 3, 'tp': 1, 'fp': 0, 'fn': 1}
            )  # fmt: skip
    covered = [r for r in per_class if r['covered']]
    mean = (sum(float(r['ap50_95']) for r in covered) / len(covered)) if covered else None
    return {
        'schema_version': 2,
        'model': key,
        'display_name': display,
        'source': 'run',
        'run_id': key.split(':', 1)[-1],
        'runtime': 'ultralytics',
        'imgsz': 640,
        'training_data': None,
        'overall': {
            'n_classes': len(covered),
            'map_50': mean,
            'map_50_95': mean,
            'map_75': mean,
            'precision': 1.0 if covered else None,
            'recall': 0.5 if covered else None,
            'f1': 2 / 3 if covered else None,
            'tp': len(covered),
            'fp': 0,
            'fn': len(covered),
        },
        'common': None,
        'per_class': per_class,
        'coverage': {
            'n_eval_classes': len(per_class),
            'n_covered': len(covered),
            'not_covered': [],
            'unmapped_model_classes': [],
            'predictions_outside_scored_classes': 0,
        },
        'class_mapping': {'method': 'names', 'warnings': []},
        'train_test_overlap': None,
        'latency_ms': {'mean': extra.get('latency', 5.0), 'p50': 0.0, 'p90': 0.0, 'p99': 0.0},
        'fps': 200.0,
        'size_mb': extra.get('size_mb', 5.0),
        'per_stratum': {},
        'test_frames': 4,
        'positive_frames': 3,
        'background_frames': 1,
    }


def _write_reports(d: Path, *reports: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for r in reports:
        (d / f'{r["model"].replace(":", "_")}.json').write_text(json.dumps(r))
    return d


def _comparison(d: Path, **kw) -> dict:
    return compare.build_comparison(
        d,
        rank_by=kw.pop('rank_by', 'map_50_95'),
        dataset_meta={'id': 'export:x', 'frozen_test_sha': 'f' * 16, 'test_label_sha': 'a' * 16},
        job_id='j',
        profile='generic',
        thresholds={'conf_floor': 0.001, 'nms_iou': 0.7, 'op_conf': 0.25, 'op_iou': 0.45},
        failed=kw.pop('failed', []),
    )


def test_comparison_common_classes_and_ties(tmp_path: Path) -> None:
    d = _write_reports(
        tmp_path / 'ds',
        _v2_report('run:a', 'bravo', {0: 0.8, 1: 0.8, 2: 0.2}, latency=3.0),
        _v2_report('run:b', 'alpha', {0: 0.80001, 1: 0.79999, 2: None}, latency=3.0),
        _v2_report('run:c', 'charlie', {0: 0.5, 1: 0.5, 2: 0.9}),
        _v2_report('run:z', 'aaa', {0: None, 1: None, 2: None}),  # covers nothing
    )
    comp = _comparison(d, failed=[{'model': 'run:dead', 'error': 'boom'}])
    assert comp['schema_version'] == 2
    assert comp['common_classes'] == [0, 1]
    assert comp['rank_scope'] == 'common'
    assert [c['eval_class_id'] for c in comp['eval_classes']] == [0, 1, 2]
    assert comp['eval_classes'][0]['n_gt'] == 2
    order = [(m['display_name'], m['rank']) for m in comp['models']]
    assert order == [('alpha', 1), ('bravo', 1), ('charlie', 3), ('aaa', None)]
    bravo = comp['models'][1]
    assert bravo['common']['n_classes'] == 2
    assert bravo['common']['map_50_95'] == pytest.approx(0.8)
    assert bravo['overall']['map_50_95'] == pytest.approx(0.6)  # own 3 classes
    assert bravo['common']['tp'] == 2
    assert comp['failed'] == [{'model': 'run:dead', 'error': 'boom'}]
    assert comp['n_models'] == 4
    assert comp['dataset']['n_images'] == 4
    assert comp['warnings'] == []

    matrix = compare.build_matrix(
        [{'id': 'export:x', 'frozen_test_sha': 'f' * 16, 'test_label_sha': 'a' * 16}],
        {'export:x': comp},
    )
    best = matrix['best']['export:x']
    assert sorted(best['map_50_95']) == ['run:a', 'run:b']
    assert sorted(best['latency_ms']) == ['run:a', 'run:b']  # lower is better, tied
    assert matrix['cells']['run:a']['export:x']['rank'] == 1
    assert matrix['cells']['run:a']['export:x']['coverage'] == pytest.approx(1.0)
    assert matrix['cells']['run:b']['export:x']['coverage'] == pytest.approx(2 / 3)
    assert matrix['datasets'][0]['rank_scope'] == 'common'
    assert matrix['datasets'][0]['n_common_classes'] == 2
    assert 'rank:1' not in compare.to_markdown(comp)  # renders without crashing
    assert compare.to_latex_rows(comp).count('\\\\') == 4


def test_comparison_disjoint_models_fall_back_to_overall_with_warning(tmp_path: Path) -> None:
    d = _write_reports(
        tmp_path / 'ds',
        _v2_report('run:a', 'a', {0: 0.4, 1: None}),
        _v2_report('run:b', 'b', {0: None, 1: 0.9}),
    )
    comp = _comparison(d)
    assert comp['common_classes'] == []
    assert comp['rank_scope'] == 'overall'
    assert comp['warnings'] == [
        "no class is covered by every model; ranked on each model's own classes, not comparable"
    ]
    assert [(m['model'], m['rank']) for m in comp['models']] == [('run:b', 1), ('run:a', 2)]
    assert comp['models'][0]['common']['n_classes'] == 0
    assert comp['models'][0]['common']['map_50_95'] is None


def test_comparison_rejects_unrankable_metric(tmp_path: Path) -> None:
    d = _write_reports(tmp_path / 'ds', _v2_report('run:a', 'a', {0: 0.4}))
    with pytest.raises(ValueError, match='cannot rank by'):
        _comparison(d, rank_by='ap_small')
    by_recall = _comparison(d, rank_by='recall')
    assert by_recall['rank_by'] == 'recall'


# --- bakeoff_runner.py (job spec v2) ------------------------------------------------------


def _v2_dataset(tmp_path: Path) -> tuple[Path, dict]:
    root = _mc_dataset(
        tmp_path / 'ds',
        {'a': [(0, *_BOX)], 'b': [(1, *_BOX)], 'bg': []},
        ['dog', 'cat'],
        fill={'a': 10, 'b': 200, 'bg': 100},
    )
    entry = {
        'id': 'export:ds',
        'dir_name': 'export__ds',
        'path': str(root),
        'test_label_sha': label_sha(root)[0],
        'frozen_test_sha': '0' * 16,
        'eval_class_ids': [0, 1],
    }
    return root, entry


def _v2_model(key: str, class_map_by_dataset: dict | None = None, **kw) -> dict:
    return {
        'model': key,
        'display_name': kw.pop('display_name', key),
        'source': 'custom',
        'run_id': None,
        'backend': kw.pop('backend', 'ultralytics'),
        'weights': None,
        'imgsz': 64,
        'mode': 'full',
        'backend_options': {},
        'triton_model': None,
        'training_data': None,
        'class_map_by_dataset': class_map_by_dataset or {'export:ds': None},
        'train_test_overlap_by_dataset': {'export:ds': None},
        **kw,
    }


def _v2_spec(tmp_path: Path, datasets_: list[dict], models: list[dict], **kw) -> dict:
    return {
        'schema_version': 2,
        'job_id': 'j2',
        'profile': 'generic',
        'out_dir': str(tmp_path / 'out'),
        'datasets': datasets_,
        'models': models,
        'quantize': None,
        **kw,
    }


def test_runner_rejects_changed_test_split(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(bakeoff_runner, '_run_task', lambda *a: calls.append(a))
    _, entry = _v2_dataset(tmp_path)
    actual = entry['test_label_sha']
    entry['test_label_sha'] = 'deadbeefdeadbeef'
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [_v2_model('custom:m')]))
    assert calls == []
    assert {
        'stage': None,
        'dataset': 'export:ds',
        'model': None,
        'error': f'test split changed since enqueue: expected deadbeefdeadbeef, now {actual}',
    } in status['failed']
    assert status['state'] == 'error'
    on_disk = json.loads((tmp_path / 'out/status.json').read_text())
    assert on_disk['schema_version'] == 2


def test_runner_rejects_v1_job_spec(tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(
        {'job_id': 'old', 'out_dir': str(tmp_path / 'out'), 'dataset': '/d', 'models': []}
    )
    assert status['state'] == 'error'
    assert 'schema_version' in status['error']


def test_runner_rejects_unknown_profile(tmp_path: Path) -> None:
    _, entry = _v2_dataset(tmp_path)
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [], profile='nope'))
    assert status['state'] == 'error'
    assert 'unknown bake-off profile' in status['error']
    assert json.loads((tmp_path / 'out/status.json').read_text())['state'] == 'error'


def test_runner_passes_v2_model_fields_to_run(tmp_path: Path, monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_task(ds_path, ds_out, ds_id, model, gpu):
        ds_out.mkdir(parents=True, exist_ok=True)
        seen.append(bakeoff_runner._model_argv(ds_path, ds_out, ds_id, model))
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)
    _, entry = _v2_dataset(tmp_path)
    model = _v2_model(
        'run:r1',
        {'export:ds': {'0': 1}},
        backend_options={'providers': 'CPUExecutionProvider'},
        train_test_overlap_by_dataset={'export:ds': {'n_images': 1, 'fraction': 0.5}},
        mode='both',
    )
    prof = tmp_path / 'prof.json'
    prof.write_text(json.dumps({'name': 'mine', 'context_class_ids': [4]}))
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [model], profile=str(prof)))
    assert status['state'] == 'done'
    assert status['models'] == ['run:r1:full', 'run:r1:crop']
    argv = seen[0]

    def flag(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert flag('--profile') == str(prof)
    assert json.loads(flag('--class-map-json')) == {'0': 1}
    assert json.loads(flag('--backend-options-json')) == {'providers': 'CPUExecutionProvider'}
    assert json.loads(flag('--train-test-overlap-json')) == {'n_images': 1, 'fraction': 0.5}
    assert {a[a.index('--model-key') + 1] for a in seen} == {'run:r1:full', 'run:r1:crop'}
    assert flag('--source') == 'custom'
    assert '--gt-class-id' not in argv


class _FakeDetector:
    """Scripted detector: 'cat'/'dog' named classes; answers by image brightness."""

    runtime = 'fake'
    class_names: dict[int, str] | None = {0: 'cat', 1: 'dog'}

    def __init__(self, name: str) -> None:
        self.name = name

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        mean = float(image_rgb.mean())
        if mean < 50:  # image 'a': a dog -> model class 1
            return [_det(1, 0.9)]
        if mean > 150:  # image 'b': a cat -> model class 0
            return [_det(0, 0.8)]
        return []


def test_runner_v2_end_to_end_with_fake_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))  # undo after the test
    registry.register_backend('fake-scripted', lambda args: _FakeDetector(args.name))

    def in_process_task(ds_path, ds_out, ds_id, model, gpu):
        argv = bakeoff_runner._model_argv(ds_path, ds_out, ds_id, model)
        assert argv[1:3] == ['-m', 'scripts.curation.bakeoff.run']
        assert run.main([*argv[3:], '--no-mlflow']) == 0
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', in_process_task)
    _, entry = _v2_dataset(tmp_path)
    models = [
        _v2_model('custom:by-name', display_name='by name', backend='fake-scripted'),
        _v2_model(
            'custom:dog-only',
            {'export:ds': {'1': 0}},
            display_name='dog only',
            backend='fake-scripted',
        ),
    ]
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], models))
    out = tmp_path / 'out'
    assert status['state'] == 'done', status
    assert json.loads((out / 'status.json').read_text())['schema_version'] == 2
    assert not (out / 'comparison.json').exists()  # per-dataset files only
    ds_out = out / 'export__ds'
    report = json.loads((ds_out / 'custom_by-name.json').read_text())
    assert report['schema_version'] == 2
    assert report['class_mapping']['method'] == 'names'
    assert report['overall']['map_50'] == pytest.approx(1.0)
    assert (report['test_frames'], report['background_frames']) == (3, 1)
    comp = json.loads((ds_out / 'comparison.json').read_text())
    assert comp['schema_version'] == 2
    assert comp['dataset']['id'] == 'export:ds'
    assert comp['common_classes'] == [0]
    assert [m['model'] for m in comp['models']] == ['custom:by-name', 'custom:dog-only']
    dog_only = comp['models'][1]
    assert dog_only['coverage']['not_covered'] == [{'eval_class_id': 1, 'name': 'cat'}]
    assert dog_only['class_mapping']['method'] == 'explicit'
    matrix = json.loads((out / 'matrix.json').read_text())
    assert matrix['schema_version'] == 2
    assert matrix['datasets'][0]['id'] == 'export:ds'
    assert sorted(matrix['best']['export:ds']['map_50_95']) == ['custom:by-name', 'custom:dog-only']


# --- domain-neutral core guard ---------------------------------------------------

# Domain content (profiles, baselines, converters, backends for one domain's
# public models) lives only under the opt-in examples/ tree. The router is
# rewritten in the W4 API wave (its v1 request model still names legacy
# backends) and joins this scan there; the two W4 service modules are scanned
# as soon as they exist.
_DOMAIN_WORDS = re.compile(r'(?i)plate|lpr|lpdnet|vehicle|open[_-]?image')
_BAKEOFF_API_FILES = (
    'src/routers/curation/bakeoff.py',
    'src/routers/curation/_bakeoff_models.py',
    'src/services/curation/eval_datasets.py',
    'src/services/curation/bakeoff_jobs.py',
)


def test_harness_core_has_no_domain_vocabulary() -> None:
    files = [p for p in sorted(HARNESS.rglob('*')) if p.suffix in {'.py', '.json', '.txt', '.md'}]
    files += [REPO_ROOT / f for f in _BAKEOFF_API_FILES]
    offenders: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if _DOMAIN_WORDS.search(line):
                offenders.append(f'{rel}:{n}: {line.strip()}')
        if _DOMAIN_WORDS.search(rel):
            offenders.append(rel)
    assert not offenders, 'domain vocabulary in the generic harness core:\n' + '\n'.join(offenders)


def test_builtin_backends_are_generic() -> None:
    """Before any plugin module is imported, only the five generic backends exist."""
    import subprocess
    import sys

    code = (
        'from scripts.curation.bakeoff.backends import registry; '
        'print(",".join(sorted(registry.registered_backends())))'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    builtins = {'ultralytics', 'triton', 'two-stage', 'onnxruntime', 'coreml'}
    assert set(proc.stdout.strip().split(',')) == builtins
    from scripts.curation.bakeoff.profile import BACKENDS

    assert set(BACKENDS) == builtins
    with pytest.raises(SystemExit, match='unknown bake-off backend'):
        registry.get_backend('no-such-backend')
    with pytest.raises(ValueError, match='already registered'):
        registry.register_backend('ultralytics', lambda _args: _FakeDetector('x'))


def test_paper_modules_are_not_in_the_harness() -> None:
    for name in ('lean_candidates', 'deskew_prototype', 'dedup_sweep', 'paper_numbers'):
        assert not (HARNESS / f'{name}.py').exists(), name
