"""Class mapping between registry and export ids, and ranking in ``compare``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation.bakeoff import class_map, compare


if TYPE_CHECKING:
    from pathlib import Path


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
        'model_to_eval_names': {
            '0': 'miata',
            '1': 'minicooper',
            '2': 'mustang',
            '3': 'porsche',
            '4': 'vw',
        },
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
