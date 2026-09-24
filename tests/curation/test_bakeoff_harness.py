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

import cv2
import numpy as np
import pytest

from scripts.curation.bakeoff import bakeoff_runner, compare, datasets, run
from scripts.curation.bakeoff.profile import (
    GENERIC_PROFILE,
    BakeoffProfile,
    example_profile_names,
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
    assert p.target_class_name == 'object'
    assert p.context_class_ids == ()
    assert p.triton_model == ''
    assert p.label_names() == ('object',)


def test_license_plate_is_an_example_not_the_default() -> None:
    assert 'license_plate' in example_profile_names()
    lp = resolve_profile('license_plate')
    assert lp.target_class_name == 'license_plate'
    assert lp.context_class_ids == (2, 3, 5, 7)
    assert lp.triton_model == ''
    assert resolve_profile(None).name == 'generic'


def test_profile_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_TARGET_CLASS_ID', '2')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CLASS_NAMES', 'pallet,box,label')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME', 'label')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '0, 1')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_OP_CONF', '0.4')
    p = resolve_profile(None)
    assert p.target_class_id == 2
    assert p.class_names == ('pallet', 'box', 'label')
    assert p.context_class_ids == (0, 1)
    assert p.op_conf == pytest.approx(0.4)


def test_env_selects_named_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', 'license_plate')
    assert resolve_profile(None).name == 'license_plate'


def test_profile_json_path_and_unknown_name(tmp_path: Path) -> None:
    f = tmp_path / 'mine.json'
    f.write_text(json.dumps({'name': 'widgets', 'target_class_name': 'widget'}))
    assert resolve_profile(str(f)).target_class_name == 'widget'
    with pytest.raises(ValueError, match='unknown bake-off profile'):
        resolve_profile('no_such_profile')


def test_profile_rejects_target_outside_label_space() -> None:
    with pytest.raises(ValueError, match='outside class_names'):
        BakeoffProfile(name='bad', target_class_id=3)
    with pytest.raises(ValueError, match='unknown BakeoffProfile field'):
        BakeoffProfile.from_dict({'name': 'x', 'plate_class_id': 0})


def test_profile_baselines_path_resolves_against_repo() -> None:
    default = Path('/nonexistent/default.json')
    assert resolve_baselines_path(GENERIC_PROFILE, default) == default
    lp = resolve_baselines_path(resolve_profile('license_plate'), default)
    assert lp.is_file()
    assert lp.is_relative_to(HARNESS / 'examples')


# --- run.py: profile vs CLI precedence ------------------------------------------


def _resolve(*argv: str):
    return run.resolve_args(run.build_parser().parse_args(list(argv)))


def test_run_generic_profile_has_no_hardcoded_context_classes() -> None:
    args, profile = _resolve('--dataset', '/x')
    assert profile.name == 'generic'
    assert args.primary_classes == ''
    assert run.parse_class_ids(args.primary_classes) == ()
    assert args.gt_class_id == 0
    assert args.gt_class_name == 'object'
    assert args.backend == 'ultralytics'


def test_run_profile_fills_unset_flags_and_cli_wins() -> None:
    args, _ = _resolve('--profile', 'license_plate', '--dataset', '/x')
    assert run.parse_class_ids(args.primary_classes) == (2, 3, 5, 7)
    assert args.gt_class_name == 'license_plate'
    args, _ = _resolve(
        '--profile', 'license_plate', '--primary-classes', '0', '--gt-class-name', 'x'
    )
    assert args.primary_classes == '0'
    assert args.gt_class_name == 'x'


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
    prof = BakeoffProfile(
        name='shop', target_class_id=1, target_class_name='box', class_names=('pallet', 'box')
    )
    w = datasets.YoloWriter.for_profile(tmp_path, prof, split='test')
    w.write_data_yaml()
    yaml = (tmp_path / 'data.yaml').read_text()
    assert 'nc: 2\n' in yaml
    assert '  0: pallet\n  1: box\n' in yaml
    with pytest.raises(ValueError, match='outside nc'):
        w.write('a', _img(tmp_path / 'src/a.png'), [(5, 0.5, 0.5, 0.1, 0.1)])


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
    prof = BakeoffProfile(
        name='shop', target_class_id=1, target_class_name='box', class_names=('pallet', 'box')
    )
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


def test_plate_converters_come_from_the_example_profile(tmp_path: Path) -> None:
    lp = resolve_profile('license_plate')
    src = tmp_path / 'ccpd'
    _img(src / '01-90_85-10&5_60&45-x.png')
    out = tmp_path / 'out'
    assert datasets.convert('ccpd', src, out, lp) == 1
    assert datasets.available_converters()['ccpd'].example_for == 'license_plate'
    label = (out / 'labels/test/01-90_85-10&5_60&45-x.txt').read_text().split()
    assert label[0] == '0'
    assert [float(v) for v in label[1:]] == pytest.approx([0.35, 0.5, 0.5, 0.8])
    assert '  0: license_plate\n' in (out / 'data.yaml').read_text()


def test_unknown_format_error_names_registry() -> None:
    with pytest.raises(ValueError, match='unknown dataset format'):
        datasets.get_converter('no-such-format')


# --- compare.py + bakeoff_runner.py ---------------------------------------------


def _report(model: str, map5095: float, recall: float) -> dict:
    return {
        'model': model,
        'coco': {'map_50': 0.5, 'map_50_95': map5095, 'ap_small': 0.1},
        'operating_point': {'precision': 0.5, 'recall': recall, 'f1': 0.5, 'mean_iou': 0.5},
    }


def test_comparison_ranks_by_profile_metric(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text(json.dumps(_report('a', 0.9, 0.1)))
    (tmp_path / 'b.json').write_text(json.dumps(_report('b', 0.1, 0.9)))
    assert [m['model'] for m in compare.build_comparison(tmp_path)['models']] == ['a', 'b']
    by_recall = compare.build_comparison(tmp_path, rank_by='recall')
    assert [m['model'] for m in by_recall['models']] == ['b', 'a']
    assert by_recall['rank_by'] == 'recall'


def test_runner_passes_job_profile_to_every_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_task(ds_path, ds_out, model, gpu):
        ds_out.mkdir(parents=True, exist_ok=True)  # run.py creates its --out-dir
        seen.append(bakeoff_runner._model_argv(ds_path, ds_out, model))
        return model['name'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)
    ds = tmp_path / 'ds'
    ds.mkdir()
    status = bakeoff_runner.run_job(
        {
            'job_id': 'j1',
            'profile': 'license_plate',
            'verify_frozen': False,
            'out_dir': str(tmp_path / 'out'),
            'datasets': [str(ds)],
            'models': [
                {'backend': 'ultralytics', 'name': 'm1'},
                {'backend': 'ultralytics', 'name': 'm2', 'profile': 'generic'},
            ],
        }
    )
    assert status['state'] == 'done'
    assert status['profile'] == 'license_plate'
    profiles = {argv[argv.index('--name') + 1]: argv[argv.index('--profile') + 1] for argv in seen}
    assert profiles == {'m1': 'license_plate', 'm2': 'generic'}


def test_runner_rejects_unknown_profile(tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(
        {'job_id': 'j2', 'profile': 'nope', 'out_dir': str(tmp_path), 'models': []}
    )
    assert status['state'] == 'error'
    assert 'unknown bake-off profile' in status['error']
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'error'


# --- domain-neutral core guard ---------------------------------------------------

# Files that are domain-specific by nature: the example tree and two
# backends wrapping public license-plate-only models.
_DOMAIN_FILES = {'backends/lpdnet.py', 'backends/open_image_models.py'}
_DOMAIN_WORDS = re.compile(r'\b(plates?|lpr|licen[cs]e(?:[_ -]plates?)?|vehicles?)\b', re.I)


def test_harness_core_has_no_domain_vocabulary() -> None:
    offenders: list[str] = []
    for path in sorted(HARNESS.rglob('*')):
        rel = path.relative_to(HARNESS).as_posix()
        if path.suffix not in {'.py', '.json'} or rel.startswith('examples/'):
            continue
        if rel in _DOMAIN_FILES:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if _DOMAIN_WORDS.search(line):
                offenders.append(f'{rel}:{n}: {line.strip()}')
    assert not offenders, 'domain vocabulary in the generic harness core:\n' + '\n'.join(offenders)


def test_paper_modules_are_not_in_the_harness() -> None:
    for name in ('lean_candidates', 'deskew_prototype', 'dedup_sweep', 'paper_numbers'):
        assert not (HARNESS / f'{name}.py').exists(), name


def test_datasets_cli_loads_profile_converters(tmp_path: Path) -> None:
    """``python -m ...datasets`` must see plugin converters (``__main__`` vs package)."""
    import subprocess
    import sys

    src = tmp_path / 'ccpd'
    _img(src / '01-90_85-10&5_60&45-x.png')
    out = tmp_path / 'out'
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'scripts.curation.bakeoff.datasets',
            '--profile',
            'license_plate',
            '--format',
            'ccpd',
            '--src',
            str(src),
            '--out',
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'converted 1 images (ccpd, profile=license_plate)' in proc.stdout
    assert (out / 'labels/test/01-90_85-10&5_60&45-x.txt').read_text().startswith('0 ')
