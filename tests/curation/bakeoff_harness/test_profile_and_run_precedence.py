"""BakeoffProfile resolution and CLI-versus-profile precedence in ``run``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curation.bakeoff_harness._helpers import HARNESS
from scripts.curation.bakeoff import run
from scripts.curation.bakeoff.profile import (
    GENERIC_PROFILE,
    BakeoffProfile,
    resolve_baselines_path,
    resolve_profile,
)


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
