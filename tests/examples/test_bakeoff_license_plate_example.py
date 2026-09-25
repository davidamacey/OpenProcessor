"""The opt-in license_plate bake-off example (``examples/bakeoff/license_plate/``).

Nothing in ``src/`` or ``scripts/`` imports this example: it is loaded only by
an explicit profile path, which pulls in its converter and backend plugins.
These tests load it exactly that way.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.curation.bakeoff import datasets, run
from scripts.curation.bakeoff.backends import registry
from scripts.curation.bakeoff.profile import (
    BACKENDS,
    load_converter_plugins,
    resolve_baselines_path,
    resolve_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / 'examples/bakeoff/license_plate'
PROFILE = EXAMPLE / 'profile.json'


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


def test_profile_loads_by_path_only() -> None:
    with pytest.raises(ValueError, match='unknown bake-off profile'):
        resolve_profile('license_plate')
    lp = resolve_profile(str(PROFILE))
    assert lp.name == 'license_plate'
    assert lp.class_names == ('license_plate',)
    assert lp.context_class_ids == (2, 3, 5, 7)
    assert lp.imgsz == 1280
    assert lp.triton_model == ''
    assert lp.converter_modules == ('examples.bakeoff.license_plate.converters',)
    assert lp.backend_modules == ('examples.bakeoff.license_plate.backends',)
    assert resolve_profile(None).name == 'generic'


def test_env_selects_example_profile_by_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', str(PROFILE))
    assert resolve_profile(None).name == 'license_plate'


def test_backend_plugins_register_on_load() -> None:
    assert not {'lpdnet', 'open-image-models'} & set(BACKENDS)
    registry.load_backend_plugins(resolve_profile(str(PROFILE)))
    names = set(registry.registered_backends())
    assert {'lpdnet', 'open-image-models'} <= names
    # Import only: constructing them needs weights / the open-image-models package.
    assert callable(registry.get_backend('lpdnet'))


def test_run_resolves_profile_and_loads_its_plugins() -> None:
    args, _ = run.resolve_args(
        run.build_parser().parse_args(
            [
                '--profile',
                str(PROFILE),
                '--dataset',
                '/x',
                '--backend',
                'lpdnet',
                '--backend-options-json',
                '{"variant": "usa"}',
            ]
        )
    )
    assert args.backend == 'lpdnet'
    assert args.backend_options == {'variant': 'usa'}
    assert run.parse_class_ids(args.primary_classes) == (2, 3, 5, 7)
    assert args.imgsz == 1280
    args, _ = run.resolve_args(
        run.build_parser().parse_args(
            ['--profile', str(PROFILE), '--primary-classes', '0', '--imgsz', '320']
        )
    )
    assert (args.primary_classes, args.imgsz) == ('0', 320)


def test_converters_register_from_the_example(tmp_path: Path) -> None:
    lp = resolve_profile(str(PROFILE))
    load_converter_plugins(lp)
    conv = datasets.available_converters()
    for name in ('ccpd', 'ufpr', 'openalpr'):
        assert conv[name].example_for == 'license_plate'
    src = tmp_path / 'ccpd'
    _img(src / '01-90_85-10&5_60&45-x.png')
    out = tmp_path / 'out'
    assert datasets.convert('ccpd', src, out, lp) == 1
    label = (out / 'labels/test/01-90_85-10&5_60&45-x.txt').read_text().split()
    assert label[0] == '0'
    assert [float(v) for v in label[1:]] == pytest.approx([0.35, 0.5, 0.5, 0.8])
    assert '  0: license_plate\n' in (out / 'data.yaml').read_text()


def test_datasets_cli_loads_profile_converters(tmp_path: Path) -> None:
    """``python -m ...datasets`` must see plugin converters (``__main__`` vs package)."""
    src = tmp_path / 'ccpd'
    _img(src / '01-90_85-10&5_60&45-x.png')
    out = tmp_path / 'out'
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'scripts.curation.bakeoff.datasets',
            '--profile',
            str(PROFILE),
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


_BASELINE_NAME_RE = re.compile(r'[A-Za-z0-9_.\-]{1,64}')


def test_baselines_file_validates() -> None:
    """Every entry matches the baseline-registry schema (plan 7.6) and has a class_map."""
    lp = resolve_profile(str(PROFILE))
    path = resolve_baselines_path(lp, Path('/nonexistent.json'))
    assert path == EXAMPLE / 'baselines.json'
    registry.load_backend_plugins(lp)
    known_backends = set(registry.registered_backends())
    baselines = json.loads(path.read_text())['baselines']
    assert baselines
    names = [b['name'] for b in baselines]
    assert len(names) == len(set(names))
    allowed = {
        'name', 'backend', 'weights', 'imgsz', 'mode', 'class_map',
        'backend_options', 'training_data', 'triton_model',
    }  # fmt: skip
    for b in baselines:
        assert set(b) <= allowed, (b['name'], set(b) - allowed)
        assert _BASELINE_NAME_RE.fullmatch(b['name'])
        assert b['backend'] in known_backends
        assert b['class_map'] == {'0': 'license_plate'}
        assert isinstance(b.get('backend_options', {}), dict)
        assert b.get('mode', 'full') in {'full', 'crop', 'both'}
    lpdnet = next(b for b in baselines if b['backend'] == 'lpdnet')
    assert lpdnet['backend_options'] == {'variant': 'usa'}
    assert 'lpr_nanov11_640' not in names


def test_api_default_profile_from_example_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OP_BAKEOFF_PROFILE=<example path>`` makes it the (configured) API default."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', str(PROFILE))
    body = TestClient(app).get('/curation/bakeoff/profiles').json()
    by_name = {p['name']: p for p in body['profiles']}
    assert body['default_profile'] == 'license_plate'
    assert by_name['license_plate']['default'] is True
    assert by_name['license_plate']['kind'] == 'configured'
    assert by_name['license_plate']['context_class_ids'] == [2, 3, 5, 7]
    assert by_name['generic']['default'] is False
