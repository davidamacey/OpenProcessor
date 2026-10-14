"""Router registration and typing, harness hygiene, and the API job spec run in the evaluator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _route_helpers import api_routes

from curation.bakeoff_router._helpers import _names_yaml, _post_run


if TYPE_CHECKING:
    import pytest
    from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[3]

# This harness used to default to owner-private absolute paths --
# one of which named the location of a licensed proprietary image corpus
# (a "/mnt/<host-specific-mount>/..." style path) and must never appear
# in this repo as a literal string again. A generic "any absolute path"
# scan is too broad (it also matches route strings like '/bakeoff/run',
# shebangs, etc.) -- narrow this to host-mount-shaped absolute paths
# (/mnt/..., /home/..., /Users/...), which is exactly the shape the
# original regression had and nothing legitimate in this harness needs.
# Every text file in the harness tree is scanned (not a hand-picked list: a
# hand-picked list once missed a backend docstring carrying such a path).
_HARNESS_SUFFIXES = {'.py', '.json', '.txt', '.md'}


def _bakeoff_harness_files() -> list[str]:
    files = [
        'src/routers/curation/bakeoff.py',
        *(
            f'src/schemas/bakeoff/{m}.py'
            for m in ['datasets', 'job_spec', 'models', 'results', 'run']
        ),
        'src/services/curation/eval_datasets.py',
        'src/services/curation/bakeoff_jobs.py',
    ]
    for root in ('scripts/curation/bakeoff', 'examples/bakeoff', 'src/schemas/bakeoff'):
        files += [
            p.relative_to(REPO_ROOT).as_posix()
            for p in sorted((REPO_ROOT / root).rglob('*'))
            if p.is_file() and p.suffix in _HARNESS_SUFFIXES
        ]
    return files


_HOST_MOUNT_PATH_RE = re.compile(r'/(?:mnt|home|Users)/[A-Za-z0-9_./\-]+')


def test_bakeoff_harness_has_no_owner_private_absolute_path_defaults() -> None:
    offenders: list[str] = []
    files = _bakeoff_harness_files()
    assert 'examples/bakeoff/license_plate/backends/lpdnet.py' in files
    for rel in files:
        text = (REPO_ROOT / rel).read_text()
        offenders.extend(f'{rel}: {match.group(0)}' for match in _HOST_MOUNT_PATH_RE.finditer(text))
    assert not offenders, (
        'bake-off harness file(s) contain a hardcoded host-mount-shaped '
        f'absolute path default: {offenders}'
    )


# =============================================================================
# Route registration + typing
# =============================================================================


def test_bakeoff_router_is_registered() -> None:
    from src.main import app

    paths = {r.path for r in api_routes(app)}
    assert '/curation/projects/{project}/bakeoff/runs' in paths
    assert '/curation/projects/{project}/bakeoff/eval_datasets' in paths


def test_bakeoff_routes_have_typed_response_models() -> None:
    from src.main import app

    routes = [
        r for r in api_routes(app) if r.path.startswith('/curation/projects/{project}/bakeoff/')
    ]
    assert len(routes) == 9
    untyped = [r.path for r in routes if getattr(r, 'response_model', None) is None]
    assert not untyped, f'bake-off routes without response_model: {untyped}'


def test_bakeoff_router_docstring_section_exists() -> None:
    from src.routers.curation import bakeoff

    doc = bakeoff.__doc__ or ''
    m = re.search(r'curation_design_rationale\.md`` §(\d+)', doc)
    assert m, 'router docstring should cite a specific rationale section'
    rationale = (REPO_ROOT / 'docs/design/curation_design_rationale.md').read_text()
    heading = re.search(rf'^## {m.group(1)}\. (.+)$', rationale, re.M)
    assert heading, f'rationale doc has no section {m.group(1)}'
    assert 'bake-off' in heading.group(1).lower()


def test_api_job_spec_runs_in_the_evaluator_and_results_are_served(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Producer -> consumer: the API's job file runs through the real runner and harness,
    and the files it writes validate against the typed result routes."""
    import cv2
    import numpy as np

    from scripts.curation.bakeoff import bakeoff_runner, run
    from scripts.curation.bakeoff.backends import registry
    from scripts.curation.bakeoff.backends.base import Detection

    class _Fake:
        """Names its classes cat/dog; a dark image holds a dog, a bright one a cat."""

        runtime = 'fake'
        class_names: dict[int, str] | None = {0: 'cat', 1: 'dog'}

        def __init__(self, name: str) -> None:
            self.name = name

        def detect(self, image_rgb: Any) -> list[Detection]:
            mean = float(image_rgb.mean())
            if mean < 50:
                return [Detection(20, 20, 60, 60, 0.9, class_id=1)]
            if mean > 150:
                return [Detection(20, 20, 60, 60, 0.8, class_id=0)]
            return []

    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))
    registry.register_backend('fake-scripted', lambda args: _Fake(args.name))

    def in_process_task(ds_path, ds_out, ds_id, model, gpu):
        argv = bakeoff_runner._model_argv(ds_path, ds_out, ds_id, model)
        assert run.main([*argv[3:], '--no-mlflow']) == 0
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', in_process_task)
    root = env['exports'] / 'e1'
    for stem, fill, rows in (('a', 10, '0'), ('b', 200, '1'), ('bg', 100, None)):
        img = root / 'images' / 'test' / f'{stem}.png'
        img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img), np.full((100, 100, 3), fill, dtype=np.uint8))
        lbl = root / 'labels' / 'test' / f'{stem}.txt'
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(f'{rows} 0.4 0.4 0.4 0.4\n' if rows else '')
    (root / 'data.yaml').write_text(_names_yaml(['dog', 'cat']))
    (root / 'manifest.json').write_text(json.dumps({'exported_at': '2026-09-25T00:00:00+00:00'}))
    r = _post_run(
        client,
        {
            'job_id': 'e2e',
            'datasets': [{'id': 'export:e1'}],
            'models': [
                {'source': 'custom', 'name': 'by-name', 'backend': 'fake-scripted', 'imgsz': 64},
                {
                    'source': 'custom',
                    'name': 'dog-only',
                    'backend': 'fake-scripted',
                    'imgsz': 64,
                    'class_map': {'1': 'dog'},
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    spec = json.loads((env['jobs'] / 'e2e.job.json').read_text())
    status = bakeoff_runner.run_job(spec)
    assert status['state'] == 'done', status

    st = client.get('/curation/projects/default/bakeoff/status/e2e').json()
    assert (st['state'], st['progress']) == ('done', {'done': 2, 'total': 2})
    comp = client.get('/curation/projects/default/bakeoff/results/e2e')
    assert comp.status_code == 200, comp.text
    body = comp.json()
    assert (body['dataset']['id'], body['rank_scope'], body['common_classes']) == (
        'export:e1',
        'common',
        [0],
    )
    by_key = {m['model']: m for m in body['models']}
    assert by_key['custom:by-name']['class_mapping']['method'] == 'names'
    assert by_key['custom:dog-only']['class_mapping']['method'] == 'explicit'
    assert by_key['custom:dog-only']['coverage']['not_covered'] == [
        {'eval_class_id': 1, 'name': 'cat'}
    ]
    matrix = client.get('/curation/projects/default/bakeoff/matrix/e2e')
    assert matrix.status_code == 200, matrix.text
    assert set(matrix.json()['cells']) == {'custom:by-name', 'custom:dog-only'}
