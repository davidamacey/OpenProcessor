"""Tests for the bake-off API (``src/routers/curation/bakeoff.py``) and its services.

Filesystem-driven: ``tmp_path`` stands in for the export root, the external
eval root, the training-jobs dir and the bake-off jobs/output dirs. Exports
are tiny YOLO trees built by :func:`make_export`; training runs are a
``status.json`` + ``manifest.json`` pair in the training-jobs dir, which is
exactly what the trainer leaves behind.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.curation.bakeoff.freeze import test_sha as freeze_test_sha
from src.services.curation.export_support import frozen_test_sha_of


REPO_ROOT = Path(__file__).resolve().parents[2]

# CFG-6: this harness used to default to owner-private absolute paths --
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
        'src/routers/curation/_bakeoff_models.py',
        'src/services/curation/eval_datasets.py',
        'src/services/curation/bakeoff_jobs.py',
    ]
    for root in ('scripts/curation/bakeoff', 'examples/bakeoff'):
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
# Fixtures + builders
# =============================================================================


def _names_yaml(names: list[str]) -> str:
    return 'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names))


def make_export(
    root: Path,
    rel: str,
    *,
    nc: int,
    test_labels: dict[str, list[int]],
    export_id_map: dict[int, int] | None = None,
    manifest: dict[str, Any] | None = None,
    names: list[str] | None = None,
    train_labels: dict[str, list[int]] | None = None,
    background: tuple[str, ...] = (),
) -> Path:
    """A minimal YOLO export under ``root/rel``.

    ``test_labels`` / ``train_labels`` map an image stem to the class ids of
    its boxes. ``background`` stems get an image and no label file.
    """
    d = root / rel
    for split, labels in (('test', test_labels), ('train', train_labels or {})):
        (d / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (d / 'images' / split).mkdir(parents=True, exist_ok=True)
        for stem, classes in labels.items():
            (d / 'labels' / split / f'{stem}.txt').write_text(
                ''.join(f'{c} 0.5 0.5 0.2 0.2\n' for c in classes)
            )
            (d / 'images' / split / f'{stem}.jpg').write_bytes(b'img')
    for stem in background:
        (d / 'images' / 'test' / f'{stem}.jpg').write_bytes(b'img')
    names = names or [f'class_{i}' for i in range(nc)]
    (d / 'data.yaml').write_text(f'nc: {nc}\n' + _names_yaml(names))
    (d / 'class_registry.json').write_text(
        json.dumps({'export_id_map': {str(k): v for k, v in (export_id_map or {}).items()}})
    )
    (d / 'manifest.json').write_text(json.dumps(manifest or {}))
    return d


# The live 84-class export's shape (plan 1.2): 5 classes present in test.
LIVE_NAMES = [f'class_{i}' for i in range(84)]
for _eid, _n in {37: 'miata', 38: 'minicooper', 43: 'mustang', 51: 'porsche', 78: 'vw'}.items():
    LIVE_NAMES[_eid] = _n
LIVE_EXPORT_ID_MAP = {r: r - 1 for r in range(1, 85)}  # registry r -> export r-1
LIVE_REMAP = {
    'original_to_new': {'38': 0, '39': 1, '44': 2, '52': 3, '79': 4},
    'new_to_original': {'0': 38, '1': 39, '2': 44, '3': 52, '4': 79},
    'single_cls': False,
    'names': ['miata', 'minicooper', 'mustang', 'porsche', 'vw'],
    'include_classes': [38, 39, 44, 52, 79],
}


def _live_test_labels() -> dict[str, list[int]]:
    return {f'img_{eid}_{k}': [eid] for eid in (37, 38, 43, 51, 78) for k in range(5)}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Point every bake-off / training dir at tmp_path and stub the GPU arbiter."""
    import src.services.training.gpu_arbiter as arbiter
    from src.routers.curation import bakeoff
    from src.services.curation import bakeoff_jobs, eval_datasets

    dirs = {
        'exports': tmp_path / 'exports',
        'external': tmp_path / 'bakeoff_eval',
        'train_jobs': tmp_path / 'train_jobs',
        'runs': tmp_path / 'runs',
        'jobs': tmp_path / 'bakeoff_jobs',
        'out': tmp_path / 'bakeoff_out',
    }
    for key in ('exports', 'external', 'train_jobs', 'runs'):
        dirs[key].mkdir()
    monkeypatch.setattr(eval_datasets, 'EXPORT_ROOT', dirs['exports'])
    monkeypatch.setattr(eval_datasets, 'EXTERNAL_ROOT', dirs['external'])
    monkeypatch.setattr(bakeoff_jobs, 'RUNS_HOST_ROOT', dirs['runs'])
    monkeypatch.setattr(bakeoff, 'JOBS_DIR', dirs['jobs'])
    monkeypatch.setattr(bakeoff, 'OUT_DIR', dirs['out'])
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(dirs['train_jobs']))
    for key in [k for k in __import__('os').environ if k.startswith('OP_BAKEOFF_PROFILE')]:
        monkeypatch.delenv(key)
    eval_datasets.clear_cache()

    class _Action:
        action = 'noop'

    async def _stop(**_kw: Any) -> _Action:
        return _Action()

    monkeypatch.setattr(arbiter, 'stop_gpu_services', _stop)
    return dirs


@pytest.fixture
def client() -> TestClient:
    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    return TestClient(app)


def make_run(
    dirs: dict[str, Path],
    run_id: str,
    *,
    export_dir: Path,
    class_remap: dict[str, Any] | None = None,
    single_cls: bool = False,
    include_classes: list[int] | None = None,
    imgsz: int = 640,
    frozen_test_sha: str | None = None,
    eval_block: dict[str, Any] | None = None,
    state: str = 'finished',
) -> Path:
    """A finished training run as the trainer leaves it: status + manifest + best.pt."""
    ckpt = dirs['runs'] / run_id / 'weights' / 'best.pt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'pt')
    jobs = dirs['train_jobs']
    (jobs / f'{run_id}.status.json').write_text(
        json.dumps(
            {
                'job_id': run_id,
                'state': state,
                'finished_at': '2026-09-24T23:51:09Z',
                'checkpoint_path': str(ckpt),
                'eval': eval_block or {'map50': 0.9356, 'split': 'test'},
            }
        )
    )
    (jobs / f'{run_id}.manifest.json').write_text(
        json.dumps(
            {
                'kind': 'train',
                'job_id': run_id,
                'lineage': {
                    'export_dir': str(export_dir),
                    'dataset_sha': None,
                    'frozen_test_sha': frozen_test_sha,
                    'include_classes': include_classes,
                    'single_cls': single_cls,
                    'class_remap': class_remap,
                },
                'spec': {
                    'model_family': 'yolo26',
                    'model_size': 'n',
                    'hyperparameters': {'imgsz': imgsz},
                },
            }
        )
    )
    return ckpt


def _live_export(dirs: dict[str, Path], rel: str = '20260924T233203Z', **kw: Any) -> Path:
    kw.setdefault(
        'manifest',
        {
            'dataset_sha': 'd' * 64,
            'exported_at': '2026-09-24T23:32:11+00:00',
            'unlabeled_items_on_exported_images': 69,
        },
    )
    return make_export(
        dirs['exports'],
        rel,
        nc=84,
        names=LIVE_NAMES,
        test_labels=_live_test_labels(),
        export_id_map=LIVE_EXPORT_ID_MAP,
        **kw,
    )


# =============================================================================
# Route registration + typing
# =============================================================================


def test_bakeoff_router_is_registered() -> None:
    from src.main import app

    assert any(r.path == '/curation/bakeoff/runs' for r in app.routes)
    assert any(r.path == '/curation/bakeoff/eval_datasets' for r in app.routes)


def test_bakeoff_routes_have_typed_response_models() -> None:
    from src.main import app

    routes = [r for r in app.routes if getattr(r, 'path', '').startswith('/curation/bakeoff/')]
    assert len(routes) == 9
    untyped = [r.path for r in routes if getattr(r, 'response_model', None) is None]
    assert not untyped, f'bake-off routes without response_model: {untyped}'


# =============================================================================
# GET /bakeoff/eval_datasets
# =============================================================================


def test_eval_datasets_lists_multiclass_export(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, background=('bg_0',))
    body = client.get('/curation/bakeoff/eval_datasets', params={'source': 'export'}).json()
    assert body['count'] == 1
    [row] = body['datasets']
    assert row['id'] == 'export:20260924T233203Z'
    assert (row['source'], row['group'], row['name']) == ('export', None, '20260924T233203Z')
    assert row['path'] == str(d)
    assert row['dataset_kind'] == 'multi_class'
    assert row['nc'] == 84
    assert [
        (c['eval_class_id'], c['name'], c['n_objects'], c['n_images']) for c in row['classes']
    ] == [
        (37, 'miata', 5, 5),
        (38, 'minicooper', 5, 5),
        (43, 'mustang', 5, 5),
        (51, 'porsche', 5, 5),
        (78, 'vw', 5, 5),
    ]
    assert row['classes'][0]['registry_class_id'] == 38
    assert (row['n_images'], row['n_objects'], row['n_background_images']) == (26, 25, 1)
    assert row['frozen_test_sha'] == frozen_test_sha_of(d)
    assert row['test_label_sha'] == freeze_test_sha(d)[0]
    assert row['sha_source'] == 'computed'
    assert row['dataset_sha'] == 'd' * 64
    assert row['unlabeled_items_on_exported_images'] == 69
    assert row['exported_at'] == '2026-09-24T23:32:11+00:00'
    assert row['frozen_ok'] is None
    assert row['is_current'] is False


def test_eval_datasets_sha_source_manifest(env: dict[str, Path], client: TestClient) -> None:
    make_export(
        env['exports'],
        'e1',
        nc=2,
        test_labels={'a': [0], 'b': [1]},
        manifest={'frozen_test_sha': 'f' * 16, 'test_label_sha': 'a' * 16},
    )
    [row] = client.get('/curation/bakeoff/eval_datasets').json()['datasets']
    assert (row['frozen_test_sha'], row['test_label_sha'], row['sha_source']) == (
        'f' * 16,
        'a' * 16,
        'manifest',
    )


def test_eval_datasets_lists_single_class_export_at_depth_two(
    env: dict[str, Path], client: TestClient
) -> None:
    profile_dir = env['exports'] / 'region_widget'
    make_export(
        env['exports'],
        'region_widget/20260924T032032Z',
        nc=1,
        names=['widget'],
        test_labels={'a': [0], 'b': [0, 0]},
        manifest={'dataset_kind': 'single_class', 'exported_at': '2026-09-24T03:20:32+00:00'},
    )
    (profile_dir / 'current').symlink_to(profile_dir / '20260924T032032Z')
    # A symlinked top-level "current" and a non-export dir are not listed.
    (env['exports'] / 'current').symlink_to(profile_dir / '20260924T032032Z')
    (env['exports'] / 'scratch').mkdir()
    rows = client.get('/curation/bakeoff/eval_datasets').json()['datasets']
    assert [r['id'] for r in rows] == ['export:region_widget/20260924T032032Z']
    [row] = rows
    assert row['is_current'] is True
    assert row['dataset_kind'] == 'single_class'
    assert (row['nc'], row['n_objects'], row['classes'][0]['registry_class_id']) == (1, 3, None)


def test_eval_datasets_external_kept_optional(env: dict[str, Path], client: TestClient) -> None:
    from scripts.curation.bakeoff.freeze import freeze

    good = make_export(env['external'], 'curated/set_a', nc=1, test_labels={'a': [0]})
    freeze(good)
    drifted = make_export(env['external'], 'public/set_b', nc=1, test_labels={'b': [0]})
    freeze(drifted)
    lbl = drifted / 'labels' / 'test' / 'b.txt'
    lbl.chmod(0o644)
    lbl.write_text('0 0.1 0.1 0.1 0.1\n')
    make_export(env['external'], 'sample/unfrozen', nc=1, test_labels={'c': [0]})

    body = client.get('/curation/bakeoff/eval_datasets', params={'source': 'external'}).json()
    by_id = {r['id']: r for r in body['datasets']}
    assert set(by_id) == {'external:curated/set_a', 'external:public/set_b'}
    assert by_id['external:curated/set_a']['group'] == 'curated'
    assert by_id['external:curated/set_a']['source'] == 'external'
    assert by_id['external:curated/set_a']['dataset_kind'] == 'external'
    assert by_id['external:curated/set_a']['frozen_ok'] is True
    assert by_id['external:public/set_b']['frozen_ok'] is False
    assert by_id['external:curated/set_a']['dataset_sha'] is None
    # Nothing under the export root: the export listing is empty, not an error.
    assert client.get('/curation/bakeoff/eval_datasets', params={'source': 'export'}).json() == {
        'datasets': [],
        'count': 0,
    }


# =============================================================================
# POST /bakeoff/run
# =============================================================================


def _post_run(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post('/curation/bakeoff/run', json=body)


def test_dataset_id_rejects_traversal(env: dict[str, Path], client: TestClient) -> None:
    make_export(env['exports'].parent, 'outside', nc=1, test_labels={'a': [0]})
    make_run(env, 'r1', export_dir=env['exports'].parent / 'outside')
    for bad in ('export:../outside', 'export:..', 'external:../../x', 'export:/etc', 'nope:x'):
        r = _post_run(
            client, {'datasets': [{'id': bad}], 'models': [{'source': 'run', 'run_id': 'r1'}]}
        )
        assert r.status_code == 400, (bad, r.text)
    assert not env['jobs'].exists()


def test_run_resolves_run_model_class_map(env: dict[str, Path], client: TestClient) -> None:
    from src.routers.curation._bakeoff_models import BakeoffJobSpec

    d = _live_export(env)
    ckpt = make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP, imgsz=640)
    r = _post_run(
        client,
        {
            'job_id': 'j1',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['status'], body['job_id'], body['profile']) == ('enqueued', 'j1', 'generic')
    assert body['datasets'] == [
        {
            'id': 'export:20260924T233203Z',
            'path': str(d),
            'frozen_test_sha': frozen_test_sha_of(d),
            'test_label_sha': freeze_test_sha(d)[0],
            'n_eval_classes': 5,
        }
    ]
    [m] = body['models']
    assert (m['model'], m['display_name'], m['source']) == ('run:run5', 'run5', 'run')
    cm = m['class_mapping']['export:20260924T233203Z']
    assert cm['method'] == 'run_class_remap'
    assert cm['model_to_eval'] == {'0': 37, '1': 38, '2': 43, '3': 51, '4': 78}
    assert cm['unmapped_model_classes'] == []
    assert cm['not_covered_eval_classes'] == []
    assert m['train_test_overlap'] == {'export:20260924T233203Z': {'n_images': 0, 'fraction': 0.0}}

    spec = BakeoffJobSpec.model_validate(json.loads((env['jobs'] / 'j1.job.json').read_text()))
    assert spec.schema_version == 2
    assert spec.out_dir == str(env['out'] / 'j1')
    [ds] = spec.datasets
    assert (ds.id, ds.dir_name, ds.path) == (
        'export:20260924T233203Z',
        'export__20260924T233203Z',
        str(d),
    )
    assert ds.eval_class_ids == [37, 38, 43, 51, 78]
    assert ds.test_label_sha == freeze_test_sha(d)[0]
    [jm] = spec.models
    assert (jm.model, jm.backend, jm.weights, jm.imgsz, jm.mode) == (
        'run:run5',
        'ultralytics',
        str(ckpt),
        640,
        'full',
    )
    assert jm.class_map_by_dataset['export:20260924T233203Z']['model_to_eval'] == {
        '0': 37,
        '1': 38,
        '2': 43,
        '3': 51,
        '4': 78,
    }


def test_run_subset_model_reports_not_covered_at_enqueue(
    env: dict[str, Path], client: TestClient
) -> None:
    d = _live_export(env)
    remap3 = {
        'original_to_new': {'38': 0, '39': 1, '44': 2},
        'new_to_original': {'0': 38, '1': 39, '2': 44},
        'single_cls': False,
        'names': ['miata', 'minicooper', 'mustang'],
        'include_classes': [38, 39, 44],
    }
    make_run(env, 'run3', export_dir=d, class_remap=remap3, include_classes=[38, 39, 44])
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run3', 'display_name': '3-class subset'}],
        },
    )
    assert r.status_code == 200, r.text
    [m] = r.json()['models']
    assert m['display_name'] == '3-class subset'
    cm = m['class_mapping']['export:20260924T233203Z']
    assert cm['model_to_eval'] == {'0': 37, '1': 38, '2': 43}
    assert cm['not_covered_eval_classes'] == [
        {'eval_class_id': 51, 'name': 'porsche'},
        {'eval_class_id': 78, 'name': 'vw'},
    ]


def test_run_full_export_model_maps_by_registry_ids(
    env: dict[str, Path], client: TestClient
) -> None:
    d = _live_export(env)
    make_run(env, 'full', export_dir=d)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'full'}],
        },
    )
    assert r.status_code == 200, r.text
    cm = r.json()['models'][0]['class_mapping']['export:20260924T233203Z']
    assert cm['method'] == 'registry_ids'
    assert {k: cm['model_to_eval'][k] for k in ('37', '51')} == {'37': 37, '51': 51}


def test_run_rejects_multiclass_single_cls_run(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    remap = {
        'original_to_new': {'38': 0, '39': 0},
        'new_to_original': {},
        'single_cls': True,
        'names': ['object'],
        'include_classes': [38, 39],
    }
    make_run(env, 'sc', export_dir=d, class_remap=remap, single_cls=True, include_classes=[38, 39])
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'sc'}],
        },
    )
    assert r.status_code == 422
    assert r.json()['detail'] == (
        'single_cls run over 2 classes cannot be scored per class; '
        'compare it on a single-class export'
    )
    assert not env['jobs'].exists()


def test_run_alias_run_dataset_id(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {'datasets': [{'id': 'run:run5'}], 'models': [{'source': 'run', 'run_id': 'run5'}]},
    )
    assert r.status_code == 200, r.text
    assert [ds['id'] for ds in r.json()['datasets']] == ['export:20260924T233203Z']

    make_run(env, 'elsewhere', export_dir=env['exports'].parent / 'not_under_root')
    r = _post_run(
        client,
        {'datasets': [{'id': 'run:elsewhere'}], 'models': [{'source': 'run', 'run_id': 'run5'}]},
    )
    assert r.status_code == 400
    assert 'export root' in r.json()['detail']


def test_run_unknown_run_or_baseline_400(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    ds = [{'id': 'export:20260924T233203Z'}]
    for model, needle in (
        ({'source': 'run', 'run_id': 'no_such_run'}, 'no_such_run'),
        ({'source': 'baseline', 'name': 'no_such_baseline'}, 'no_such_baseline'),
    ):
        r = _post_run(client, {'datasets': ds, 'models': [model]})
        assert r.status_code == 400, r.text
        assert needle in r.json()['detail']
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:missing'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 400
    assert 'export:missing' in r.json()['detail']
    assert not env['jobs'].exists()


def test_run_requires_models_and_datasets(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    r = _post_run(client, {'datasets': [{'id': 'export:20260924T233203Z'}]})
    assert r.status_code == 400
    r = _post_run(client, {'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}]})
    assert r.status_code == 400


def test_run_rejects_removed_v1_fields(env: dict[str, Path], client: TestClient) -> None:
    for body in (
        {'dataset': '/d', 'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}]},
        {'datasets': [{'path': '/d'}], 'models': []},
        {'datasets': [{'id': 'export:x'}], 'models': [{'backend': 'ultralytics', 'name': 'm'}]},
        {'datasets': [{'id': 'export:x'}], 'quantize': {'run_id': 'r', 'coreml': True}},
    ):
        assert _post_run(client, body).status_code == 422, body


def test_run_duplicate_model_keys_400(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [
                {'source': 'run', 'run_id': 'run5'},
                {'source': 'run', 'run_id': 'run5', 'display_name': 'again'},
            ],
        },
    )
    assert r.status_code == 400
    assert 'run:run5' in r.json()['detail']


def test_run_custom_model_explicit_and_name_mapping(
    env: dict[str, Path], client: TestClient
) -> None:
    _live_export(env)
    r = _post_run(
        client,
        {
            'job_id': 'jc',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [
                {
                    'source': 'custom',
                    'name': 'served',
                    'backend': 'triton',
                    'triton_model': 'my_detector',
                    'class_map': {'0': 'miata', '1': 'VW'},
                },
                {'source': 'custom', 'name': 'coco', 'backend': 'ultralytics', 'weights': 'w.pt'},
            ],
        },
    )
    assert r.status_code == 200, r.text
    by_key = {m['model']: m for m in r.json()['models']}
    explicit = by_key['custom:served']['class_mapping']['export:20260924T233203Z']
    assert (explicit['method'], explicit['model_to_eval']) == ('explicit', {'0': 37, '1': 78})
    names = by_key['custom:coco']['class_mapping']['export:20260924T233203Z']
    assert (names['method'], names['model_to_eval']) == ('names', None)
    assert names['warnings'] == ['resolved by name inside the evaluator']
    assert by_key['custom:coco']['train_test_overlap'] == {'export:20260924T233203Z': None}
    spec = json.loads((env['jobs'] / 'jc.job.json').read_text())
    coco = next(m for m in spec['models'] if m['model'] == 'custom:coco')
    assert coco['class_map_by_dataset'] == {'export:20260924T233203Z': None}
    served = next(m for m in spec['models'] if m['model'] == 'custom:served')
    assert served['triton_model'] == 'my_detector'


def test_run_quantize_block_filled_from_run(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    ckpt = make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP, imgsz=512)
    r = _post_run(
        client,
        {
            'job_id': 'run5_quant',
            'datasets': [{'id': 'run:run5'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
            'quantize': {'run_id': 'run5', 'throughput': True},
        },
    )
    assert r.status_code == 200, r.text
    q = json.loads((env['jobs'] / 'run5_quant.job.json').read_text())['quantize']
    assert q['model_key_prefix'] == 'run:run5'
    assert (q['checkpoint'], q['imgsz'], q['calib_dataset']) == (str(ckpt), 512, str(d))
    assert q['formats'] == ['fp32_onnx', 'fp16_onnx', 'int8_onnx']
    assert (q['n_calib'], q['calib_split'], q['throughput']) == (1000, 'train', True)
    assert q['out_root'] == str(env['out'] / 'run5_quant' / 'quant')
    assert q['class_map_by_dataset']['export:20260924T233203Z']['model_to_eval']['4'] == 78


def test_run_writes_queued_status(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'job_id': 'jq',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    st = client.get('/curation/bakeoff/status/jq')
    assert st.status_code == 200
    body = st.json()
    assert body['schema_version'] == 2
    assert body['state'] == 'queued'
    assert body['datasets'] == ['export:20260924T233203Z']
    assert body['models'] == ['run:run5']
    assert body['progress'] == {'done': 0, 'total': 1}
    runs = client.get('/curation/bakeoff/runs').json()['runs']
    assert [(x['job_id'], x['state']) for x in runs] == [('jq', 'queued')]
    # Re-using a job id would overwrite that job's results.
    again = _post_run(
        client,
        {
            'job_id': 'jq',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert again.status_code == 409


def test_run_stop_failure_returns_409_and_removes_job(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.services.training.gpu_arbiter as arbiter

    async def _fail(**_kw: Any) -> None:
        raise arbiter.GpuArbiterStopFailedError('docker socket unavailable')

    monkeypatch.setattr(arbiter, 'stop_gpu_services', _fail)
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'job_id': 'jf',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 409
    assert 'docker socket unavailable' in r.json()['detail']
    assert not (env['jobs'] / 'jf.job.json').exists()
    assert list(env['jobs'].glob('*.job.json')) == []
    st = client.get('/curation/bakeoff/status/jf').json()
    assert st['state'] == 'error'
    assert 'docker socket unavailable' in st['error']


# =============================================================================
# GET /bakeoff/trained_models
# =============================================================================


def test_trained_models_for_dataset(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, train_labels={'img_37_0': [37], 'train_only': [38]})
    make_run(
        env,
        'run5',
        export_dir=d,
        class_remap=LIVE_REMAP,
        frozen_test_sha=frozen_test_sha_of(d),
        eval_block={'map50': 0.9356, 'split': 'test'},
    )
    make_run(env, 'unfinished', export_dir=d, state='running')
    body = client.get(
        '/curation/bakeoff/trained_models', params={'dataset_id': 'export:20260924T233203Z'}
    ).json()
    assert body['count'] == 1
    [m] = body['models']
    assert m['run_id'] == 'run5'
    assert (m['model_family'], m['model_size'], m['imgsz']) == ('yolo26', 'n', 640)
    assert m['train_export_id'] == 'export:20260924T233203Z'
    assert m['class_names'] == ['miata', 'minicooper', 'mustang', 'porsche', 'vw']
    assert m['single_cls'] is False
    assert (m['trainer_map50'], m['trainer_map50_split']) == (0.9356, 'test')
    fd = m['for_dataset']
    assert fd['dataset_id'] == 'export:20260924T233203Z'
    assert fd['same_export'] is True
    assert fd['same_frozen_test'] is True
    assert fd['n_classes_mapped'] == 5
    assert fd['train_test_overlap'] == {'n_images': 1, 'fraction': 1 / 25}

    plain = client.get('/curation/bakeoff/trained_models').json()['models'][0]
    assert plain['for_dataset'] is None


# =============================================================================
# GET /bakeoff/results, /status, /runs, /matrix
# =============================================================================


def _v2_comparison(job_id: str) -> dict[str, Any]:
    block = {
        'n_classes': 1,
        'map_50': 0.5,
        'map_50_95': 0.4,
        'map_75': 0.3,
        'ap_small': None,
        'ap_medium': None,
        'ap_large': 0.4,
        'precision': 1.0,
        'recall': 0.5,
        'f1': 0.667,
        'mean_iou': 0.8,
        'tp': 1,
        'fp': 0,
        'fn': 1,
    }
    common = {
        k: block[k]
        for k in ('n_classes', 'map_50', 'map_50_95', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn')
    }
    return {
        'schema_version': 2,
        'job_id': job_id,
        'profile': 'generic',
        'thresholds': {'conf_floor': 0.001, 'nms_iou': 0.7, 'op_conf': 0.25, 'op_iou': 0.45},
        'dataset': {
            'id': 'export:e1',
            'frozen_test_sha': 'f' * 16,
            'test_label_sha': 'a' * 16,
            'n_images': 2,
            'n_objects': 2,
            'n_background_images': 0,
        },
        'eval_classes': [{'eval_class_id': 0, 'name': 'a', 'n_gt': 2}],
        'common_classes': [0],
        'rank_by': 'map_50_95',
        'rank_scope': 'common',
        'models': [
            {
                'rank': 1,
                'model': 'run:r1',
                'display_name': 'r1',
                'source': 'run',
                'run_id': 'r1',
                'runtime': 'ultralytics',
                'imgsz': 640,
                'training_data': None,
                'overall': block,
                'common': common,
                'per_class': [
                    {
                        'eval_class_id': 0,
                        'name': 'a',
                        'n_gt': 2,
                        'covered': True,
                        'model_class_ids': [0],
                        'ap50': 0.5,
                        'ap50_95': 0.4,
                        'ap75': 0.3,
                        'precision': 1.0,
                        'recall': 0.5,
                        'f1': 0.667,
                        'tp': 1,
                        'fp': 0,
                        'fn': 1,
                    }
                ],
                'coverage': {
                    'n_eval_classes': 1,
                    'n_covered': 1,
                    'not_covered': [],
                    'unmapped_model_classes': [
                        {'model_class_id': 3, 'name': 'b', 'n_predictions': 4}
                    ],
                    'predictions_outside_scored_classes': 0,
                },
                'class_mapping': {'method': 'run_class_remap', 'warnings': []},
                'train_test_overlap': {'n_images': 0, 'fraction': 0.0},
                'latency_ms': {'mean': 1.0, 'p50': 1.0, 'p90': 1.0, 'p99': 1.0},
                'fps': 1000.0,
                'size_mb': 5.4,
                'per_stratum': {},
                'test_frames': 2,
            }
        ],
        'failed': [{'model': 'baseline:x', 'error': 'boom'}],
        'warnings': [],
        'n_models': 1,
    }


def _write_job_out(out: Path, job_id: str, *, status: dict[str, Any] | None = None) -> Path:
    job = out / job_id
    job.mkdir(parents=True)
    (job / 'status.json').write_text(
        json.dumps(
            status
            or {
                'schema_version': 2,
                'job_id': job_id,
                'state': 'done',
                'profile': 'generic',
                'datasets': ['export:e1', 'external:curated/x'],
                'models': ['run:r1'],
                'started_at': '2026-09-25T01:00:00+00:00',
                'finished_at': '2026-09-25T01:05:00+00:00',
                'progress': {'done': 2, 'total': 2},
                'completed': [{'dataset': 'export:e1', 'model': 'run:r1'}],
                'failed': [],
                'error': None,
            }
        )
    )
    return job


def test_results_serves_v2_and_rejects_v1(env: dict[str, Path], client: TestClient) -> None:
    job = _write_job_out(env['out'], 'j2')
    (job / 'export__e1').mkdir()
    (job / 'export__e1' / 'comparison.json').write_text(json.dumps(_v2_comparison('j2')))
    (job / 'external__curated__x').mkdir()
    (job / 'external__curated__x' / 'comparison.json').write_text(
        json.dumps({**_v2_comparison('j2'), 'dataset': {'id': 'external:curated/x'}})
    )

    r = client.get('/curation/bakeoff/results/j2')  # defaults to the job's first dataset
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['dataset']['id'] == 'export:e1'
    row = body['models'][0]
    assert (row['rank'], row['common']['map_50_95'], row['coverage']['n_covered']) == (1, 0.4, 1)
    assert row['coverage']['unmapped_model_classes'][0]['n_predictions'] == 4
    r = client.get('/curation/bakeoff/results/j2', params={'dataset_id': 'external:curated/x'})
    assert r.status_code == 200
    assert r.json()['dataset']['id'] == 'external:curated/x'
    assert (
        client.get('/curation/bakeoff/results/j2', params={'dataset_id': 'export:nope'}).status_code
        == 404
    )

    v1 = _write_job_out(env['out'], 'j1')
    (v1 / 'export__e1').mkdir()
    (v1 / 'export__e1' / 'comparison.json').write_text(
        json.dumps({'models': [{'rank': 1, 'name': 'm', 'map_50': 0.5}], 'rank_by': 'map_50'})
    )
    r = client.get('/curation/bakeoff/results/j1')
    assert r.status_code == 409
    assert r.json()['detail'] == (
        'bake-off result comparison.json has an unsupported schema (schema_version != 2)'
    )


def test_results_404_when_missing(env: dict[str, Path], client: TestClient) -> None:
    assert client.get('/curation/bakeoff/results/no-such-job').status_code == 404
    assert client.get('/curation/bakeoff/status/no-such-job').status_code == 404
    assert client.get('/curation/bakeoff/matrix/no-such-job').status_code == 404


def test_runs_skip_v1_and_sort_newest_first(env: dict[str, Path], client: TestClient) -> None:
    assert client.get('/curation/bakeoff/runs').json() == {'runs': []}
    _write_job_out(env['out'], 'older')
    newer = _write_job_out(env['out'], 'newer')
    st = json.loads((newer / 'status.json').read_text())
    st.update(job_id='newer', started_at='2026-09-26T00:00:00+00:00')
    (newer / 'status.json').write_text(json.dumps(st))
    legacy = env['out'] / 'legacy'
    legacy.mkdir()
    (legacy / 'status.json').write_text(json.dumps({'state': 'done', 'models': []}))
    runs = client.get('/curation/bakeoff/runs').json()['runs']
    assert [r['job_id'] for r in runs] == ['newer', 'older']
    assert runs[0]['datasets'] == ['export:e1', 'external:curated/x']


def test_matrix_serves_v2(env: dict[str, Path], client: TestClient) -> None:
    job = _write_job_out(env['out'], 'jm')
    cell = {
        'map_50': 0.5,
        'map_50_95': 0.4,
        'precision': 1.0,
        'recall': 0.5,
        'f1': 0.667,
        'latency_ms': 1.0,
        'size_mb': 5.4,
        'coverage': 1.0,
        'rank': 1,
    }
    (job / 'matrix.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'job_id': 'jm',
                'rank_by': 'map_50_95',
                'datasets': [
                    {
                        'id': 'export:e1',
                        'frozen_test_sha': None,
                        'test_label_sha': 'a',
                        'rank_scope': 'common',
                        'n_common_classes': 1,
                    }
                ],
                'models': [{'model': 'run:r1', 'display_name': 'r1', 'source': 'run'}],
                'metrics': ['map_50', 'map_50_95'],
                'cells': {'run:r1': {'export:e1': cell}},
                'best': {'export:e1': {'map_50_95': ['run:r1']}},
            }
        )
    )
    body = client.get('/curation/bakeoff/matrix/jm').json()
    assert body['best'] == {'export:e1': {'map_50_95': ['run:r1']}}
    assert body['cells']['run:r1']['export:e1']['coverage'] == 1.0
    (job / 'matrix.json').write_text(json.dumps({'cells': {}, 'best': {'x': {'map_50': 'm'}}}))
    assert client.get('/curation/bakeoff/matrix/jm').status_code == 409


# =============================================================================
# GET /bakeoff/profiles, /bakeoff/baseline_models
# =============================================================================


def test_profiles_lists_generic_only_by_default(env: dict[str, Path], client: TestClient) -> None:
    r = client.get('/curation/bakeoff/profiles')
    assert r.status_code == 200
    body = r.json()
    assert [p['name'] for p in body['profiles']] == ['generic']
    [row] = body['profiles']
    assert (row['kind'], row['default'], row['class_filter']) == ('registered', True, [])
    assert row['rank_metric'] == 'map_50_95'
    assert row['context_class_ids'] == []
    assert 'target_class_id' not in row
    assert 'class_names' not in row
    assert (body['default_profile'], body['default_error'], body['count']) == ('generic', None, 1)


def test_profiles_configured_json_default_row(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / 'widgets.json'
    f.write_text(json.dumps({'name': 'widgets', 'class_filter': ['widget']}))
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', str(f))
    body = client.get('/curation/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['class_filter']) == ('widgets', 'configured', ['widget'])
    assert body['count'] == len(body['profiles']) == 2

    monkeypatch.delenv('OP_BAKEOFF_PROFILE')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '4')
    body = client.get('/curation/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['context_class_ids']) == ('generic', 'registered', [4])


def test_profiles_bad_default_is_reported(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', 'no_such_profile')
    body = client.get('/curation/bakeoff/profiles').json()
    assert body['default_profile'] is None
    assert 'unknown bake-off profile' in body['default_error']
    assert not any(p['default'] for p in body['profiles'])


def test_run_rejects_unknown_profile(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    r = _post_run(
        client,
        {
            'profile': 'no_such_profile',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 400
    assert 'unknown bake-off profile' in r.json()['detail']


def test_run_profile_class_filter_narrows_scored_classes(
    env: dict[str, Path], client: TestClient, tmp_path: Path
) -> None:
    _live_export(env)
    prof = tmp_path / 'two.json'
    prof.write_text(json.dumps({'name': 'two', 'class_filter': ['Miata', 'vw']}))
    r = _post_run(
        client,
        {
            'job_id': 'jp',
            'profile': str(prof),
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()['profile'] == 'two'
    assert r.json()['datasets'][0]['n_eval_classes'] == 2
    spec = json.loads((env['jobs'] / 'jp.job.json').read_text())
    assert spec['profile'] == str(prof)
    assert spec['datasets'][0]['eval_class_ids'] == [37, 78]


def test_default_baseline_registry_is_empty(env: dict[str, Path], client: TestClient) -> None:
    body = client.get('/curation/bakeoff/baseline_models').json()
    assert body == {'baselines': [], 'count': 0}


def test_baseline_models_per_profile_and_run_lookup(
    env: dict[str, Path], client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import bakeoff_jobs

    reg = tmp_path / 'baselines.json'
    reg.write_text(
        json.dumps(
            {
                'baselines': [
                    {
                        'name': 'coco-yolo11n',
                        'backend': 'ultralytics',
                        'weights': './weights/yolo11n.pt',
                        'imgsz': 640,
                        'training_data': 'COCO',
                    },
                    {'name': 'bad entry!', 'backend': 'ultralytics'},
                ]
            }
        )
    )
    monkeypatch.setattr(bakeoff_jobs, 'BASELINES_PATH', reg)
    body = client.get('/curation/bakeoff/baseline_models').json()
    assert body['count'] == 1
    [b] = body['baselines']
    assert b == {
        'name': 'coco-yolo11n',
        'backend': 'ultralytics',
        'weights': './weights/yolo11n.pt',
        'imgsz': 640,
        'mode': 'full',
        'class_map': None,
        'backend_options': {},
        'training_data': 'COCO',
        'triton_model': None,
    }
    for bad in ('license_plate', 'nope', '../../etc/x.json'):
        r = client.get('/curation/bakeoff/baseline_models', params={'profile': bad})
        assert r.status_code == 400, bad
    assert (
        client.get('/curation/bakeoff/baseline_models', params={'profile': 'generic'}).json()[
            'count'
        ]
        == 1
    )

    _live_export(env)
    r = _post_run(
        client,
        {
            'job_id': 'jb',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'baseline', 'name': 'coco-yolo11n'}],
        },
    )
    assert r.status_code == 200, r.text
    [m] = json.loads((env['jobs'] / 'jb.job.json').read_text())['models']
    assert (m['model'], m['weights'], m['training_data']) == (
        'baseline:coco-yolo11n',
        './weights/yolo11n.pt',
        'COCO',
    )


def test_bakeoff_router_docstring_section_exists() -> None:
    from src.routers.curation import bakeoff

    doc = bakeoff.__doc__ or ''
    m = re.search(r'curation_design_rationale\.md`` §(\d+)', doc)
    assert m, 'router docstring should cite a specific rationale section'
    rationale = (REPO_ROOT / 'docs/design/curation_design_rationale.md').read_text()
    heading = re.search(rf'^## {m.group(1)}\. (.+)$', rationale, re.M)
    assert heading, f'rationale doc has no section {m.group(1)}'
    assert 'bake-off' in heading.group(1).lower()


def test_run_warns_about_train_test_overlap(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, train_labels={'img_37_0': [37], 'img_38_1': [38]})
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['models'][0]['train_test_overlap'] == {
        'export:20260924T233203Z': {'n_images': 2, 'fraction': 2 / 25}
    }
    assert body['warnings'] == [
        'run:run5: 2 of the test images of export:20260924T233203Z are in its '
        'training/validation splits'
    ]


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

    st = client.get('/curation/bakeoff/status/e2e').json()
    assert (st['state'], st['progress']) == ('done', {'done': 2, 'total': 2})
    comp = client.get('/curation/bakeoff/results/e2e')
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
    matrix = client.get('/curation/bakeoff/matrix/e2e')
    assert matrix.status_code == 200, matrix.text
    assert set(matrix.json()['cells']) == {'custom:by-name', 'custom:dog-only'}
