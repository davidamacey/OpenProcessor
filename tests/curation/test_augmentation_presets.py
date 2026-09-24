"""Augmentation presets: one catalog, served to clients, enforced before any
side effect.

The train form offered preset ids the trainer's ``augment.PRESETS`` didn't
have; the trainer only raised ``unknown augmentation preset`` after
``/train/start`` had already claimed the GPU (stopping the GPU-resident
service). The catalog now lives in
``src/services/training/augmentation_presets.py``: the API validates
against it and serves it, and the trainer image copies the same file and
builds ``PRESETS`` from it.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.training import augmentation_presets as catalog


REPO = Path(__file__).resolve().parents[2]
AUGMENT_PY = REPO / 'docker' / 'trainer' / 'augment.py'
TRAINER_DOCKERFILE = REPO / 'docker' / 'trainer' / 'Dockerfile'
CATALOG_PY = REPO / 'src' / 'services' / 'training' / 'augmentation_presets.py'

BAD = {'enabled': True, 'preset': 'no_such_preset'}


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    from src.services.training import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, '_state_dir', lambda: tmp_path)

    class _Reg:
        def load(self) -> Any:
            class _Snap:
                classes: list[Any] = []

            return _Snap()

        def get(self, _cid: int) -> Any:
            return None

    monkeypatch.setattr('src.routers.curation_train.get_class_registry', lambda: _Reg())
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as train_router

    app = FastAPI()
    app.include_router(train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        yield client


def _check(body: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in body['checks'] if c['name'] == name)


# ------------------------------------------------------------------ served


def test_presets_are_served_with_labels_and_descriptions(app_client: TestClient) -> None:
    r = app_client.get('/curation/train/augmentation_presets')
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [p['id'] for p in body['presets']]
    assert ids == list(catalog.PRESET_IDS)
    assert body['default'] == catalog.DEFAULT_AUGMENTATION_PRESET
    assert body['default'] in ids
    for p in body['presets']:
        assert p['label']
        assert p['description']
        assert isinstance(p['orientation_sensitive'], bool)


def test_presets_route_is_typed_in_the_contract(app_client: TestClient) -> None:
    spec = app_client.get('/openapi.json').json()
    op = spec['paths']['/curation/train/augmentation_presets']['get']
    ref = op['responses']['200']['content']['application/json']['schema']['$ref']
    schema = spec['components']['schemas'][ref.rsplit('/', 1)[-1]]
    assert set(schema['properties']) >= {'presets', 'default'}


# --------------------------------------------------------------- preflight


def test_preflight_blocks_an_unknown_preset(app_client: TestClient) -> None:
    body = app_client.post(
        '/curation/train/preflight',
        json={'dataset_export_dir': '/data/exports/x', 'augmentation': BAD},
    ).json()
    check = _check(body, 'augmentation_preset')
    assert check['severity'] == 'block'
    assert 'no_such_preset' in check['message']
    assert 'balanced_default' in check['message']
    assert check['detail']['valid_presets'] == list(catalog.PRESET_IDS)
    assert body['blocked'] is True


def test_preflight_passes_a_known_preset(app_client: TestClient) -> None:
    body = app_client.post(
        '/curation/train/preflight',
        json={
            'dataset_export_dir': '/data/exports/x',
            'augmentation': {'enabled': True, 'preset': 'low_light'},
        },
    ).json()
    assert _check(body, 'augmentation_preset')['severity'] == 'ok'


def test_preflight_ignores_the_preset_of_a_disabled_block(app_client: TestClient) -> None:
    body = app_client.post(
        '/curation/train/preflight',
        json={
            'dataset_export_dir': '/data/exports/x',
            'augmentation': {**BAD, 'enabled': False},
        },
    ).json()
    assert _check(body, 'augmentation_preset')['severity'] == 'ok'


# ------------------------------------------------ start: 422 before any claim


@pytest.mark.parametrize('force', ['false', 'true'])
def test_start_refuses_an_unknown_preset_before_claiming_the_gpu(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: str
) -> None:
    claim = AsyncMock()
    monkeypatch.setattr('src.services.training.gpu_arbiter.claim_gpus_for_training', claim)
    r = app_client.post(
        f'/curation/train/start?force={force}',
        json={'dataset_export_dir': '/data/exports/x', 'augmentation': BAD},
    )
    assert r.status_code == 422, r.text
    assert 'no_such_preset' in r.text
    claim.assert_not_called()
    assert list(tmp_path.glob('*.job.json')) == []


def test_start_campaign_refuses_an_unknown_preset_before_claiming_the_gpu(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim = AsyncMock()
    monkeypatch.setattr('src.services.training.gpu_arbiter.claim_gpus_for_training', claim)
    r = app_client.post(
        '/curation/train/start_campaign?force=true',
        json={
            'dataset_export_dir': '/data/exports/x',
            'runs': [{'profile': 'nano', 'model_size': 'n'}],
            'augmentation': BAD,
        },
    )
    assert r.status_code == 422, r.text
    assert 'no_such_preset' in r.text
    claim.assert_not_called()
    assert list(tmp_path.glob('*.job.json')) == []


# ------------------------------------------- one source of truth, both sides


def test_catalog_imports_only_the_standard_library() -> None:
    """The trainer image copies this file flat, without the rest of src/."""
    tree = ast.parse(CATALOG_PY.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split('.')[0])
    assert modules <= set(sys.stdlib_module_names) | {'__future__'}, modules


def test_trainer_image_ships_the_catalog_next_to_augment() -> None:
    dockerfile = TRAINER_DOCKERFILE.read_text()
    assert re.search(
        r'^COPY\b.*\bsrc/services/training/augmentation_presets\.py\s+\./augmentation_presets\.py',
        dockerfile,
        re.MULTILINE,
    ), 'trainer Dockerfile must copy the preset catalog into /app'


def test_trainer_builds_presets_from_the_catalog() -> None:
    """augment.py (not importable here: no albumentations on the host) takes
    its preset ids from the catalog and defines a ``preset_<id>`` factory
    for every one of them."""
    tree = ast.parse(AUGMENT_PY.read_text())
    imports_catalog = any(
        isinstance(n, ast.ImportFrom) and n.module == 'augmentation_presets' for n in ast.walk(tree)
    )
    assert imports_catalog
    factories = {
        n.name.removeprefix('preset_')
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith('preset_')
    }
    assert set(catalog.PRESET_IDS) <= factories
    literal_keys = {
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and k.value in factories
    }
    assert not literal_keys, f'augment.py still hardcodes preset ids: {literal_keys}'
