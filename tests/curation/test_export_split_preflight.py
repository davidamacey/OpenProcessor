"""Training preflight blocks an export that can't train: a split with no
instances overall, or a class the run trains on with nothing in train or
val.

``export_not_empty`` alone counted total images, so an export of
``train=0, val=0, test=174`` passed preflight as "All checks passed".
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import dataset_thresholds as T  # noqa: N812
from src.services.curation.export_readiness import export_class_split_check, export_splits_check


if TYPE_CHECKING:
    from pathlib import Path


def _class_row(class_id: int, name: str, train: int, val: int, test: int) -> dict[str, Any]:
    return {
        'class_id': class_id,
        'export_id': class_id,
        'class_name': name,
        'train': train,
        'val': val,
        'test': test,
    }


BALANCED = [
    _class_row(0, 'alpha', 27, 3, 5),
    _class_row(1, 'beta', 27, 3, 5),
    _class_row(2, 'gamma', 27, 3, 5),
]


# -------------------------------------------------------------- pure checks


def test_splits_check_blocks_the_all_test_export_naming_the_empty_splits() -> None:
    sev, msg, detail = export_splits_check({'split_counts': {'train': 0, 'val': 0, 'test': 174}})
    assert sev == 'block'
    assert 'train' in msg
    assert 'val' in msg
    assert detail['split_counts'] == {'train': 0, 'val': 0, 'test': 174}
    assert detail['empty_splits'] == ['train', 'val']


def test_splits_check_blocks_val_only_gap() -> None:
    sev, msg, detail = export_splits_check({'split_counts': {'train': 90, 'val': 0, 'test': 10}})
    assert sev == 'block'
    assert detail['empty_splits'] == ['val']
    assert "'val'" in msg


def test_splits_check_ok_and_unknown() -> None:
    assert export_splits_check({'split_counts': {'train': 9, 'val': 1, 'test': 0}})[0] == 'ok'
    assert export_splits_check({})[0] == 'unknown'
    assert export_splits_check({'image_count': 3})[0] == 'unknown'


def test_class_split_check_names_the_class_and_split() -> None:
    rows = [*BALANCED[:2], _class_row(2, 'gamma', 4, 0, 5), _class_row(3, 'delta', 0, 0, 5)]
    sev, msg, detail = export_class_split_check({'class_split_counts': rows}, None)
    assert sev == 'block'
    assert 'gamma' in msg
    assert 'delta' in msg
    assert 'alpha' not in msg
    gaps = {g['class_name']: g['missing_splits'] for g in detail['classes']}
    assert gaps == {'gamma': ['val'], 'delta': ['train', 'val']}


def test_class_split_check_only_judges_included_classes() -> None:
    rows = [*BALANCED[:2], _class_row(2, 'gamma', 4, 0, 5)]
    manifest = {'class_split_counts': rows}
    assert export_class_split_check(manifest, [0, 1])[0] == 'ok'
    sev, msg, _ = export_class_split_check(manifest, [1, 2])
    assert sev == 'block'
    assert 'gamma' in msg


def test_class_split_check_uses_the_threshold_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import export_readiness

    monkeypatch.setattr(export_readiness, 'MIN_VAL_INSTANCES_PER_CLASS', 4)
    sev, _msg, detail = export_class_split_check({'class_split_counts': BALANCED}, None)
    assert sev == 'block'
    assert len(detail['classes']) == 3
    assert T.MIN_TRAIN_INSTANCES_PER_CLASS == 1
    assert T.MIN_VAL_INSTANCES_PER_CLASS == 1


def test_class_split_check_unknown_without_per_class_counts() -> None:
    assert export_class_split_check({'split_counts': {'train': 1}}, None)[0] == 'unknown'


def test_class_split_check_ok_when_every_class_is_covered() -> None:
    sev, msg, _ = export_class_split_check({'class_split_counts': BALANCED}, None)
    assert sev == 'ok'
    assert '3' in msg


# ------------------------------------------------------ preflight end-to-end


@pytest.fixture
def train_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
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
    fake.indices.get_settings = AsyncMock(return_value={})

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as train_router

    app = FastAPI()
    app.include_router(train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _export(tmp_path: Path, **manifest: Any) -> str:
    d = tmp_path / 'multi'
    (d / 'labels' / 'train').mkdir(parents=True)
    (d / 'manifest.json').write_text(json.dumps(manifest))
    (d / 'class_registry.json').write_text(
        json.dumps({'export_id_map': {str(i): i for i in range(3)}})
    )
    return str(d)


def _check(body: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in body['checks'] if c['name'] == name)


def test_preflight_blocks_the_all_test_export(train_client: TestClient, tmp_path: Path) -> None:
    export_dir = _export(
        tmp_path,
        image_count=174,
        split_counts={'train': 0, 'val': 0, 'test': 174},
        class_split_counts=[_class_row(i, n, 0, 0, 35) for i, n in enumerate('abc')],
    )
    body = train_client.post(
        '/curation/train/preflight', json={'dataset_export_dir': export_dir, 'profile': 'medium'}
    ).json()
    assert _check(body, 'export_not_empty')['severity'] == 'ok'
    splits = _check(body, 'export_splits_nonempty')
    assert splits['severity'] == 'block'
    assert 'train' in splits['message']
    assert _check(body, 'export_class_split_coverage')['severity'] == 'block'
    assert body['blocked'] is True


def test_preflight_blocks_a_per_class_gap_on_an_included_class(
    train_client: TestClient, tmp_path: Path
) -> None:
    export_dir = _export(
        tmp_path,
        image_count=100,
        split_counts={'train': 58, 'val': 6, 'test': 15},
        class_split_counts=[*BALANCED[:2], _class_row(2, 'gamma', 4, 0, 5)],
    )
    body = train_client.post(
        '/curation/train/preflight',
        json={'dataset_export_dir': export_dir, 'profile': 'medium', 'include_classes': [1, 2]},
    ).json()
    assert _check(body, 'export_splits_nonempty')['severity'] == 'ok'
    coverage = _check(body, 'export_class_split_coverage')
    assert coverage['severity'] == 'block'
    assert 'gamma' in coverage['message']
    assert 'val' in coverage['message']
    assert body['blocked'] is True

    ok_body = train_client.post(
        '/curation/train/preflight',
        json={'dataset_export_dir': export_dir, 'profile': 'medium', 'include_classes': [0, 1]},
    ).json()
    assert _check(ok_body, 'export_class_split_coverage')['severity'] == 'ok'


def test_preflight_single_class_export_skips_per_class_coverage(
    train_client: TestClient, tmp_path: Path
) -> None:
    d = tmp_path / 'single'
    d.mkdir()
    (d / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'single_class',
                'class_name': 'region',
                'positive_images': 20,
                'image_count': 23,
                'split_counts': {'train': 21, 'val': 0, 'test': 2},
            }
        )
    )
    body = train_client.post(
        '/curation/train/preflight', json={'dataset_export_dir': str(d), 'profile': 'medium'}
    ).json()
    assert _check(body, 'export_splits_nonempty')['severity'] == 'block'
    assert _check(body, 'export_class_split_coverage')['severity'] == 'ok'
