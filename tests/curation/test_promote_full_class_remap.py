"""Root-cause fix: a full-class promote must build ``labels.txt`` from the
run's own ``class_remap.json`` (dense export id -> registry id/name), never
from the live/pinned registry keyed by registry id directly.

``_build_export_id_map`` (``src/services/curation/export_support.py``)
skips deprecated classes and renumbers the survivors ``0..N-1`` in ascending
registry-id order. A registry with any gap or deprecated class therefore
has dense id != registry id for every class after the gap -- a full-class
promote that used the registry directly (the old ``_resolve_full_registry_
for_promote`` path, still reachable as the legacy fallback for a run with
no resolvable remap) silently shifted every class name after the gap.

These tests exercise ``POST /curation/train/promote/{job_id}`` end to end
(``promote_yolo26_to_triton`` mocked, same pattern as
``test_promote_registry_pin.py``) so the assertion is on the exact
``class_id_to_name`` map the router hands to the promoter.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'hits': {'hits': [], 'total': {'value': 0}},
            'aggregations': {'by_class': {'buckets': []}},
        }
    )
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


@pytest.fixture
def app_client(
    fake_opensearch: AsyncMock,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as curation_train_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client


def _gapped_registry():
    """Registry ids {0, 2, 5} -- id 1 is deprecated (a middle gap)."""

    class _Entry:
        def __init__(self, class_id: int, class_name: str, deprecated: bool = False) -> None:
            self.class_id = class_id
            self.class_name = class_name
            self.deprecated = deprecated

    class _Snapshot:
        classes = [
            _Entry(0, 'car'),
            _Entry(1, 'retired_class', deprecated=True),
            _Entry(2, 'truck'),
            _Entry(5, 'van'),
        ]

    class _Reg:
        def load(self) -> _Snapshot:
            return _Snapshot()

    return _Reg()


def _mock_promote(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any], triton_name: str
) -> None:
    async def _fake_promote(**kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        captured.update(kwargs)
        status_obj = kwargs.get('status')
        return PromoteResult(
            job_id=status_obj.job_id if status_obj is not None else 'job',
            triton_name=triton_name,
            onnx_path=f'/triton-models/{triton_name}/1/model.onnx',
            config_path=f'/triton-models/{triton_name}/config.pbtxt',
            labels_path=f'/triton-models/{triton_name}/labels.txt',
            triton_loaded=True,
        )

    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )


def test_full_class_promote_uses_class_remap_over_a_registry_gap(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This must FAIL on pre-fix code: the trainer never wrote a full-class
    ``class_remap.json``, so ``resolve_class_remap`` always returned
    ``source='none'`` here and the router built ``labels.txt`` straight
    from the (gapped) registry -- dense id 1 (registry id 2, ``truck``)
    would have been labelled with registry id 1's name (the deprecated
    class) instead.
    """
    job_id = 'gap-job'
    checkpoint_dir = tmp_path / 'runs' / job_id / 'weights'
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / 'best.pt').write_bytes(b'not-a-real-checkpoint')
    # Dense export id -> registry id, exactly as write_full_class_remap
    # (docker/trainer/dataset_prep.py) would have produced for this export.
    (checkpoint_dir / 'class_remap.json').write_text(
        json.dumps(
            {
                'original_to_new': {'0': 0, '2': 1, '5': 2},
                'new_to_original': {'0': 0, '1': 2, '2': 5},
                'single_cls': False,
                'names': ['car', 'truck', 'van'],
                'include_classes': None,
            }
        )
    )
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps({'job_id': job_id, 'dataset_export_dir': '/data/exports/gap'})
    )

    monkeypatch.setattr('src.services.training.promote_gate.get_class_registry', _gapped_registry)

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=str(checkpoint_dir / 'best.pt'),
        eval={'map50': 0.90, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)

    captured: dict[str, Any] = {}
    _mock_promote(monkeypatch, captured, 'yolo26m_gap_test')

    r = app_client.post(
        '/curation/projects/default/train/promote/gap-job?wait=true',
        json={'triton_name': 'yolo26m_gap_test'},
    )
    assert r.status_code == 200, r.text

    class_id_to_name = captured['class_id_to_name']
    assert class_id_to_name == {0: 'car', 1: 'truck', 2: 'van'}


def test_full_class_promote_without_remap_refuses_when_registry_has_a_gap(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Older run: no class_remap.json anywhere. The identity map cannot be
    proven correct against a gapped registry -- refuse unless forced."""
    job_id = 'gap-no-remap-job'
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps({'job_id': job_id, 'dataset_export_dir': '/data/exports/gap'})
    )
    monkeypatch.setattr('src.services.training.promote_gate.get_class_registry', _gapped_registry)

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={'map50': 0.90, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)

    r = app_client.post(
        '/curation/projects/default/train/promote/gap-no-remap-job?wait=true',
        json={'triton_name': 'yolo26m_gap_no_remap'},
    )
    assert r.status_code == 422, r.text
    assert r.json()['detail']['failures'][0]['code'] == 'class_remap_missing_full_class'

    captured: dict[str, Any] = {}
    _mock_promote(monkeypatch, captured, 'yolo26m_gap_no_remap')
    r = app_client.post(
        '/curation/projects/default/train/promote/gap-no-remap-job?wait=true',
        json={'triton_name': 'yolo26m_gap_no_remap', 'force': True},
    )
    assert r.status_code == 200, r.text


def test_full_class_promote_without_remap_allowed_when_registry_is_contiguous(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Older run, no class_remap.json, but the pinned/live registry is
    provably gap-free (ids 0..N-1) -- the identity map IS correct here, so
    no force is required."""
    job_id = 'contiguous-no-remap-job'
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps({'job_id': job_id, 'dataset_export_dir': '/data/exports/contiguous'})
    )

    class _Entry:
        def __init__(self, class_id: int, class_name: str) -> None:
            self.class_id = class_id
            self.class_name = class_name
            self.deprecated = False

    class _Snapshot:
        classes = [_Entry(0, 'car'), _Entry(1, 'truck'), _Entry(2, 'van')]

    class _Reg:
        def load(self) -> _Snapshot:
            return _Snapshot()

    monkeypatch.setattr('src.services.training.promote_gate.get_class_registry', lambda: _Reg())

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={'map50': 0.90, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)

    captured: dict[str, Any] = {}
    _mock_promote(monkeypatch, captured, 'yolo26m_contiguous')

    r = app_client.post(
        '/curation/projects/default/train/promote/contiguous-no-remap-job?wait=true',
        json={'triton_name': 'yolo26m_contiguous'},
    )
    assert r.status_code == 200, r.text
    assert captured['class_id_to_name'] == {0: 'car', 1: 'truck', 2: 'van'}


def test_promote_force_is_accepted_as_a_query_parameter_too(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#128: ``?force=true`` (as on train/start) bypasses the gate like the body field."""
    job_id = 'gap-query-force-job'
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps({'job_id': job_id, 'dataset_export_dir': '/data/exports/gap'})
    )
    monkeypatch.setattr('src.services.training.promote_gate.get_class_registry', _gapped_registry)

    from src.services.training.jobs import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={'map50': 0.90, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    captured: dict[str, Any] = {}
    _mock_promote(monkeypatch, captured, 'yolo26m_qforce')

    url = f'/curation/projects/default/train/promote/{job_id}?wait=true'
    refused = app_client.post(url, json={'triton_name': 'yolo26m_qforce'})
    assert refused.status_code == 422, refused.text
    assert '?force=true' in refused.json()['detail']['override']

    r = app_client.post(url + '&force=true', json={'triton_name': 'yolo26m_qforce'})
    assert r.status_code == 200, r.text
    assert r.json()['force_used'] is True
