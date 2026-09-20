"""Tests for P1-12 (audit remediation phase 8): promote must build
``labels.txt`` from the class registry pinned at *submit* time, not the
live registry at promote time.

Before the fix, ``POST /curation/train/promote/{job_id}`` always called
``get_class_registry().load()`` — the live registry. If a class was renamed
between export and promote (a realistic gap: campaigns can run for hours),
the served model's ``labels.txt`` silently picked up the new name even
though the checkpoint was trained against the old one.

We don't exercise the real Triton handoff — ``promote_yolo26_to_triton`` is
mocked (same pattern as ``test_train_router.py``'s existing promote tests)
so the test is about which ``class_id_to_name`` map the router computes and
hands to it, not the filesystem/network side effects.
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
    app.include_router(curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client


def _renamed_live_registry():
    """A live ClassRegistry stand-in reflecting a rename that happened
    AFTER the job was submitted — class_id=5 is now 'suv_renamed'."""

    class _Entry:
        def __init__(self, class_id: int, class_name: str) -> None:
            self.class_id = class_id
            self.class_name = class_name
            self.deprecated = False

    class _Snapshot:
        classes = [_Entry(5, 'suv_renamed')]

    class _Reg:
        def load(self) -> _Snapshot:
            return _Snapshot()

    return _Reg()


def test_labels_txt_uses_pinned_registry_not_live(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'pin-job'

    # 1. The registry snapshot pinned at submit time (P1-12) — class_id=5
    #    was 'suv' when this job was submitted.
    snapshot_path = tmp_path / f'{job_id}.registry_snapshot.json'
    snapshot_path.write_text(
        json.dumps(
            {
                'version': 1,
                'updated_at': '2026-09-01T00:00:00Z',
                'classes': [{'class_id': 5, 'class_name': 'suv', 'deprecated': False}],
            }
        )
    )

    # 2. job.json carries the pin's location, as write_job would have written it.
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps(
            {
                'job_id': job_id,
                'dataset_export_dir': '/data/exports/x',
                'registry_snapshot_path': str(snapshot_path),
                'registry_sha': 'irrelevant-for-this-test',
            }
        )
    )

    # 3. The LIVE registry has since been renamed — class_id=5 is now
    #    'suv_renamed'. Before the fix, promote would follow this rename.
    monkeypatch.setattr(
        'src.routers.curation_train.get_class_registry',
        _renamed_live_registry,
    )

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

    async def _fake_promote(**kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        captured.update(kwargs)
        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_pin_test',
            onnx_path='/triton-models/yolo26m_pin_test/1/model.onnx',
            config_path='/triton-models/yolo26m_pin_test/config.pbtxt',
            labels_path='/triton-models/yolo26m_pin_test/labels.txt',
            triton_loaded=True,
        )

    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        f'/curation/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_pin_test'},
    )
    assert r.status_code == 200, r.text

    class_id_to_name = captured['class_id_to_name']
    assert class_id_to_name[5] == 'suv', (
        f'labels.txt must use the registry pinned at submit time (suv), '
        f'got {class_id_to_name[5]!r} — followed the live rename instead'
    )


def test_labels_txt_falls_back_to_live_registry_without_a_pin(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older runs (submitted before this fix shipped) have no
    registry_snapshot_path — promote must still work, falling back to the
    live registry rather than erroring."""
    job_id = 'no-pin-job'
    (tmp_path / f'{job_id}.job.json').write_text(
        json.dumps({'job_id': job_id, 'dataset_export_dir': '/data/exports/x'})
    )

    monkeypatch.setattr(
        'src.routers.curation_train.get_class_registry',
        _renamed_live_registry,
    )

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

    async def _fake_promote(**kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        captured.update(kwargs)
        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_no_pin',
            onnx_path='/triton-models/yolo26m_no_pin/1/model.onnx',
            config_path='/triton-models/yolo26m_no_pin/config.pbtxt',
            labels_path='/triton-models/yolo26m_no_pin/labels.txt',
            triton_loaded=True,
        )

    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        f'/curation/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_no_pin'},
    )
    assert r.status_code == 200, r.text
    assert captured['class_id_to_name'][5] == 'suv_renamed'
