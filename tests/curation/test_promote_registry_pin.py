"""Tests for P1-12 (audit remediation phase 8): promote must build
``labels.txt`` from the class registry pinned at *submit* time, not the
live registry at promote time.

Before the fix, ``POST /curation/train/promote/{job_id}`` always called
``get_class_registry().load()`` — the live registry. If a class was renamed
between export and promote (a realistic gap: campaigns can run for hours),
the served model's ``labels.txt`` silently picked up the new name even
though the checkpoint was trained against the old one.

We don't exercise the real Triton handoff — ``promote_yolo26_to_triton`` is
mocked (same pattern as ``tests/curation/train_router/``'s existing promote tests)
so the test is about which ``class_id_to_name`` map the router computes and
hands to it, not the filesystem/network side effects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from _class_registry_fake import InMemoryRegistry, use_registry
from _project_paths import default_train_jobs_dir
from _train_run_fakes import install_run_fakes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._promote_start_fake import install as install_start_fake


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
    from src.routers.curation.train import router as curation_train_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client


def _renamed_live_registry() -> InMemoryRegistry:
    """The live registry after a rename that happened AFTER the job was submitted:
    class_id=5 is now 'suv_renamed'."""
    registry = InMemoryRegistry()
    registry.seed_rows([{'class_id': 5, 'class_name': 'suv_renamed'}])
    return use_registry(registry)


@pytest.fixture
def run_fakes(monkeypatch: pytest.MonkeyPatch):
    return install_run_fakes(monkeypatch)


def test_labels_txt_uses_pinned_registry_not_live(
    app_client: TestClient,
    run_fakes: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = 'pin-job'

    # 1. The registry snapshot pinned at submit time (P1-12) — class_id=5
    #    was 'suv' when this job was submitted.
    snapshot_path = default_train_jobs_dir(tmp_path) / f'{job_id}.registry_snapshot.json'
    snapshot_path.write_text(
        json.dumps(
            {
                'version': 1,
                'updated_at': '2026-09-01T00:00:00Z',
                'classes': [{'class_id': 5, 'class_name': 'suv', 'deprecated': False}],
            }
        )
    )

    # 2. the stored spec carries the pin's location, as start_run would have stored it.
    run_fakes.specs[job_id] = {
        'job_id': job_id,
        'dataset_export_dir': '/data/exports/x',
        'registry_snapshot_path': str(snapshot_path),
        'registry_sha': 'irrelevant-for-this-test',
    }

    # 3. The LIVE registry has since been renamed — class_id=5 is now
    #    'suv_renamed'. Before the fix, promote would follow this rename.
    _renamed_live_registry()

    from src.services.training.job_models import TrainJobStatus

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

    install_start_fake(monkeypatch, captured)

    # No class_remap.json here either (older run) and the single pinned id
    # (5) isn't contiguous from 0, so this also needs force -- the pin-vs-
    # live distinction under test is orthogonal to that gate.
    r = app_client.post(
        f'/curation/projects/default/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_pin_test', 'force': True},
    )
    assert r.status_code == 202, r.text

    class_id_to_name = captured['class_id_to_name']
    assert class_id_to_name[5] == 'suv', (
        f'labels.txt must use the registry pinned at submit time (suv), '
        f'got {class_id_to_name[5]!r} — followed the live rename instead'
    )


def test_labels_txt_falls_back_to_live_registry_without_a_pin(
    app_client: TestClient,
    run_fakes: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older runs (submitted before this fix shipped) have no
    registry_snapshot_path — promote must still work, falling back to the
    live registry rather than erroring."""
    job_id = 'no-pin-job'
    run_fakes.specs[job_id] = {'job_id': job_id, 'dataset_export_dir': '/data/exports/x'}

    _renamed_live_registry()

    from src.services.training.job_models import TrainJobStatus

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

    install_start_fake(monkeypatch, captured)

    # No class_remap.json (older run) and the pinned/live registry's single
    # id (5) is not contiguous from 0 -- the identity map is unprovable, so
    # this now 422s unless forced (the class-remap correctness fix).
    r = app_client.post(
        f'/curation/projects/default/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_no_pin'},
    )
    assert r.status_code == 422, r.text
    assert r.json()['detail']['failures'][0]['code'] == 'class_remap_missing_full_class'

    r = app_client.post(
        f'/curation/projects/default/train/promote/{job_id}',
        json={'triton_name': 'yolo26m_no_pin', 'force': True},
    )
    assert r.status_code == 202, r.text
    assert captured['class_id_to_name'][5] == 'suv_renamed'
