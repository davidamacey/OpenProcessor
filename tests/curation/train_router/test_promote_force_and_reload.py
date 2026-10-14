"""Forced promotion, manifest stamping failures and the reload_promoted route."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock


if TYPE_CHECKING:
    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# Phase 9 — P1-9 force=true must return the gate report + persist force_used
# =============================================================================


def test_force_promote_returns_gate_report(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the fix: the computed gate report is discarded on force=true —
    the 200 response carries no evidence the gate ever ran or what it found."""
    from pathlib import Path

    from src.services.training.job_models import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='force-report-job',
        state='finished',
        checkpoint_path='/jobs/force-report-job/best.pt',
        eval={'map50': 0.10, 'per_class': []},  # well below the 0.65 floor
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'force-report-job' else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id='force-report-job',
            triton_name='yolo26m_forced_report',
            onnx_path=str(Path('/triton-models/yolo26m_forced_report/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced_report/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced_report/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        '/curation/projects/default/train/promote/force-report-job?wait=true',
        json={'triton_name': 'yolo26m_forced_report', 'force': True},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['force_used'] is True
    assert out['gate_report'] is not None
    assert any('mAP50' in f['message'] for f in out['gate_report']['failures'])
    assert out['gate_report']['thresholds']['map50_min'] == 0.65


def test_force_promote_records_force_used_in_manifest(
    app_client: TestClient, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the fix: nothing about force=true reaches the manifest at all."""
    import json
    from pathlib import Path

    from src.services.training.job_models import TrainJobStatus

    job_id = 'force-manifest-job'
    (tmp_path / 'projects' / 'default' / f'{job_id}.manifest.json').write_text(
        json.dumps({'kind': 'train', 'job_id': job_id, 'promoted_to': None})
    )

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={'map50': 0.10, 'per_class': []},
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_forced_manifest',
            onnx_path=str(Path('/triton-models/yolo26m_forced_manifest/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced_manifest/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced_manifest/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )

    r = app_client.post(
        f'/curation/projects/default/train/promote/{job_id}?wait=true',
        json={'triton_name': 'yolo26m_forced_manifest', 'force': True},
    )
    assert r.status_code == 200, r.text
    assert r.json()['lineage_stamped'] is True

    manifest = json.loads(
        (tmp_path / 'projects' / 'default' / f'{job_id}.manifest.json').read_text()
    )
    promoted_to = manifest['promoted_to']
    assert promoted_to['force_used'] is True
    assert promoted_to['gate_report'] is not None
    assert any('mAP50' in f['message'] for f in promoted_to['gate_report']['failures'])


# =============================================================================
# Phase 9 — P1-10 a stamp_manifest_promotion failure must not be swallowed
# =============================================================================


def test_stamp_failure_is_not_swallowed(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: `except Exception: logger.warning(...)` around the
    stamp call means a real write failure is invisible — the caller gets a
    normal 200 with no signal that lineage was never recorded.

    ``configure_logging()`` only runs at FastAPI startup (src/main.py), not
    in pytest context, so structlog's default wrapper never reaches
    stdlib logging — ``caplog`` would silently capture nothing here. Use
    structlog's own test capture instead (same pattern as
    ``tests/integration/test_request_id_propagation.py``).
    """
    from pathlib import Path

    from structlog.testing import capture_logs

    from src.services.training.job_models import TrainJobStatus

    job_id = 'stamp-fail-job'

    fake_status = TrainJobStatus(
        job_id=job_id,
        state='finished',
        checkpoint_path=f'/jobs/{job_id}/best.pt',
        eval={
            'map50': 0.91,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.94, 'support': 1240},
            ],
        },
    )

    async def _fake_read_status(jid: str) -> TrainJobStatus | None:
        return fake_status if jid == job_id else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id=job_id,
            triton_name='yolo26m_stamp_fail',
            onnx_path=str(Path('/triton-models/yolo26m_stamp_fail/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_stamp_fail/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_stamp_fail/labels.txt')),
            triton_loaded=True,
        )

    async def _fake_stamp(*_args: Any, **_kwargs: Any) -> bool:
        msg = 'disk exploded'
        raise OSError(msg)

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton', _fake_promote
    )
    monkeypatch.setattr('src.services.training.jobs.stamp_manifest_promotion', _fake_stamp)

    with capture_logs() as cap:
        r = app_client.post(
            f'/curation/projects/default/train/promote/{job_id}?wait=true',
            json={'triton_name': 'yolo26m_stamp_fail'},
        )

    # Promote itself succeeded (the model is already live in Triton by this
    # point) — we don't turn that into a 500, but the response must say
    # lineage was NOT recorded, and the failure must be logged loudly (at
    # error, not the previous silent warning-and-forget).
    assert r.status_code == 200, r.text
    assert r.json()['lineage_stamped'] is False
    stamp_failure_logs = [e for e in cap if e.get('event') == 'manifest_stamp_failed']
    assert stamp_failure_logs, cap
    assert stamp_failure_logs[0]['log_level'] == 'error'
    assert 'disk exploded' in stamp_failure_logs[0]['error']


# =============================================================================
# /reload_promoted
# =============================================================================


def test_reload_promoted_route_returns_the_result_shape(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 3 (fresh-start E2E findings, round 2): make reload-promoted
    hits this route so an operator can force a reload right after
    bouncing Triton, without a full API restart."""
    fake_reload = AsyncMock(
        return_value={'status': 'ok', 'reloaded': ['op_v2'], 'failed': ['op_v3']}
    )
    monkeypatch.setattr('src.services.training.triton_reload.reload_promoted_models', fake_reload)

    r = app_client.post('/curation/projects/default/train/reload_promoted')

    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {'status': 'ok', 'reloaded': ['op_v2'], 'failed': ['op_v3']}
    fake_reload.assert_awaited_once()


def test_reload_promoted_route_defaults_to_empty_lists(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_reload = AsyncMock(return_value={'status': 'ok'})
    monkeypatch.setattr('src.services.training.triton_reload.reload_promoted_models', fake_reload)

    r = app_client.post('/curation/projects/default/train/reload_promoted')

    assert r.status_code == 200, r.text
    assert r.json() == {'status': 'ok', 'reloaded': [], 'failed': []}
