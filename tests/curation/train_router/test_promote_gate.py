"""The promote gate: metric thresholds, strictness and the promote/manifest endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# Promote gate (design §15.2)
# =============================================================================


def test_promote_gate_passes_when_metrics_meet_thresholds() -> None:
    from src.services.training.promote_gate import evaluate_promote_gate

    eval_block = {
        'map50': 0.91,
        'map50_95': 0.74,
        'per_class': [
            {'class_id': 12, 'name': 'pickup', 'precision': 0.94, 'support': 1240},
            {'class_id': 81, 'name': 'motorcycle', 'precision': 0.83, 'support': 412},
        ],
    }
    assert evaluate_promote_gate(eval_block) == []


def test_promote_gate_blocks_low_map50() -> None:
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate({'map50': 0.50, 'per_class': []})
    assert any('mAP50' in f.message for f in failures)
    assert any(f.code == 'map50_below_floor' for f in failures)


def test_promote_gate_blocks_low_class_precision() -> None:
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.30, 'support': 100},
            ],
        }
    )
    assert any('precision' in f.message for f in failures)
    assert any(f.code == 'per_class_precision_below_floor' for f in failures)


def test_promote_gate_blocks_low_support() -> None:
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.95, 'support': 2},
            ],
        }
    )
    assert any('support' in f.message for f in failures)
    assert any(f.code == 'per_class_support_below_floor' for f in failures)


def test_promote_gate_blocks_missing_eval_block() -> None:
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate(None)
    assert len(failures) == 1
    assert failures[0].code == 'no_eval_block'
    assert failures[0].message == 'no eval block in status.json — trainer never ran val()'


def test_promote_endpoint_returns_422_when_gate_fails(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: gate failure surfaces as 422 with structured detail."""
    from src.services.training.job_models import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='gate-fail-job',
        state='finished',
        checkpoint_path='/jobs/gate-fail-job/best.pt',
        eval={'map50': 0.50, 'per_class': []},  # below 0.65 floor
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'gate-fail-job' else None

    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        _fake_read_status,
    )

    r = app_client.post(
        '/curation/projects/default/train/promote/gate-fail-job?wait=true',
        json={'triton_name': 'yolo26m_fail'},
    )
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['message'] == 'promote gate failed'
    assert any('mAP50' in f['message'] for f in detail['failures'])
    assert any(f['code'] == 'map50_below_floor' for f in detail['failures'])
    assert detail['force_allowed'] is True
    assert detail['thresholds']['map50_min'] == 0.65


def test_promote_endpoint_job_not_ready_422_is_structured_and_force_disallowed(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every promote-blocking
    422 -- not just the score gate -- must carry {code, message} failures
    and say whether force=true can help. A job that isn't finished/exporting
    yet can't be force-promoted (there's no export to promote)."""
    from src.services.training.job_models import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='still-running-job',
        state='running',
        checkpoint_path=None,
        eval=None,
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'still-running-job' else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _fake_read_status)

    r = app_client.post(
        '/curation/projects/default/train/promote/still-running-job?wait=true',
        json={'triton_name': 'yolo26m_running', 'force': True},
    )
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['force_allowed'] is False
    assert detail['failures'][0]['code'] == 'job_not_promote_ready'
    assert 'running' in detail['failures'][0]['message']


def test_manifest_endpoint_returns_404_when_absent(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/train/manifest/no-such-job')
    assert r.status_code == 404


def test_manifest_endpoint_returns_payload_when_present(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """The labeler reads the manifest verbatim; we return whatever JSON is on disk."""
    job_id = '20260509-test-job'
    payload = {
        'kind': 'train',
        'job_id': job_id,
        'lineage': {'export_dir': str(project_export_root / 'x'), 'include_classes': [12, 81]},
        'results': {'eval': {'map50': 0.91}},
        'promoted_to': None,
    }
    import json

    (tmp_path / 'projects' / 'default' / f'{job_id}.manifest.json').write_text(json.dumps(payload))

    r = app_client.get(f'/curation/projects/default/train/manifest/{job_id}')
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['kind'] == 'train'
    assert out['lineage']['include_classes'] == [12, 81]


def test_promote_endpoint_force_bypasses_gate(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=true skips the gate and proceeds to the (mocked) handoff."""
    from pathlib import Path

    from src.services.training.job_models import TrainJobStatus

    fake_status = TrainJobStatus(
        job_id='force-job',
        state='finished',
        checkpoint_path='/jobs/force-job/best.pt',
        eval={'map50': 0.10, 'per_class': []},
    )

    async def _fake_read_status(job_id: str) -> TrainJobStatus | None:
        return fake_status if job_id == 'force-job' else None

    async def _fake_promote(**_kwargs: Any) -> Any:
        from src.services.training.triton_promote import PromoteResult

        return PromoteResult(
            job_id='force-job',
            triton_name='yolo26m_forced',
            onnx_path=str(Path('/triton-models/yolo26m_forced/1/model.onnx')),
            config_path=str(Path('/triton-models/yolo26m_forced/config.pbtxt')),
            labels_path=str(Path('/triton-models/yolo26m_forced/labels.txt')),
            triton_loaded=True,
        )

    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        _fake_read_status,
    )
    monkeypatch.setattr(
        'src.services.training.triton_promote.promote_yolo26_to_triton',
        _fake_promote,
    )

    r = app_client.post(
        '/curation/projects/default/train/promote/force-job?wait=true',
        json={'triton_name': 'yolo26m_forced', 'force': True},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['triton_name'] == 'yolo26m_forced'
    assert out['triton_loaded'] is True
    # Final E2E run 2026-09-26: first inference after promote took ~85s
    # (TensorRT JIT engine build on first request, not on /load). The
    # promote response must surface this so a caller doesn't mistake it
    # for a hang.
    assert out['cold_start_expected_on_first_inference'] is True


# =============================================================================
# Phase 9 — P1-9 gate strictness (non-numeric metrics must FAIL, not skip)
# =============================================================================


def test_gate_blocks_null_precision() -> None:
    """Before the fix: a null precision short-circuits `isinstance(...) and`
    to False, so the row is silently treated as passing."""
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': None, 'support': 100},
            ],
        }
    )
    assert any('pickup' in f.message and 'precision' in f.message for f in failures), failures


def test_gate_blocks_string_support() -> None:
    """Before the fix: a string support ("n/a") short-circuits the same way."""
    from src.services.training.promote_gate import evaluate_promote_gate

    failures = evaluate_promote_gate(
        {
            'map50': 0.80,
            'per_class': [
                {'class_id': 12, 'name': 'pickup', 'precision': 0.95, 'support': 'n/a'},
            ],
        }
    )
    assert any('pickup' in f.message and 'support' in f.message for f in failures), failures
