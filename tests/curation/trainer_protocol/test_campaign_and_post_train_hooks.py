"""Campaign policy: stop-when, auto-skip, auto-promote callbacks and auto-quantize."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import campaign
import job_spec
import job_status
import pytest

from curation.trainer_protocol._helpers import _write_job
from src.services.training import jobs as train_jobs


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ('stop_when', 'eval_block', 'expected'),
    [
        (None, {'map50': 0.9}, False),
        ({'map50_at_least': 0.7}, {'map50': 0.71}, True),
        ({'map50_at_least': 0.7}, {'map50': 0.69}, False),
        ({'map50_at_least': 0.7}, {}, False),
        ({'map50_95_at_least': 0.4}, {'map50_95': 0.5}, True),
        ({'map50_at_least': 0.1, 'map50_95_at_least': 0.9}, {'map50': 0.5, 'map50_95': 0.5}, False),
        # Unknown rule fails CLOSED -- never skip siblings on a policy we
        # don't understand.
        ({'f1_at_least': 0.1}, {'map50': 0.99}, False),
    ],
)
def test_stop_when_policy(
    stop_when: dict[str, float] | None, eval_block: dict[str, Any], expected: bool
) -> None:
    assert campaign.stop_when_satisfied(stop_when, eval_block) is expected


def test_campaign_auto_skip_marks_queued_siblings_skipped(jobs_dir: Path, export_dir: Path) -> None:
    campaign_id = 'camp-1'
    winner = _write_job(
        job_id=f'{campaign_id}_run00_probe',
        campaign_id=campaign_id,
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.5},
    )
    loser = _write_job(
        job_id=f'{campaign_id}_run01_medium',
        campaign_id=campaign_id,
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.5},
    )
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{winner}.job.json')
    state = job_status.StatusState(job_id=winner, campaign_id=campaign_id, state='finished')
    state.eval = {'map50': 0.8}

    campaign.maybe_handle_campaign(spec, state)

    sibling = asyncio.run(train_jobs.read_status(loser))
    assert sibling is not None
    assert sibling.state == 'skipped'
    assert sibling.error == 'auto_skip_threshold_met'


def test_campaign_hook_is_a_no_op_for_a_standalone_job(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    state = job_status.StatusState(job_id=job_id, state='finished')
    state.eval = {'map50': 0.9}

    campaign.maybe_handle_campaign(spec, state)  # must not raise or write


def test_auto_quantize_posts_the_bakeoff_run_request(
    jobs_dir: Path, export_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer hands a finished run to the bake-off through the API, not a job file.

    The API resolves the run (checkpoint, imgsz, class map, eval dataset) and
    claims the GPU; a job file dropped straight into the evaluator's queue
    would bypass both.
    """
    import requests

    bakeoff_jobs = tmp_path / 'bakeoff_jobs'
    bakeoff_jobs.mkdir()
    monkeypatch.setattr(campaign, 'BAKEOFF_JOBS_DIR', bakeoff_jobs, raising=False)
    monkeypatch.setattr(campaign, 'API_BASE_URL', 'http://api.test:8000')
    monkeypatch.setattr(campaign, 'API_PREFIX', '/curation')
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f'{self.status_code} error')

    status_code = 200

    def _post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        calls.append((url, json))
        return _Resp(status_code)

    monkeypatch.setattr(requests, 'post', _post)
    job_id = _write_job(dataset_export_dir=str(export_dir), auto_quantize_bakeoff=True)
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    state = job_status.StatusState(job_id=job_id, state='finished')
    state.checkpoint_path = str(tmp_path / 'best.pt')

    campaign.write_quant_bakeoff_job(spec, state)

    assert calls == [
        (
            'http://api.test:8000/curation/projects/default/bakeoff/run',
            {
                'job_id': f'{job_id}_quant',
                'datasets': [{'id': f'run:{job_id}'}],
                'models': [{'source': 'run', 'run_id': job_id}],
                'quantize': {
                    'run_id': job_id,
                    'formats': ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
                    'n_calib': 1000,
                    'throughput': True,
                },
            },
        )
    ]
    assert list(bakeoff_jobs.iterdir()) == []
    # The request body is accepted by the API's own request model.
    from src.routers.curation._bakeoff_models import BakeoffRunRequest

    BakeoffRunRequest.model_validate(calls[0][1])

    # A rejected POST is logged, never raised into the trainer's finally block.
    status_code = 409
    campaign.write_quant_bakeoff_job(spec, state)
    assert len(calls) == 2


def test_trainer_calls_back_under_the_project_that_wrote_the_job(
    jobs_dir: Path, export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``job.json`` names its project; the campaign auto-promote goes to
    that project's prefix, and a job with no project posts nothing (there
    is no unscoped route to fall back to)."""

    import requests

    monkeypatch.setattr(campaign, 'API_BASE_URL', 'http://api.test:8000')
    monkeypatch.setattr(campaign, 'API_PREFIX', '/curation')
    calls: list[str] = []

    class _Ok:
        def raise_for_status(self) -> None:
            return None

    def _post(url: str, **_kw: Any) -> _Ok:
        calls.append(url)
        return _Ok()

    monkeypatch.setattr(requests, 'post', _post)
    job_id = _write_job(dataset_export_dir=str(export_dir))
    raw = json.loads((jobs_dir / f'{job_id}.job.json').read_text())
    assert raw['project'] == 'default'
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    assert spec.project == 'default'

    assert campaign._post_promote(spec.project, job_id, 'best')
    assert calls == [f'http://api.test:8000/curation/projects/default/train/promote/{job_id}']
    assert not campaign._post_promote(None, job_id, 'best')
    assert len(calls) == 1
