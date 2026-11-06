"""Campaign policy: stop-when, auto-skip, auto-promote callbacks and auto-quantize."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import campaign
import job_status
import pytest

from curation.trainer_protocol._helpers import make_spec
from src.db.repos.training_runs import RunRow


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


class _Session:
    async def run_sync(self, fn: Any) -> Any:
        return fn(None)


class _Ctx:
    job_id = uuid.uuid4()

    @asynccontextmanager
    async def session(self):
        yield _Session()


def _run_row(status: str, map50: float | None = None) -> RunRow:
    manifest = None if map50 is None else {'results': {'eval': {'map50': map50}}}
    return RunRow(
        job_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        spec={},
        class_names=[],
        family='yolo26',
        base_model='yolo26n',
        artifact_prefix='projects/p/training/x',
        manifest=manifest,
        mlflow_run_id=None,
        gpu_uuids=[],
        status=status,
        params={},
        progress={},
        error_code=None,
        error_message=None,
        created_at=dt.datetime.now(dt.UTC),
        started_at=None,
        finished_at=None,
        heartbeat_at=None,
    )


@pytest.fixture
def siblings(monkeypatch: pytest.MonkeyPatch):
    """The campaign's other runs (the repo is faked; its SQL is tested against PostgreSQL)."""
    rows: list[RunRow] = []
    skipped: list[tuple[str, uuid.UUID]] = []

    class _Repo:
        @staticmethod
        def in_campaign(_s, _campaign_id):
            return rows

        @staticmethod
        def skip_queued(_s, _campaign_id, except_job_id, code, _message):
            skipped.append((code, except_job_id))
            return sum(1 for r in rows if r.status == 'queued')

    monkeypatch.setattr(campaign, 'runs_repo', _Repo)
    return SimpleNamespace(rows=rows, skipped=skipped)


def test_a_met_stop_when_skips_the_queued_siblings_with_the_skip_reason(
    export_dir: Path, siblings: Any
) -> None:
    siblings.rows.append(_run_row('queued'))
    spec = make_spec(
        campaign_id='camp-1',
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.5},
    )
    state = job_status.StatusState(job_id=spec.job_id, campaign_id='camp-1', state='finished')
    state.eval = {'map50': 0.8}

    asyncio.run(campaign.maybe_handle_campaign(_Ctx(), spec, state))

    assert siblings.skipped == [('auto_skip_threshold_met', _Ctx.job_id)]


def test_an_unmet_stop_when_skips_nothing(export_dir: Path, siblings: Any) -> None:
    siblings.rows.append(_run_row('queued'))
    spec = make_spec(
        campaign_id='camp-1',
        dataset_export_dir=str(export_dir),
        stop_when={'map50_at_least': 0.9},
    )
    state = job_status.StatusState(job_id=spec.job_id, campaign_id='camp-1', state='finished')
    state.eval = {'map50': 0.8}

    asyncio.run(campaign.maybe_handle_campaign(_Ctx(), spec, state))

    assert siblings.skipped == []


def test_the_last_run_promotes_the_best_finished_sibling(
    export_dir: Path, siblings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    best = _run_row('completed', 0.9)
    siblings.rows.extend([_run_row('completed', 0.6), best, _run_row('failed')])
    promoted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        campaign, '_post_promote', lambda _project, job_id, name: promoted.append((job_id, name))
    )
    spec = make_spec(
        campaign_id='camp-1',
        dataset_export_dir=str(export_dir),
        auto_promote_best=True,
        is_last_in_campaign=True,
    )
    state = job_status.StatusState(job_id=spec.job_id, campaign_id='camp-1', state='finished')
    state.eval = {'map50': 0.7}  # this run is not the best

    asyncio.run(campaign.maybe_handle_campaign(_Ctx(), spec, state))

    assert promoted == [(str(best.job_id), 'camp-1_best')]


def test_campaign_hook_is_a_no_op_for_a_standalone_run(export_dir: Path, siblings: Any) -> None:
    spec = make_spec(dataset_export_dir=str(export_dir))
    state = job_status.StatusState(job_id=spec.job_id, state='finished')
    state.eval = {'map50': 0.9}

    asyncio.run(campaign.maybe_handle_campaign(_Ctx(), spec, state))  # must not raise or write

    assert siblings.skipped == []


def test_auto_quantize_posts_the_bakeoff_run_request(
    export_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    spec = make_spec(
        dataset_export_dir=str(export_dir), auto_quantize_bakeoff=True, project='default'
    )
    state = job_status.StatusState(job_id=spec.job_id, state='finished')
    state.checkpoint_path = str(tmp_path / 'best.pt')

    campaign.write_quant_bakeoff_job(spec, state)

    assert calls == [
        (
            'http://api.test:8000/curation/projects/default/bakeoff/run',
            {
                'job_id': f'{spec.job_id}_quant',
                'datasets': [{'id': f'run:{spec.job_id}'}],
                'models': [{'source': 'run', 'run_id': spec.job_id}],
                'quantize': {
                    'run_id': spec.job_id,
                    'formats': ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
                    'n_calib': 1000,
                    'throughput': True,
                },
            },
        )
    ]
    assert list(bakeoff_jobs.iterdir()) == []
    # The request body is accepted by the API's own request model.
    from src.schemas.bakeoff.run import BakeoffRunRequest

    BakeoffRunRequest.model_validate(calls[0][1])

    # A rejected POST is logged, never raised into the trainer's finally block.
    status_code = 409
    campaign.write_quant_bakeoff_job(spec, state)
    assert len(calls) == 2


def test_trainer_calls_back_under_the_project_that_started_the_run(
    export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored spec names its project; the campaign auto-promote goes to that project's
    prefix, and a run with no project posts nothing (there is no unscoped route to fall back
    to)."""

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
    spec = make_spec(dataset_export_dir=str(export_dir), project='default')
    assert spec.project == 'default'

    assert campaign._post_promote(spec.project, 'run-1', 'best')
    assert calls == ['http://api.test:8000/curation/projects/default/train/promote/run-1']
    assert not campaign._post_promote(None, 'run-1', 'best')
    assert len(calls) == 1
