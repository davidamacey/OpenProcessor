"""TrainJobStatus metric fields keep their meaning for every status.json.

``best_checkpoint_metric`` is the best checkpoint's validation-split score.
``eval`` may hold test-split numbers, so a status without the field must not
borrow them, and the retired ``best_metric`` / ``last_metric`` keys (a
per-key running max that could mix epochs) must not reach the wire.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.training.jobs import TrainJobStatus


if TYPE_CHECKING:
    from pathlib import Path


def _legacy_status() -> dict:
    return {
        'job_id': 'run-1',
        'state': 'finished',
        'best_metric': {'map50': 0.995, 'map50_95': 0.857},
        'last_metric': {'map50': 0.99, 'map50_95': 0.85},
        'eval': {'split': 'test', 'map50': 0.917, 'map50_95': 0.811},
    }


def test_test_split_eval_is_not_reported_as_best_checkpoint_metric() -> None:
    status = TrainJobStatus.model_validate(_legacy_status())

    assert status.best_checkpoint_metric is None
    assert status.eval is not None
    assert status.eval['map50'] == 0.917


def test_retired_metric_keys_are_dropped_from_the_wire() -> None:
    wire = TrainJobStatus.model_validate(_legacy_status()).model_dump()

    assert 'best_metric' not in wire
    assert 'last_metric' not in wire


def test_current_metric_fields_pass_through_unchanged() -> None:
    raw = {
        'job_id': 'run-2',
        'state': 'finished',
        'last_epoch_metric': {'epoch': 20, 'map50': 0.98, 'map50_95': 0.84},
        'best_checkpoint_metric': {'epoch': 18, 'map50': 0.99, 'map50_95': 0.86},
        'eval': {'split': 'test', 'map50': 0.92, 'map50_95': 0.81},
    }

    status = TrainJobStatus.model_validate(raw)

    assert status.best_checkpoint_metric == raw['best_checkpoint_metric']
    assert status.last_epoch_metric == raw['last_epoch_metric']


@pytest.mark.asyncio
async def test_trained_models_reports_the_eval_score_with_its_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``trainer_map50`` is ``eval.map50`` (labelled by ``eval.split``), never the
    best checkpoint's validation score."""
    from src.routers.curation import bakeoff
    from src.services.curation import bakeoff_jobs

    ckpt = tmp_path / 'runs' / 'run-3' / 'weights' / 'best.pt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'pt')
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    (jobs_dir / 'run-3.status.json').write_text(
        json.dumps(
            {
                'job_id': 'run-3',
                'state': 'finished',
                'checkpoint_path': '/runs/run-3/weights/best.pt',
                'best_checkpoint_metric': {'epoch': 18, 'map50': 0.99, 'map50_95': 0.86},
                'eval': {'split': 'test', 'map50': 0.92, 'map50_95': 0.81},
            }
        )
    )
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(jobs_dir))
    monkeypatch.setattr(bakeoff_jobs, 'RUNS_HOST_ROOT', tmp_path / 'runs')

    body = await bakeoff.bakeoff_trained_models()

    model = body.models[0]
    assert model.trainer_map50 == 0.92
    assert model.trainer_map50_split == 'test'
