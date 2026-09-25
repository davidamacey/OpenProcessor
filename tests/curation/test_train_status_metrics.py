"""TrainJobStatus metric fields keep their meaning for every status.json.

``best_checkpoint_metric`` is the best checkpoint's validation-split score.
``eval`` may hold test-split numbers, so a status without the field must not
borrow them, and the retired ``best_metric`` / ``last_metric`` keys (a
per-key running max that could mix epochs) must not reach the wire.
"""

from __future__ import annotations

import pytest

from src.services.training.jobs import TrainJobStatus


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
async def test_trained_models_reports_the_eval_score_with_its_split(monkeypatch) -> None:
    from src.routers.curation import bakeoff
    from src.services.training import jobs

    run = TrainJobStatus.model_validate(
        {
            'job_id': 'run-3',
            'state': 'finished',
            'checkpoint_path': '/runs/run-3/weights/best.pt',
            'best_checkpoint_metric': {'epoch': 18, 'map50': 0.99, 'map50_95': 0.86},
            'eval': {'split': 'test', 'map50': 0.92, 'map50_95': 0.81},
        }
    )

    async def fake_list_runs(limit: int, offset: int) -> list[TrainJobStatus]:
        return [run]

    monkeypatch.setattr(jobs, 'list_runs', fake_list_runs)
    monkeypatch.setattr(bakeoff, '_checkpoint_exists', lambda _path: True)

    body = await bakeoff.bakeoff_trained_models()

    model = body['models'][0]
    assert model['map50'] == 0.92
    assert model['map50_split'] == 'test'
