"""An assist run's VLM calls land in ``vlm_usage_hourly`` under ``assist_run``: the run asks the
labeler factory for that usage source, so ten crops sent to the model add ten crops."""

from __future__ import annotations

import json
import time
from typing import Any

import psycopg
import pytest

from src.services.curation.assist import runner


pytest_plugins = ['curation.assist_queue']
pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]

CLUSTER = 10000
CROPS = 10


@pytest.fixture(autouse=True)
def _queue(assist_queue: Any) -> None:
    """Runs go through the queue, with a stand-in worker (built before ``readable_crops``)."""


def _seed(stack: Any, crop_id: str) -> None:
    stack.seed(
        crop_id,
        cluster_id=CLUSTER,
        cluster_subid=f'{CLUSTER}a',
        class_source='item_model',
        updated_at='2026-10-01T00:00:00+00:00',
    )


def _wait(stack: Any, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = stack.client.get(f'/curation/projects/default/assist/runs/{run_id}').json()
        if state['status'] not in ('queued', 'running'):
            return state
        time.sleep(0.05)
    raise AssertionError('the run did not finish')


def test_ten_crops_of_an_assist_run_add_ten_crops_under_assist_run(
    stack: Any, job_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint = stack.usage_endpoint_id
    sources: list[Any] = []
    real = runner.get_vlm_labeler

    def spy(*args: Any, **kwargs: Any) -> Any:
        sources.append(kwargs.get('usage_source'))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, 'get_vlm_labeler', spy)
    for i in range(CROPS):
        _seed(stack, f'c{i}')
    stack.upstream.vlm_content = json.dumps(
        {'results': [{'img': i, 'class': 'gadget', 'confidence': 'high'} for i in range(1, 11)]}
    )

    with stack.client:
        started = stack.post(
            '/assist/runs',
            target={'kind': 'cluster', 'cluster_id': CLUSTER},
            actions=['classify'],
            dry_run=False,
        )
        assert started.status_code == 202, started.text
        state = _wait(stack, started.json()['run_id'])

    assert state['status'] == 'completed', state
    assert sources == ['assist_run']
    with psycopg.connect(job_env.conninfo('super')) as conn:
        rows = conn.execute(
            'SELECT source, sum(crops) FROM vlm_usage_hourly WHERE endpoint_config_id = %s '
            'GROUP BY source',
            (endpoint,),
        ).fetchall()
    assert rows == [('assist_run', CROPS)]
