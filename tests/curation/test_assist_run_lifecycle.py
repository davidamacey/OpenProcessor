"""Reading, listing and cancelling assist runs through the routes."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from curation.test_assist_run_classify import CLUSTER, RUNS, _cluster_body, _seed, _wait


pytestmark = pytest.mark.usefixtures('readable_crops')


@pytest.fixture(autouse=True)
def _env(stack: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry', lambda: stack.registry
    )
    monkeypatch.setenv('OP_ASSIST_JOBS_DIR', str(tmp_path / 'assist_jobs'))


def test_an_unknown_run_id_is_a_404_on_read_and_cancel(stack: Any) -> None:
    got = stack.client.get(f'/curation/projects/default{RUNS}/asr_nope')
    cancelled = stack.post(f'{RUNS}/asr_nope/cancel')
    assert got.status_code == 404
    assert cancelled.status_code == 404


def test_the_list_returns_the_runs_that_name_the_cluster_newest_first(stack: Any) -> None:
    _seed(stack, 'free1')
    stack.upstream.vlm_content = json.dumps(
        {'results': [{'img': 1, 'class': 'gadget', 'confidence': 'high'}]}
    )
    with stack.client:
        ids = []
        for _ in range(2):
            started = stack.post(RUNS, **_cluster_body())
            assert started.status_code == 200, started.text
            ids.append(started.json()['run_id'])
            _wait(stack, ids[-1])
            time.sleep(0.01)
        listed = stack.client.get(f'/curation/projects/default{RUNS}?cluster_id={CLUSTER}')
        other = stack.client.get(f'/curation/projects/default{RUNS}?cluster_id=77')
    assert [r['run_id'] for r in listed.json()['runs']] == ids[::-1]
    assert other.json()['runs'] == []


def test_cancelling_a_live_run_ends_it_cancelled(stack: Any) -> None:
    for cid in ('free1', 'free2', 'free3'):
        _seed(stack, cid)
    stack.upstream.vlm_content = json.dumps(
        {'results': [{'img': 1, 'class': 'gadget', 'confidence': 'high'}]}
    )
    gate: dict[str, Any] = {}

    async def hold() -> None:
        import asyncio

        gate['reached'] = True
        while not gate.get('release'):
            await asyncio.sleep(0.01)

    stack.upstream.vlm_delay = hold
    with stack.client:
        started = stack.post(RUNS, **_cluster_body())
        run_id = started.json()['run_id']
        deadline = time.monotonic() + 10
        while not gate.get('reached') and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled = stack.post(f'{RUNS}/{run_id}/cancel')
        gate['release'] = True
        final = _wait(stack, run_id)
    assert cancelled.status_code == 200
    assert final['status'] == 'cancelled'
