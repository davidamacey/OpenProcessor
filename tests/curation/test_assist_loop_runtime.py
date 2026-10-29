"""The worker's side of a loop: read the stored request and the pins, build the VLM asker, the
budget and the audit from them, and run the loop in its job directory. Fakes stand in for the
endpoint registry and the VLM; the loop, the budget and the job files are the real ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, make_world
from src.schemas.curation.assist_loop import AssistLoopRequest
from src.services.curation.assist_loop import job, runtime


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

PINNED = {
    'vlm': {'name': 'primary', 'revision': 3, 'model': 'fake-vlm'},
    'prompt_pack': {'name': 'generic_item', 'revision': None},
    'audit_vlm': None,
}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(tmp_path / 'loops'))
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(10001, {'car': 300}, prefix='a')
    return w


class Registry:
    """Stands in for the endpoint registry: resolves a pinned (name, revision) to an endpoint
    whose labeler is a fake VLM, and records what was asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, vlms: dict[str, FakeVlm]) -> None:
        self.vlms = vlms
        self.resolved: list[tuple[str | None, int | None]] = []
        self.connected: list[str] = []

        async def job_endpoint(_os: Any, **kw: Any) -> Any:
            assert kw['resolved'] is True, 'a worker resolves exactly what start pinned'
            self.resolved.append((kw['pinned_name'], kw['pinned_revision']))
            return SimpleNamespace(name=kw['pinned_name'], body=None)

        async def assert_may_connect(endpoint: Any) -> None:
            self.connected.append(endpoint.name)

        monkeypatch.setattr(runtime, 'job_endpoint', job_endpoint)
        monkeypatch.setattr(runtime, 'assert_may_connect', assert_may_connect)
        monkeypatch.setattr(
            runtime, 'get_vlm_labeler', lambda _p, _r, *, endpoint: self.vlms[endpoint.name]
        )


def _queue(world: World, *, pinned: dict[str, Any] | None = None, **request: Any) -> str:
    body: dict[str, Any] = {'budget': {'max_vlm_crops': 5000}, 'dry_run': False, **request}
    parsed = AssistLoopRequest.model_validate(body)
    loop = job.create_loop(
        state={
            'status': 'queued',
            'created_at': '2026-10-28T00:00:00+00:00',
            'pinned': pinned or PINNED,
        },
        request=parsed.model_dump(mode='json'),
    )
    return loop.directory.name


def _run(world: World, loop_id: str) -> Any:
    import asyncio

    return asyncio.run(runtime.run_job(world.os, loop_id))  # type: ignore[arg-type]


def _sources(world: World) -> dict[str, int]:
    out: dict[str, int] = {}
    for doc in world.os.docs(ITEMS).values():
        out[doc['class_source']] = out.get(doc['class_source'], 0) + 1
    return out


def test_a_queued_loop_runs_to_completion_with_the_pinned_endpoint(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm = FakeVlm(world.truth)
    registry = Registry(monkeypatch, {'primary': vlm})
    loop_id = _queue(world)

    outcome = _run(world, loop_id)

    state = job.read_state(loop_id)
    assert state is not None
    assert (outcome.status, state['status']) == ('completed', 'completed')
    assert registry.resolved == [('primary', 3)]
    assert registry.connected == ['primary']
    assert _sources(world)['assist_propagated'] > 0
    assert state['audit'] == {'status': 'off'}
    assert state['budget']['vlm']['max_crops'] == 5000


def test_the_stored_request_sets_the_parameters_of_the_loop(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm = FakeVlm(world.truth)
    Registry(monkeypatch, {'primary': vlm})
    loop_id = _queue(
        world,
        budget={'max_vlm_crops': 12, 'max_vlm_requests': 50},
        recursion={'max_depth': 0},
        node_concurrency=1,
    )

    _run(world, loop_id)

    state = job.read_state(loop_id)
    assert state is not None
    assert state['stop_reason'] == 'budget_crops'
    assert state['budget']['vlm']['crops_spent'] <= 12
    assert state['budget']['vlm']['max_requests'] == 50
    assert len(vlm.asked) <= 12


def test_a_loop_cancelled_before_the_worker_took_it_asks_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm = FakeVlm(world.truth)
    Registry(monkeypatch, {'primary': vlm})
    loop_id = _queue(world)
    job.open_job(loop_id).request_cancel()

    outcome = _run(world, loop_id)

    assert outcome.status == 'cancelled'
    assert vlm.asked == []
    assert job.read_state(loop_id)['status'] == 'cancelled'  # type: ignore[index]


def test_a_pinned_audit_model_is_asked_to_judge_the_propagated_labels(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, judge = FakeVlm(world.truth), FakeVlm(world.truth, model='judge')
    registry = Registry(monkeypatch, {'primary': primary, 'second': judge})
    pinned = {**PINNED, 'audit_vlm': {'name': 'second', 'revision': 1}}
    loop_id = _queue(world, pinned=pinned, audit={'vlm': 'second', 'min_per_batch': 30})

    _run(world, loop_id)

    state = job.read_state(loop_id)
    assert state is not None
    assert ('second', 1) in registry.resolved
    assert state['audit']['sampled'] >= 30
    assert state['audit']['status'] == 'ok'
    assert judge.asked, 'the audit model was asked'


def test_a_loop_whose_request_is_unreadable_fails_with_the_reason(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    Registry(monkeypatch, {'primary': FakeVlm(world.truth)})
    loop_id = _queue(world)
    (job.open_job(loop_id).directory / 'request.json').write_text('{not json')

    outcome = _run(world, loop_id)

    state = job.read_state(loop_id)
    assert state is not None
    assert outcome.status == 'failed'
    assert state['status'] == 'failed'
    assert 'request.json' in state['error']
    assert state['finished_at']


def test_a_loop_that_cannot_reach_its_pinned_endpoint_fails_closed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError

    async def gone(_os: Any, **_kw: Any) -> Any:
        msg = 'the pinned VLM endpoint primary@3 no longer exists'
        raise VlmEndpointUnavailableError(msg)

    monkeypatch.setattr(runtime, 'job_endpoint', gone)
    loop_id = _queue(world)

    outcome = _run(world, loop_id)

    assert outcome.status == 'failed'
    assert 'no longer exists' in (outcome.error or '')
    assert _sources(world) == {'item_proposal': 300}
    assert job.open_job(loop_id).state_file.exists()


def test_the_recursion_and_gate_of_the_request_reach_the_driver(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from curation.assist_loop_fakes import decisions
    from src.services.curation.assist_loop.store import LoopStore

    world.add_cluster(10002, {'car': 60, 'truck': 60}, prefix='m')
    Registry(monkeypatch, {'primary': FakeVlm(world.truth)})
    loop_id = _queue(world, recursion={'max_depth': 0}, scope={'cluster_ids': [10002]})

    _run(world, loop_id)

    got = decisions(LoopStore(job.open_job(loop_id).directory))
    assert got == {'r10002': 'review_leaf'}, 'only the scoped cluster, and it was not split'


def test_a_loop_with_no_pinned_prompt_pack_fails_closed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    Registry(monkeypatch, {'primary': FakeVlm(world.truth)})
    loop_id = _queue(world, pinned={**PINNED, 'prompt_pack': {}})

    outcome = _run(world, loop_id)

    assert outcome.status == 'failed'
    assert 'prompt pack' in (outcome.error or '')


def test_a_loop_the_worker_has_is_running_with_a_heartbeat_before_anything_is_resolved(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that dies while setting a loop up must leave one a stale heartbeat can
    repair to ``interrupted``, not a ``queued`` loop nobody will ever claim."""
    seen: dict[str, Any] = {}

    async def look(_os: Any, **_kw: Any) -> Any:
        loop = job.open_job(loop_id)
        seen.update(status=loop.read()['status'], beat=loop.heartbeat_file.exists())
        msg = 'stop here'
        raise ValueError(msg)

    monkeypatch.setattr(runtime, 'job_endpoint', look)
    loop_id = _queue(world)

    _run(world, loop_id)

    assert seen == {'status': 'running', 'beat': True}
