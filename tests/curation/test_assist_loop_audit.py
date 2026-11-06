"""The second-opinion audit: a seeded sample of propagated items judged by a second model.
Its answers never write a label; agreement is reported with its Wilson lower bound; below
the floor the loop pauses (and with ``on_fail: rollback`` restores the batch). A remote
audit endpoint passes the existing external-images consent or is refused.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import (
    ITEMS,
    FakeVlm,
    World,
    counts_of,
    decisions,
    drive,
    make_world,
    reload,
)
from src.core.exceptions import RequestRejectedError
from src.domain.loop_stats import wilson_lower_bound
from src.services.curation.assist_loop import driver
from src.services.curation.assist_loop.audit import AuditParams, resolve_audit_vlm
from src.services.curation.assist_loop.vlm_failure import VlmFailedError


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(monkeypatch)


def test_the_wilson_bounds_of_the_design() -> None:
    assert wilson_lower_bound(95, 100, 1.645) == pytest.approx(0.90, abs=0.005)
    assert wilson_lower_bound(90, 100, 1.645) == pytest.approx(0.84, abs=0.005)
    assert wilson_lower_bound(48, 50, 1.645) == pytest.approx(0.886, abs=0.005)


# ---------------------------------------------------------------- inside the loop


def _two_roots(world: World) -> None:
    world.add_cluster(10001, {'car': 600}, prefix='a')
    world.add_cluster(10002, {'truck': 400}, prefix='b')


FAILING = {'accuracy': 0.5, 'confusion': {'car': 'bus', 'truck': 'bus'}, 'seed': 4}
AUDIT = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100)


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_a_failing_audit_pauses_the_loop_and_keeps_the_labels(
    world: World, loop_env: LoopEnv
) -> None:
    _two_roots(world)
    judge = FakeVlm(world.truth, **FAILING)

    run = asyncio.run(
        drive(world, loop_env, FakeVlm(world.truth), audit=AUDIT, judge=judge, node_concurrency=1)
    )

    assert run.outcome.status == 'paused'
    assert run.outcome.pause_reason == 'audit'
    assert decisions(run.record) == {'r10001': 'pass'}, 'nothing is scheduled after the pause'
    docs = world.os.docs(ITEMS)
    assert sum(1 for d in docs.values() if d['class_source'] == 'assist_propagated') > 500
    assert not any(d['class_name'] == 'bus' for d in docs.values()), 'audit answers write nothing'
    audit = run.record.tree.audit
    assert audit['status'] == 'fail'
    assert audit['lower_bound'] < 0.85
    assert audit['covered'] == ['r10001']
    fresh = asyncio.run(reload(loop_env, run.ref))
    assert fresh.tree.audit == audit, 'the audit survives a slice'
    assert counts_of(loop_env, run)['audit']['status'] == 'fail'


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_on_fail_rollback_undoes_the_batchs_operations(
    world: World, loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_roots(world)
    judge = FakeVlm(world.truth, **FAILING)
    audit = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100, on_fail='rollback')
    undone: list[dict[str, Any]] = []

    async def undo(_os: Any, **kwargs: Any) -> None:
        undone.append(kwargs)

    monkeypatch.setattr(driver, 'undo_loop', undo)

    run = asyncio.run(
        drive(world, loop_env, FakeVlm(world.truth), audit=audit, judge=judge, node_concurrency=1)
    )

    assert run.outcome.pause_reason == 'audit'
    (call,) = undone
    assert call['node_paths'] == ['r10001']
    assert call['dry_run'] is False
    assert call['loop_id'] == str(run.ref.run_id)


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_a_passing_audit_lets_the_loop_finish(world: World, loop_env: LoopEnv) -> None:
    _two_roots(world)

    run = asyncio.run(
        drive(world, loop_env, FakeVlm(world.truth), audit=AUDIT, judge=FakeVlm(world.truth))
    )

    assert run.outcome.status == 'completed'
    audit = run.record.tree.audit
    assert audit['batches'] == 2, 'one batch per root here, the last one at the end'
    assert audit['status'] == 'ok'
    assert audit['total_agreed'] == audit['total_sampled'] > 0
    spent = run.budget.snapshot()['audit']
    assert spent['crops_spent'] == audit['total_sampled']


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_the_audit_never_exceeds_its_own_cap(world: World, loop_env: LoopEnv) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')

    run = asyncio.run(
        drive(
            world,
            loop_env,
            FakeVlm(world.truth),
            audit=AuditParams(rate=1.0, min_per_batch=30, floor=0.85, batch_items=10),
            judge=FakeVlm(world.truth),
            audit_max_crops=10,
            max_depth=0,
        )
    )

    assert run.record.tree.audit['status'] == 'budget_exhausted'
    assert run.budget.snapshot()['audit']['crops_spent'] <= 10


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_an_audit_whose_every_call_fails_is_not_recorded_as_ok(
    world: World, loop_env: LoopEnv
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    ref = loop_env.new_run()

    with pytest.raises(VlmFailedError):
        asyncio.run(
            drive(
                world,
                loop_env,
                FakeVlm(world.truth),
                ref=ref,
                audit=AUDIT,
                judge=FakeVlm(world.truth, fail_rate=1.0),
                max_depth=0,
            )
        )

    record = asyncio.run(reload(loop_env, ref))
    assert record.tree.audit == {}, 'the batch stays uncovered'


# ---------------------------------------------------------------- consent


def test_an_external_audit_endpoint_needs_the_acknowledgement(vlm_api: Any) -> None:
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)

    with pytest.raises(RequestRejectedError) as refused:
        asyncio.run(resolve_audit_vlm('ext', pack=None, acknowledge_external=False))
    assert 'vlm_external_not_acknowledged' in str(refused.value.detail)

    pinned = asyncio.run(resolve_audit_vlm('ext', pack=None, acknowledge_external=True))
    assert pinned.name == 'ext'


def test_a_denied_external_policy_refuses_the_audit_endpoint(
    vlm_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', 'deny')

    with pytest.raises(RequestRejectedError) as refused:
        asyncio.run(resolve_audit_vlm('ext', pack=None, acknowledge_external=True))
    assert 'vlm_external' in str(refused.value.detail)


def test_no_audit_endpoint_means_the_audit_is_off(vlm_api: Any) -> None:
    assert asyncio.run(resolve_audit_vlm(None, pack=None, acknowledge_external=False)) is None
