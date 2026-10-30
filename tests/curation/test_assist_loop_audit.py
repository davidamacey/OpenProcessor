"""The second-opinion audit: a seeded sample of propagated items judged by a second model.
Its answers never write a label; agreement is reported with its Wilson lower bound; below
the floor the loop pauses (and with ``on_fail: rollback`` restores the batch). A remote
audit endpoint passes the existing external-images consent or is refused.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world
from src.core.exceptions import RequestRejectedError
from src.domain.loop_stats import wilson_lower_bound
from src.services.curation.assist_loop.audit import (
    AuditJudge,
    Auditor,
    AuditParams,
    resolve_audit_vlm,
)
from src.services.curation.assist_loop.budget import BudgetLimits, LoopBudget
from src.services.curation.assist_loop.store import LoopStore, replay
from src.services.curation.class_write_guard import ClassWriteGuard
from src.services.curation.vlm_class_batch import class_context


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

LOOP = 'alp_20261028T000000Z_000003'


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(tmp_path, monkeypatch)


def test_the_wilson_bounds_of_the_design() -> None:
    assert wilson_lower_bound(95, 100, 1.645) == pytest.approx(0.90, abs=0.005)
    assert wilson_lower_bound(90, 100, 1.645) == pytest.approx(0.84, abs=0.005)
    assert wilson_lower_bound(48, 50, 1.645) == pytest.approx(0.886, abs=0.005)


# ---------------------------------------------------------------- one batch


def _batch_world(world: World, tmp_path: Path) -> tuple[LoopStore, list[str]]:
    ids = world.add_cluster(10001, {'car': 100}, prefix='a')
    store = LoopStore(tmp_path / LOOP)
    store.append('node_open', path='r10001', parent=None, n=100, digest='d', level=0)
    store.append('decision', path='r10001', kind='pass', class_name='car', level=0, n=100)
    store.append('propagate_begin', path='r10001', class_name='car', n_eligible=100)
    store.append(
        'propagated', path='r10001', class_name='car', written=100, written_ids=ids, skipped={}
    )
    return store, ids


def _auditor(store: LoopStore, judge: FakeVlm, world: World, **params: Any) -> Auditor:
    limits = BudgetLimits.from_request(
        max_vlm_crops=10,
        max_vlm_requests=None,
        max_wall_s=None,
        images_per_call=4,
        vlm_concurrency=4,
        audit_max_crops=params.pop('audit_max_crops', 500),
    )
    options: dict[str, Any] = {'rate': 1.0, 'min_per_batch': 30, 'floor': 0.85}
    options.update(params)
    return Auditor(
        store,
        LoopBudget(store, limits),
        AuditJudge(
            world.os,  # type: ignore[arg-type]
            class_context(judge, ClassWriteGuard('assist_loop_audit'), revision=None),
            images_per_call=4,
            semaphore=asyncio.Semaphore(4),
        ),
        AuditParams(**options),
        seed=0,
    )


@pytest.mark.parametrize(('disagree', 'failing'), [(10, True), (5, False)])
def test_ninety_of_a_hundred_pauses_at_the_default_floor(
    world: World, tmp_path: Path, disagree: int, failing: bool
) -> None:
    store, ids = _batch_world(world, tmp_path)
    judge = FakeVlm(world.truth, answers=dict.fromkeys(ids[:disagree], 'truck'))
    auditor = _auditor(store, judge, world)
    before = world.snapshot()

    result = asyncio.run(auditor.run_batch(replay(store.events())))

    assert result.failing is failing
    assert result.nodes == ['r10001']
    (event,) = [e for e in store.events() if e['event'] == 'audit']
    assert (event['sampled'], event['agreed']) == (100, 100 - disagree)
    assert event['lower_bound'] == pytest.approx(
        wilson_lower_bound(100 - disagree, 100, 1.645), abs=1e-4
    )
    rows = (tmp_path / LOOP / 'audit.jsonl').read_text().splitlines()
    assert len(rows) == 100
    assert world.snapshot() == before, 'an audit answer never writes'


def test_fewer_than_the_minimum_audited_never_pauses(world: World, tmp_path: Path) -> None:
    store, ids = _batch_world(world, tmp_path)
    judge = FakeVlm(world.truth, answers=dict.fromkeys(ids, 'truck'))
    auditor = _auditor(store, judge, world, rate=0.1, min_per_batch=20)

    result = asyncio.run(auditor.run_batch(replay(store.events())))

    (event,) = [e for e in store.events() if e['event'] == 'audit']
    assert event['sampled'] == 20
    assert result.failing is False, 'under 30 audited answers the bound decides nothing'


def test_the_audit_never_exceeds_its_own_cap(world: World, tmp_path: Path) -> None:
    store, _ids = _batch_world(world, tmp_path)
    auditor = _auditor(store, FakeVlm(world.truth), world, audit_max_crops=10)

    asyncio.run(auditor.run_batch(replay(store.events())))

    (event,) = [e for e in store.events() if e['event'] == 'audit']
    assert event['sampled'] <= 10
    assert event['status'] == 'budget_exhausted'
    spent = sum(e['crops'] for e in store.events() if e['event'] == 'reserve')
    assert spent <= 10


def test_an_audited_batch_is_not_audited_again(world: World, tmp_path: Path) -> None:
    store, _ids = _batch_world(world, tmp_path)
    auditor = _auditor(store, FakeVlm(world.truth), world)
    asyncio.run(auditor.run_batch(replay(store.events())))

    assert not auditor.pending(replay(store.events()))


# ---------------------------------------------------------------- inside the loop


def _two_roots(world: World) -> None:
    world.add_cluster(10001, {'car': 600}, prefix='a')
    world.add_cluster(10002, {'truck': 400}, prefix='b')


def test_a_failing_audit_pauses_the_loop_and_keeps_the_labels(world: World, tmp_path: Path) -> None:
    _two_roots(world)
    judge = FakeVlm(world.truth, accuracy=0.5, confusion={'car': 'bus', 'truck': 'bus'}, seed=4)
    audit = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100)

    run = asyncio.run(
        drive(
            world,
            tmp_path / LOOP,
            FakeVlm(world.truth),
            audit=audit,
            judge=judge,
            node_concurrency=1,
        )
    )

    assert run.outcome.status == 'paused'
    assert run.outcome.pause_reason == 'audit'
    assert decisions(run.store) == {'r10001': 'pass'}, 'nothing is scheduled after the pause'
    docs = world.os.docs(ITEMS)
    assert sum(1 for d in docs.values() if d['class_source'] == 'assist_propagated') > 500
    assert not any(d['class_name'] == 'bus' for d in docs.values()), 'audit answers write nothing'
    state = run.store.job.read()
    assert state['pause_reason'] == 'audit'
    assert state['audit']['status'] == 'fail'
    assert state['audit']['lower_bound'] < 0.85


def test_on_fail_rollback_restores_the_batch(world: World, tmp_path: Path) -> None:
    _two_roots(world)
    before = world.snapshot()
    judge = FakeVlm(world.truth, accuracy=0.5, confusion={'car': 'bus', 'truck': 'bus'}, seed=4)
    audit = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100, on_fail='rollback')

    run = asyncio.run(
        drive(
            world,
            tmp_path / LOOP,
            FakeVlm(world.truth),
            audit=audit,
            judge=judge,
            node_concurrency=1,
        )
    )

    assert run.outcome.status == 'paused'
    docs = world.os.docs(ITEMS)
    assert not any(d['class_source'] == 'assist_propagated' for d in docs.values())
    for crop_id, doc in docs.items():
        if doc['class_source'] != 'vlm':
            assert doc['class_name'] == before[crop_id]['class_name']
            assert doc['cluster_id'] == before[crop_id]['cluster_id']
    (rollback,) = [e for e in run.store.events() if e['event'] == 'rollback']
    assert rollback['restored'] > 500


def test_a_passing_audit_lets_the_loop_finish(world: World, tmp_path: Path) -> None:
    _two_roots(world)
    audit = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100)

    run = asyncio.run(
        drive(world, tmp_path / LOOP, FakeVlm(world.truth), audit=audit, judge=FakeVlm(world.truth))
    )

    assert run.outcome.status == 'completed'
    audits = [e for e in run.store.events() if e['event'] == 'audit']
    assert len(audits) == 2, 'one batch per root here, the last one at the end'
    state = run.store.job.read()['audit']
    assert state['status'] == 'ok'
    assert state['agreement'] == 1.0
    assert state['crops_spent'] == state['sampled']


# ---------------------------------------------------------------- consent


def test_an_external_audit_endpoint_needs_the_acknowledgement(vlm_api: Any) -> None:
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)

    with pytest.raises(RequestRejectedError) as refused:
        asyncio.run(
            resolve_audit_vlm(vlm_api.fake_os, 'ext', pack=None, acknowledge_external=False)
        )
    assert 'vlm_external_not_acknowledged' in str(refused.value.detail)

    pinned = asyncio.run(
        resolve_audit_vlm(vlm_api.fake_os, 'ext', pack=None, acknowledge_external=True)
    )
    assert pinned.name == 'ext'


def test_a_denied_external_policy_refuses_the_audit_endpoint(
    vlm_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', 'deny')

    with pytest.raises(RequestRejectedError) as refused:
        asyncio.run(resolve_audit_vlm(vlm_api.fake_os, 'ext', pack=None, acknowledge_external=True))
    assert 'vlm_external' in str(refused.value.detail)


def test_no_audit_endpoint_means_the_audit_is_off(vlm_api: Any) -> None:
    assert (
        asyncio.run(resolve_audit_vlm(vlm_api.fake_os, None, pack=None, acknowledge_external=False))
        is None
    )
