"""The loop's budgets are write-ahead reservations: a reservation is in the event log and in
``budget.json`` before any request is sent, a crash turns every unsettled reservation into
spend, and no sequence of reservations, failures, retries and crashes ever exceeds a cap.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import CLASSES, FakeVlm, Killed, KillSwitch, drive, make_world
from src.services.curation.assist_loop.budget import (
    AUDIT,
    VLM,
    BudgetLimits,
    LoopBudget,
    PoolLimits,
)
from src.services.curation.assist_loop.partition import PartitionScope
from src.services.curation.assist_loop.store import LoopStore


if TYPE_CHECKING:
    from pathlib import Path


def _limits(crops: int = 10, requests: int = 100, wall: float = 3600.0) -> BudgetLimits:
    return BudgetLimits(
        pools={VLM: PoolLimits(crops, requests), AUDIT: PoolLimits(5, 5)}, max_wall_s=wall
    )


def _budget(store: LoopStore, limits: BudgetLimits | None = None, **kw: object) -> LoopBudget:
    return LoopBudget(store, limits or _limits(), **kw)  # type: ignore[arg-type]


def _ids(n: int, prefix: str = 'c') -> list[str]:
    return [f'{prefix}{i}' for i in range(n)]


# ---------------------------------------------------------------- limits


def test_the_crop_cap_is_required() -> None:
    with pytest.raises(ValueError, match='max_vlm_crops'):
        BudgetLimits.from_request(
            max_vlm_crops=None,
            max_vlm_requests=None,
            max_wall_s=None,
            images_per_call=4,
            vlm_concurrency=16,
            audit_max_crops=500,
        )
    with pytest.raises(ValueError, match='max_vlm_crops'):
        BudgetLimits.from_request(
            max_vlm_crops=0,
            max_vlm_requests=None,
            max_wall_s=None,
            images_per_call=4,
            vlm_concurrency=16,
            audit_max_crops=500,
        )


def test_request_and_time_defaults_follow_the_design() -> None:
    limits = BudgetLimits.from_request(
        max_vlm_crops=30000,
        max_vlm_requests=None,
        max_wall_s=None,
        images_per_call=4,
        vlm_concurrency=16,
        audit_max_crops=500,
    )

    assert limits.pools[VLM] == PoolLimits(max_crops=30000, max_requests=7500 + 16)
    assert limits.pools[AUDIT] == PoolLimits(max_crops=500, max_requests=125 + 16)
    assert limits.max_wall_s == 21600


# ---------------------------------------------------------------- reserve / settle


def test_a_reservation_is_persisted_before_it_is_returned(tmp_path: Path) -> None:
    store = LoopStore(tmp_path / 'loop')
    budget = _budget(store)

    res = budget.reserve(VLM, 'r1', _ids(4))

    assert res is not None
    (event,) = store.events()
    assert event['event'] == 'reserve'
    assert (event['rid'], event['crops'], event['requests'], event['ids']) == (
        res.rid,
        4,
        1,
        _ids(4),
    )
    on_disk = store.read_json('budget.json')
    assert on_disk[VLM]['crops_reserved'] == 4
    assert on_disk[VLM]['crops_spent'] == 0


def test_a_reservation_that_cannot_be_persisted_is_not_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LoopStore(tmp_path / 'loop')
    budget = _budget(store)

    def broken(*_a: object, **_k: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(store, 'append', broken)

    with pytest.raises(OSError, match='disk full'):
        budget.reserve(VLM, 'r1', _ids(4))
    assert budget.snapshot()[VLM]['crops_reserved'] == 0


def test_settling_moves_the_share_from_reserved_to_spent(tmp_path: Path) -> None:
    budget = _budget(LoopStore(tmp_path / 'loop'))
    res = budget.reserve(VLM, 'r1', _ids(3))
    assert res is not None

    budget.settle(res)

    snap = budget.snapshot()[VLM]
    assert (snap['crops_reserved'], snap['crops_spent']) == (0, 3)
    assert (snap['requests_reserved'], snap['requests_spent']) == (0, 1)


def test_a_reservation_over_the_crop_cap_is_refused(tmp_path: Path) -> None:
    budget = _budget(LoopStore(tmp_path / 'loop'), _limits(crops=10))
    first = budget.reserve(VLM, 'r1', _ids(8))
    assert first is not None

    assert budget.reserve(VLM, 'r1', _ids(3, 'd')) is None
    assert budget.refused == 'budget_crops'
    assert budget.reserve(VLM, 'r1', _ids(2, 'e')) is not None, 'exactly the cap is allowed'


def test_a_reservation_over_the_request_cap_is_refused(tmp_path: Path) -> None:
    budget = _budget(LoopStore(tmp_path / 'loop'), _limits(crops=100, requests=2))
    assert budget.reserve(VLM, 'r1', ['a']) is not None
    assert budget.reserve(VLM, 'r1', ['b']) is not None

    assert budget.reserve(VLM, 'r1', ['c']) is None
    assert budget.refused == 'budget_requests'


def test_no_reservation_once_the_wall_time_is_used(tmp_path: Path) -> None:
    now = [0.0]
    budget = _budget(LoopStore(tmp_path / 'loop'), _limits(wall=60), clock=lambda: now[0])
    assert budget.reserve(VLM, 'r1', ['a']) is not None

    now[0] = 61.0

    assert budget.reserve(VLM, 'r1', ['b']) is None
    assert budget.refused == 'budget_time'
    assert budget.time_exhausted()


def test_the_audit_pool_has_its_own_cap(tmp_path: Path) -> None:
    budget = _budget(LoopStore(tmp_path / 'loop'), _limits(crops=10))
    assert budget.reserve(VLM, 'r1', _ids(10)) is not None

    assert budget.reserve(AUDIT, 'r1', _ids(5, 'a')) is not None
    assert budget.reserve(AUDIT, 'r1', ['z']) is None
    snap = budget.snapshot()
    assert (snap[VLM]['crops_reserved'], snap[AUDIT]['crops_reserved']) == (10, 5)


def test_would_refuse_says_why_and_does_not_reserve(tmp_path: Path) -> None:
    budget = _budget(LoopStore(tmp_path / 'loop'), _limits(crops=10, requests=3))

    assert budget.would_refuse(VLM, crops=10, requests=3) is None
    assert budget.would_refuse(VLM, crops=11, requests=3) == 'budget_crops'
    assert budget.would_refuse(VLM, crops=1, requests=4) == 'budget_requests'
    assert budget.snapshot()[VLM]['crops_reserved'] == 0
    assert budget.refused is None


# ---------------------------------------------------------------- crash and resume


def test_a_crash_after_reserve_counts_the_reservation_as_spent(tmp_path: Path) -> None:
    store = LoopStore(tmp_path / 'loop')
    budget = _budget(store, _limits(crops=10))
    assert budget.reserve(VLM, 'r1', _ids(6)) is not None
    # the process dies here: no answers, no settle

    resumed = _budget(LoopStore(tmp_path / 'loop'), _limits(crops=10))

    snap = resumed.snapshot()[VLM]
    assert (snap['crops_spent'], snap['crops_reserved']) == (6, 0)
    assert resumed.reserve(VLM, 'r1', _ids(5, 'd')) is None
    assert resumed.reserve(VLM, 'r1', _ids(4, 'd')) is not None


def test_reservation_ids_keep_increasing_across_a_resume(tmp_path: Path) -> None:
    store = LoopStore(tmp_path / 'loop')
    first = _budget(store).reserve(VLM, 'r1', ['a'])
    second = _budget(LoopStore(tmp_path / 'loop')).reserve(VLM, 'r1', ['b'])

    assert first is not None
    assert second is not None
    assert second.rid > first.rid


def test_wall_time_carries_over_a_resume(tmp_path: Path) -> None:
    now = [0.0]
    store = LoopStore(tmp_path / 'loop')
    budget = _budget(store, _limits(wall=100), clock=lambda: now[0])
    now[0] = 70.0
    assert budget.reserve(VLM, 'r1', ['a']) is not None

    later = [1000.0]
    resumed = _budget(LoopStore(tmp_path / 'loop'), _limits(wall=100), clock=lambda: later[0])
    assert resumed.wall_s() == pytest.approx(70.0)
    later[0] = 1031.0

    assert resumed.reserve(VLM, 'r1', ['b']) is None
    assert resumed.refused == 'budget_time'


# ---------------------------------------------------------------- randomized


def _replayed_totals(store: LoopStore, pool: str) -> list[tuple[int, int]]:
    """(crops, requests) committed after each event: every reservation in the log counts,
    whether settled or not, which is what a crash at that event would leave."""
    crops = requests = 0
    out = []
    for event in store.events():
        if event['event'] == 'reserve' and event['pool'] == pool:
            crops += event['crops']
            requests += event['requests']
        out.append((crops, requests))
    return out


async def _workload(store_dir: Path, seed: int) -> tuple[LoopStore, BudgetLimits]:
    rng = random.Random(seed)
    limits = _limits(crops=rng.randint(5, 60), requests=rng.randint(3, 30))
    store = LoopStore(store_dir)
    budget = _budget(store, limits)
    crashes = rng.randint(0, 2)

    async def worker(name: str) -> None:
        nonlocal budget
        for step in range(rng.randint(2, 8)):
            ids = _ids(rng.randint(1, 4), f'{name}-{step}-')
            for _attempt in range(2):  # one retry of a failed request
                res = budget.reserve(VLM, 'r1', ids)
                if res is None:
                    return
                snap = budget.snapshot()[VLM]
                assert snap['crops_spent'] + snap['crops_reserved'] <= limits.pools[VLM].max_crops
                assert (
                    snap['requests_spent'] + snap['requests_reserved']
                    <= limits.pools[VLM].max_requests
                )
                await asyncio.sleep(0)
                failed = rng.random() < 0.3
                store.append('answers', path='r1', rid=res.rid, answers={}, failed=ids)
                budget.settle(res)
                if not failed:
                    break

    tasks = [asyncio.create_task(worker(f'w{i}')) for i in range(rng.randint(1, 4))]
    for _ in range(crashes):
        await asyncio.sleep(0)
        # a crash: the in-memory budget is lost and rebuilt from the job directory
        budget = _budget(LoopStore(store_dir), limits)
    await asyncio.gather(*tasks)
    return store, limits


@pytest.mark.parametrize('seed', range(200))
def test_no_sequence_of_reservations_failures_retries_and_crashes_exceeds_a_cap(
    tmp_path: Path, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A crash is simulated by rebuilding from the files; durability is not what this checks.
    monkeypatch.setattr(os, 'fsync', lambda _fd: None)
    store, limits = asyncio.run(_workload(tmp_path / 'loop', seed))

    for crops, requests in _replayed_totals(store, VLM):
        assert crops <= limits.pools[VLM].max_crops
        assert requests <= limits.pools[VLM].max_requests


# ---------------------------------------------------------------- the whole loop


def _loop_case(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        'roots': [
            {c: rng.randint(0, 40) for c in rng.sample(CLASSES, rng.randint(1, 3))}
            for _ in range(rng.randint(1, 3))
        ],
        'fail_rate': rng.choice([0.0, 0.1, 0.3]),
        'accuracy': rng.choice([1.0, 0.95, 0.8]),
        'max_crops': rng.randint(4, 150),
        'max_requests': rng.choice([None, rng.randint(2, 40)]),
        'images_per_call': rng.choice([1, 2, 4]),
        'node_concurrency': rng.randint(1, 4),
        'vlm_concurrency': rng.randint(1, 4),
        'kill_at': rng.choice([None, rng.randint(1, 40)]),
    }


@pytest.mark.usefixtures('readable_crops')
@pytest.mark.parametrize('seed', range(200))
def test_a_loop_never_exceeds_its_caps_through_failures_retries_and_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    case = _loop_case(seed)
    world = make_world(tmp_path, monkeypatch)
    for i, mix in enumerate(case['roots']):
        world.add_cluster(10001 + i, mix, prefix=f'r{i}')
    vlm = FakeVlm(world.truth, accuracy=case['accuracy'], fail_rate=case['fail_rate'], seed=seed)
    options = {
        'max_crops': case['max_crops'],
        'max_requests': case['max_requests'],
        'images_per_call': case['images_per_call'],
        'node_concurrency': case['node_concurrency'],
        'vlm_concurrency': case['vlm_concurrency'],
        'min_node_size': 10,
        'scope': PartitionScope(cluster_ids=None, min_root_size=10),
    }
    loop_dir = tmp_path / 'loop'
    if case['kill_at'] is not None:
        store = LoopStore(loop_dir)
        switch = KillSwitch('reserve', case['kill_at'])
        switch.arm(store, world.os)
        with contextlib.suppress(Killed):
            asyncio.run(drive(world, loop_dir, vlm, store=store, **options))
        switch.disarm()

    run = asyncio.run(drive(world, loop_dir, vlm, **options))

    assert run.outcome.status == 'completed'
    caps = run.budget.snapshot()['vlm']
    for crops, requests in _replayed_totals(run.store, VLM):
        assert crops <= caps['max_crops']
        assert requests <= caps['max_requests']
    assert len(vlm.asked) <= caps['max_crops']
