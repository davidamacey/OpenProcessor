"""The budget follows what the endpoint reports: a call with usage settles at it, a call without
is charged what was reserved (fail closed) and counted ``usage_missing``; the prompt estimate
learns from settled calls."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import FakeVlm, World, drive, make_world
from src.db.repos import assist_loop_record as record_rows


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 300}, prefix='a')
    return w


def _tokens(run: object) -> tuple[int, int]:
    budget = run.budget.snapshot()['vlm']  # type: ignore[attr-defined]
    return budget['prompt_tokens_spent'], budget['completion_tokens_spent']


def test_a_call_with_usage_is_charged_what_the_endpoint_reported(
    world: World, loop_env: LoopEnv
) -> None:
    vlm = FakeVlm(world.truth, usage_per_image=(100, 5))

    run = asyncio.run(drive(world, loop_env, vlm, max_depth=0))

    asked = len(vlm.asked)
    assert _tokens(run) == (100 * asked, 5 * asked)
    assert run.budget.usage_missing == 0
    assert run.budget.snapshot()['vlm']['crops_spent'] == asked
    assert run.budget.snapshot()['vlm']['crops_reserved'] == 0


def test_a_call_without_usage_is_charged_its_reservation_and_counted(
    world: World, loop_env: LoopEnv
) -> None:
    vlm = FakeVlm(world.truth)

    run = asyncio.run(drive(world, loop_env, vlm, max_depth=0))

    prompt, completion = _tokens(run)
    assert completion >= 768 * len(vlm.asked), 'charged the max_tokens it could have used'
    assert prompt > 0
    assert run.budget.usage_missing > 0


def test_the_prompt_estimate_learns_from_settled_calls(world: World, loop_env: LoopEnv) -> None:
    vlm = FakeVlm(world.truth, usage_per_image=(40, 5))
    run = asyncio.run(drive(world, loop_env, vlm, max_depth=0, vlm_concurrency=1))
    import psycopg

    with psycopg.connect(loop_env.pg.conninfo('super'), autocommit=True) as conn:
        history = conn.execute(
            'SELECT p.amount::float8 / c.amount FROM assist_loop_budget_ledger p '
            'JOIN assist_loop_budget_ledger c ON c.run_id = p.run_id AND c.request_id = p.request_id '
            "AND c.unit = 'crops' AND c.entry = 'settle' WHERE p.run_id = %s "
            "AND p.unit = 'prompt_tokens' AND p.entry = 'settle'",
            (run.ref.run_id,),
        ).fetchall()
    assert history
    assert {round(h[0]) for h in history} == {40}
    assert record_rows  # the history read by the reservation is this ledger
