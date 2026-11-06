"""A combine job must not read terminal while its target is still ``building``
(GH #47): the next step the job serves would 409 ``project_building``."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.projects.combine import execute

from .test_combine_execute import MAPPING, build
from .world import World, run_job


@pytest.mark.asyncio
async def test_the_target_is_settled_before_the_result_is_returned(world: World) -> None:
    build(world)
    order: list[str] = []
    real_settle_log = world.settled

    class Spy(list):  # records the settle call in the same ordered log
        def append(self, ok: bool) -> None:
            order.append(f'settle:{ok}')
            real_settle_log.append(ok)

    world.settled = Spy()  # type: ignore[assignment]
    run, _target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    order.append('returned')
    assert order == ['settle:True', 'returned']
    assert run.status == 'completed'


@pytest.mark.asyncio
async def test_a_failed_settle_fails_the_job_instead_of_completing_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    calls: list[bool] = []

    async def failing_settle(ok: bool) -> None:
        calls.append(ok)
        raise RuntimeError('registry down')

    real = execute.run_combine

    async def with_failing_settle(ctx: Any, client: Any, **kwargs: Any) -> Any:
        return await real(ctx, client, **{**kwargs, 'settle': failing_settle})

    from src.services.projects.combine import execute as module

    monkeypatch.setattr(module, 'run_combine', with_failing_settle)
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    assert run.status == 'failed'
    assert 'could not finish the target' in str(run.error)
    assert calls == [True]  # the target was not settled a second time as failed
