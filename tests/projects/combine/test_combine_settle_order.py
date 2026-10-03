"""A combine job must not read terminal while its target is still ``building``
(GH #47): the next step the job serves would 409 ``project_building``."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.projects.combine.store import COMPLETED_STATUSES

from .test_combine_execute import MAPPING, build
from .world import World, run_job


@pytest.mark.asyncio
async def test_the_target_is_settled_before_the_job_reads_completed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    order: list[tuple[str, Any]] = []
    from src.services.curation.file_job import FileJob

    original = FileJob.update

    def recording_update(self: FileJob, **fields: Any) -> Any:
        if fields.get('status') in COMPLETED_STATUSES:
            order.append(('job', fields['status']))
        return original(self, **fields)

    monkeypatch.setattr(FileJob, 'update', recording_update)
    real_settle_log = world.settled

    class Spy(list):  # records the settle call in the same ordered log
        def append(self, ok: bool) -> None:
            order.append(('settle', ok))
            real_settle_log.append(ok)

    world.settled = Spy()  # type: ignore[assignment]
    store, _target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    assert [e[0] for e in order] == ['settle', 'job']
    assert store.job.read()['status'] == 'completed'


@pytest.mark.asyncio
async def test_a_failed_settle_fails_the_job_instead_of_completing_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.projects.combine import service as svc

    build(world)
    store, target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    calls: list[bool] = []

    async def finish_building(_c: Any, _t: Any, *, ok: bool) -> None:
        calls.append(ok)
        raise RuntimeError('registry down')

    monkeypatch.setattr(svc.lifecycle, 'finish_building', finish_building)
    store.job.update(status='interrupted')
    await svc._run(world.fake, store, [world.records['cars-a'], world.records['cars-b']], target)
    state = store.job.read()
    assert state['status'] == 'failed'
    assert 'could not finish the target' in state['error']
    assert calls == [True]
