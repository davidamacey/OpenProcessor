"""Fixtures of the job tests: test kinds, a recording fake producer, database environment.

Imported by name into ``tests/jobs/conftest.py`` and the task tests that need them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from src.clients import task_queue
from src.config.job_kinds import KIND_BY_TASK, KINDS
from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES
from src.db.engine import dispose_engines


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

HOUR = 3600
_BASE: dict[str, Any] = {
    'queue': TaskQueues.JOBS,
    'soft_limit_s': HOUR,
    'hard_limit_s': HOUR + 900,
    'max_queue_wait_s': HOUR,
}
TEST_KINDS = (
    JobKind(name='t_plain', task_name='op.test.plain', project_scope='project', **_BASE),
    JobKind(
        name='t_excl',
        task_name='op.test.excl',
        project_scope='project',
        exclusive_group='excl',
        exclusive_scope='target',
        auto_resume=True,
        **_BASE,
    ),
    JobKind(
        name='t_pause',
        task_name='op.test.pause',
        project_scope='project',
        exclusive_group='pausing',
        pausable=True,
        slice_s=600,
        **{**_BASE, 'soft_limit_s': 2 * HOUR, 'hard_limit_s': 2 * HOUR + 900},
    ),
    JobKind(
        name='t_slice',
        task_name='op.test.slice',
        project_scope='project',
        queue=TaskQueues.JOBS,
        soft_limit_s=400,
        hard_limit_s=500,
        max_queue_wait_s=HOUR,
        exclusive_group='slicing',
        pausable=True,
        slice_s=1,
    ),
    JobKind(
        name='t_short',
        task_name='op.test.short',
        project_scope='project',
        queue=TaskQueues.JOBS,
        soft_limit_s=4,
        hard_limit_s=10,
        max_queue_wait_s=HOUR,
    ),
    JobKind(
        name='t_dep',
        task_name='op.test.dep',
        project_scope='deployment',
        exclusive_group='dep',
        exclusive_scope='slug',
        **_BASE,
    ),
)


@pytest.fixture
def kinds(monkeypatch: pytest.MonkeyPatch) -> dict[str, JobKind]:
    """Register the test kinds and their routes for the length of one test."""
    for kind in TEST_KINDS:
        monkeypatch.setitem(KINDS, kind.name, kind)
        monkeypatch.setitem(KIND_BY_TASK, kind.task_name, kind)
        monkeypatch.setitem(ROUTES, kind.task_name, kind.queue)
    return {kind.name: kind for kind in TEST_KINDS}


@dataclass
class FakeProducer:
    """Records every broker call in order; ``control`` is a trap: no revoke, no control command."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    fail_send: bool = False
    on_send: Any = None

    def send_task(self, name: str, *, args: list[str], task_id: str) -> None:
        if self.fail_send:
            raise task_queue.KombuError('broker is down')
        if self.on_send is not None:
            self.on_send(name, task_id)
        assert args == [task_id]
        self.calls.append(('send', task_id))

    @property
    def control(self) -> Any:
        raise AssertionError('the platform never uses a Celery control command or revoke')

    def sent(self) -> list[str]:
        return [job for what, job in self.calls if what == 'send']


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch, kinds: dict[str, JobKind]) -> FakeProducer:
    fake = FakeProducer()
    monkeypatch.setattr(task_queue, 'producer_app', lambda: fake)

    def mark_queued(job_id: str, *, ttl_s: int) -> None:
        assert ttl_s > 0
        fake.calls.append(('mark', job_id))

    monkeypatch.setattr(task_queue, 'mark_queued', mark_queued)
    monkeypatch.setattr(
        task_queue, 'clear_queued', lambda job_id: fake.calls.append(('clear', job_id))
    )
    return fake


@pytest.fixture
def wakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """The Valkey wake-ups the service and the runtime send, recorded instead of sent."""
    recorded: list[Any] = []
    monkeypatch.setattr('src.services.jobs.service.wake', recorded.append)
    for module in ('job_task', 'context'):
        monkeypatch.setattr(f'src.tasks.runtime.{module}.wake', recorded.append)
    return recorded


@pytest_asyncio.fixture
async def job_env(pg_env) -> AsyncIterator[Any]:
    yield pg_env
    await dispose_engines()


@dataclass
class FakeQueueHooks:
    """The runtime's producer seam, recorded; ``fail`` makes the send raise ``HookError``."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    fail: bool = False

    def enqueue(self, task_name: str, *, job_id: str) -> None:
        from src.tasks.runtime.hooks import HookError

        if self.fail:
            raise HookError('broker is down')
        self.calls.append(('send', f'{task_name}:{job_id}'))

    def mark_queued(self, job_id: str, *, ttl_s: int) -> None:
        assert ttl_s > 0
        self.calls.append(('mark', job_id))

    def clear_queued(self, job_id: str) -> None:
        self.calls.append(('clear', job_id))


@pytest.fixture(autouse=True)
def queue_hooks() -> Iterator[FakeQueueHooks]:
    """No test of the job runtime may reach a real broker: the seam is a recording fake."""
    from src.tasks.runtime import hooks

    previous = hooks.queue_hooks()
    fake = FakeQueueHooks()
    hooks.install_queue_hooks(fake)
    yield fake
    hooks.install_queue_hooks(previous)
