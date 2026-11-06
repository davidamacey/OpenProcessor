"""``op.infra.sleep_noop``: the live checks' long-enough job (progress each second, stops on cancel)."""

from __future__ import annotations

import pytest

from src.config.job_kinds import KINDS
from src.config.task_routes import ROUTES
from src.tasks import infra


class _Ctx:
    def __init__(self, seconds: object, cancel_after: int | None = None) -> None:
        self.params = {'seconds': seconds}
        self.progress_calls: list[tuple[int, int | None, str | None]] = []
        self._cancel_after = cancel_after

    def cancelled(self) -> bool:
        return self._cancel_after is not None and len(self.progress_calls) >= self._cancel_after

    async def progress(self, done: int, total: int | None = None, stage: str | None = None) -> None:
        self.progress_calls.append((done, total, stage))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(infra.asyncio, 'sleep', instant)


def test_the_kind_and_its_route_exist_and_outlast_the_longest_sleep() -> None:
    kind = KINDS['sleep_noop']
    assert kind.task_name == 'op.infra.sleep_noop'
    assert ROUTES[kind.task_name] == kind.queue
    assert kind.soft_limit_s > infra.MAX_SLEEP_S


@pytest.mark.asyncio
async def test_it_reports_progress_each_second_and_returns() -> None:
    ctx = _Ctx(3)
    result = await infra.sleep_noop(ctx)  # type: ignore[arg-type]
    assert ctx.progress_calls == [(1, 3, 'sleeping'), (2, 3, 'sleeping'), (3, 3, 'sleeping')]
    assert result == {'slept': 3, 'cancelled': False}


@pytest.mark.asyncio
async def test_it_stops_early_when_cancelled() -> None:
    ctx = _Ctx(100, cancel_after=2)
    result = await infra.sleep_noop(ctx)  # type: ignore[arg-type]
    assert len(ctx.progress_calls) == 2
    assert result == {'slept': 2, 'cancelled': True}


@pytest.mark.asyncio
@pytest.mark.parametrize('seconds', [0, -1, 601, '5', None, True, 1.5])
async def test_seconds_outside_one_to_six_hundred_is_refused(seconds: object) -> None:
    with pytest.raises(ValueError, match='seconds'):
        await infra.sleep_noop(_Ctx(seconds))  # type: ignore[arg-type]
