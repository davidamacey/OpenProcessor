"""The auto-label handler: one run through the existing coroutine, progress flushed, cancel and pause."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.curation.autolabel import vlm_stage
from src.services.curation.autolabel.progress import PauseRequestedError
from src.services.curation.autolabel.run_state import Run
from src.tasks import trigger
from src.tasks.runtime.context import Paused


class _Ctx:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.job_id = 'j'
        self.kind = 'autolabel'
        self.project_slug = 'p'
        self.snapshots: list[tuple[Any, ...]] = []
        self.cancel = False
        self.pause = False

    def cancelled(self) -> bool:
        return self.cancel

    def paused(self) -> bool:
        return self.pause

    async def progress(self, done, total=None, stage=None, _counters=None) -> None:
        self.snapshots.append((done, total, stage))

    async def client(self, name: str) -> str:
        return f'client:{name}'

    def session(self):
        raise AssertionError('no usage is written in these tests')


@pytest.fixture(autouse=True)
def _bound(monkeypatch: pytest.MonkeyPatch) -> None:
    async def record(_ctx: object) -> str:
        return 'record'

    from contextlib import contextmanager

    @contextmanager
    def bind(_ctx: object, _record: object):
        yield

    monkeypatch.setattr(trigger, 'project_record', record)
    monkeypatch.setattr(trigger, 'bound_job_project', bind)


@pytest.mark.asyncio
async def test_the_run_is_called_once_with_the_params_the_client_and_a_progress(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def run(**kwargs: Any) -> dict[str, Any]:
        kwargs['progress'].start_stage('vlm', total=10)
        kwargs['progress'].update(4)
        calls.append(kwargs)
        await asyncio.sleep(0)
        return {'stages': {}}

    monkeypatch.setattr(trigger, 'run_auto_label', run)
    ctx = _Ctx({'run_vlm': True})
    assert await trigger.autolabel(ctx) == {'stages': {}}  # type: ignore[arg-type]
    assert len(calls) == 1
    assert calls[0]['opensearch'] == 'client:opensearch'
    assert calls[0]['run_vlm'] is True
    assert ctx.snapshots[-1] == (4, 10, 'vlm')


@pytest.mark.asyncio
async def test_a_failed_stage_fails_the_job(monkeypatch) -> None:
    async def run(**_: Any) -> dict[str, Any]:
        return {'stages': {'vlm': {'status': 'error'}}}

    monkeypatch.setattr(trigger, 'run_auto_label', run)
    with pytest.raises(RuntimeError, match='vlm'):
        await trigger.autolabel(_Ctx())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_pause_at_a_boundary_returns_paused(monkeypatch) -> None:
    async def run(**_: Any) -> dict[str, Any]:
        raise PauseRequestedError

    monkeypatch.setattr(trigger, 'run_auto_label', run)
    assert isinstance(await trigger.autolabel(_Ctx()), Paused)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_cancel_at_a_boundary_ends_quietly_but_a_stray_cancellation_propagates(
    monkeypatch,
) -> None:
    async def run(**_: Any) -> dict[str, Any]:
        raise asyncio.CancelledError

    monkeypatch.setattr(trigger, 'run_auto_label', run)
    cancelled = _Ctx()
    cancelled.cancel = True
    assert await trigger.autolabel(cancelled) == {}  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await trigger.autolabel(_Ctx())  # type: ignore[arg-type]


class _Progress:
    def __init__(self) -> None:
        self.cancelled = False
        self.paused = False
        self.advanced = 0

    def start_stage(self, *_a: Any, **_k: Any) -> None: ...

    def advance(self, n: int) -> None:
        self.advanced += n


def _run(progress: _Progress) -> Run:
    return Run(
        opensearch=None, progress=progress, class_id=None, cluster_uid=None, item_filter=None,
        scope_filter=None,  # type: ignore[arg-type]
    )  # fmt: skip


@pytest.mark.asyncio
async def test_the_vlm_chunks_run_under_a_bound_and_all_of_them_run(monkeypatch) -> None:
    live = peak = 0
    seen: list[list[str]] = []

    async def chunk(_run: Run, _ctx: Any, ids: list[str], _meta: Any) -> Any:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.001)
        seen.append(ids)
        live -= 1
        return vlm_stage.ChunkResult(len(ids), len(ids), [])

    monkeypatch.setattr(vlm_stage, '_label_chunk', chunk)
    progress = _Progress()
    ids = [str(i) for i in range(200)]
    out = await vlm_stage._label_all(
        _run(progress),
        None,
        ids,
        batch_size=4,
        concurrency=3,
        meta=None,  # type: ignore[arg-type]
    )
    assert peak == 3
    assert sum(len(c) for c in seen) == 200
    assert out.predicted == 200
    assert progress.advanced == 200


@pytest.mark.asyncio
async def test_a_pause_stops_taking_chunks_and_raises_after_the_ones_in_flight(monkeypatch) -> None:
    progress = _Progress()
    done: list[int] = []

    async def chunk(_run: Run, _ctx: Any, ids: list[str], _meta: Any) -> Any:
        done.append(len(ids))
        progress.paused = len(done) >= 2
        return vlm_stage.ChunkResult(len(ids), 0, [])

    monkeypatch.setattr(vlm_stage, '_label_chunk', chunk)
    with pytest.raises(PauseRequestedError):
        await vlm_stage._label_all(
            _run(progress), None, [str(i) for i in range(40)], batch_size=2, concurrency=1,  # type: ignore[arg-type]
            meta=None,
        )  # fmt: skip
    assert len(done) == 2
