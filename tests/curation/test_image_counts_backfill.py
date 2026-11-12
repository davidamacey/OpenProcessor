"""The ``image_counts_backfill`` job: every image gets its counts, resumably and idempotently."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.curation import image_counts_backfill as job
from src.tasks.runtime.context import JobResult, Paused, Yield


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)


class _Ctx:
    def __init__(
        self, client: Any, resume_from: dict[str, Any] | None = None, *, pause_after: int = 0
    ) -> None:
        self.resume_from = resume_from
        self.checkpoints: list[dict[str, Any]] = []
        self.progress_calls: list[tuple[Any, ...]] = []
        self._client = client
        self._pause_after = pause_after

    def cancelled(self) -> bool:
        return False

    def paused(self) -> bool:
        return bool(self._pause_after) and len(self.checkpoints) >= self._pause_after

    def slice_due(self) -> bool:
        return False

    async def client(self, _name: str) -> Any:
        return self._client

    async def checkpoint(self, cursor: Any) -> None:
        self.checkpoints.append(dict(cursor))

    async def progress(self, *args: Any, **_kw: Any) -> None:
        self.progress_calls.append(args)


@pytest.fixture(autouse=True)
def _bound(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def bound(_ctx: Any) -> AsyncIterator[str]:
        yield 'p'

    monkeypatch.setattr(job, 'bound_job_project', bound)
    monkeypatch.setattr(job, 'BATCH', 2)


def _world(n: int = 5) -> QueryFakeOpenSearch:
    images = {f'img-{i}': {'image_id': f'img-{i}'} for i in range(n)}
    items = {
        f'c{i}': {'crop_id': f'c{i}', 'image_id': f'img-{i}', 'class_id': 1} for i in range(0, n, 2)
    }
    return QueryFakeOpenSearch({IMAGES: images, ITEMS: items})


@pytest.mark.asyncio
async def test_every_image_gets_counts_in_pages_with_a_checkpoint_each() -> None:
    fake = _world()
    ctx = _Ctx(fake)
    out = await job.run(ctx)  # type: ignore[arg-type]
    assert isinstance(out, JobResult)
    assert out.result == {'counted': 5}
    docs = fake.docs(IMAGES)
    assert all('counts_at' in d for d in docs.values())
    assert docs['img-0']['counts']['items'] == 1
    assert docs['img-1']['counts']['items'] == 0
    assert [c['after'] for c in ctx.checkpoints] == ['img-1', 'img-3', 'img-4']
    assert ctx.progress_calls[-1][:2] == (5, 5)


@pytest.mark.asyncio
async def test_a_paused_job_resumes_after_its_checkpoint_and_loses_no_count() -> None:
    fake = _world()
    first = _Ctx(fake, pause_after=1)
    out = await job.run(first)  # type: ignore[arg-type]
    assert isinstance(out, Paused)
    done = [i for i, d in fake.docs(IMAGES).items() if 'counts_at' in d]
    assert sorted(done) == ['img-0', 'img-1']

    resumed = await job.run(_Ctx(fake, first.checkpoints[-1]))  # type: ignore[arg-type]
    assert isinstance(resumed, JobResult)
    assert resumed.result == {'counted': 5}
    assert all('counts_at' in d for d in fake.docs(IMAGES).values())


@pytest.mark.asyncio
async def test_a_rerun_recomputes_the_same_counts() -> None:
    fake = _world()
    await job.run(_Ctx(fake))  # type: ignore[arg-type]
    first = {i: dict(d['counts']) for i, d in fake.docs(IMAGES).items()}
    await job.run(_Ctx(fake))  # type: ignore[arg-type]
    assert {i: dict(d['counts']) for i, d in fake.docs(IMAGES).items()} == first


@pytest.mark.asyncio
async def test_a_slice_boundary_yields_without_losing_the_cursor() -> None:
    fake = _world()

    class Sliced(_Ctx):
        def slice_due(self) -> bool:
            return len(self.checkpoints) >= 1

    ctx = Sliced(fake)
    assert isinstance(await job.run(ctx), Yield)  # type: ignore[arg-type]
    assert ctx.checkpoints == [{'after': 'img-1', 'counted': 2}]
