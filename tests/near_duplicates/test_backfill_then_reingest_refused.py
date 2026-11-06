"""RR-10: an image the 0.6.0 upgrade carried over is refused as an exact duplicate once the
source backfill has recorded its content, and not before."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch

from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import content_sha256
from src.services.source_backfill import job
from src.services.source_backfill.content import record_contents
from src.tasks.runtime.context import JobResult, Paused, Yield

from .conftest import make_project, scope_for


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


class _Search:
    """``search`` over a dict of images: sorted by id, after the cursor."""

    def __init__(self, paths: dict[str, str]) -> None:
        self.paths = paths
        self.calls = 0

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        after = (body.get('search_after') or [''])[0]
        ids = [i for i in sorted(self.paths) if i > after][: body['size']]
        return {
            'hits': {'hits': [{'_id': i, '_source': {'image_path': self.paths[i]}} for i in ids]}
        }


class _Ctx:
    def __init__(self, project, client, *, resume_from=None, slice_due=False, paused=False):
        self.project_id = project
        self.project_slug = 'default'
        self.job_id = 'job'
        self.resume_from = resume_from
        self._client = client
        self._slice_due = slice_due
        self._paused = paused
        self.session = scope_for(project)
        self.checkpoints: list[dict[str, Any]] = []

    def cancelled(self) -> bool:
        return False

    def paused(self) -> bool:
        return self._paused

    def slice_due(self) -> bool:
        return self._slice_due

    async def client(self, _name: str) -> Any:
        return self._client

    async def checkpoint(self, cursor: Any) -> None:
        self.checkpoints.append(cursor)

    async def progress(self, *_args: Any, **_kw: Any) -> None:
        return None


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch):
    from contextlib import asynccontextmanager

    state: dict[str, Any] = {}

    async def project_id() -> Any:
        return state['project']

    @asynccontextmanager
    async def bound_job_project(ctx: Any):
        yield ctx.project_id

    monkeypatch.setattr(content_key, 'bound_project_id', project_id)
    monkeypatch.setattr(job, 'bound_job_project', bound_job_project)
    monkeypatch.setattr(job, 'images_index', lambda: 'images')
    return state


async def test_an_upgraded_image_is_refused_after_the_backfill(
    pg_registry, bound, tmp_path: Path
) -> None:
    project = await make_project()
    bound['project'] = project
    data = _jpeg_bytes(seed=21)
    stored = tmp_path / 'old.jpg'
    stored.write_bytes(data)
    svc_os = FakeIngestOpenSearch(
        images={'old-image': {'image_id': 'old-image', 'image_path': str(stored)}}
    )
    svc, _, _ = _make_service(opensearch=svc_os)
    assert await svc._check_duplicates([content_sha256(data)]) == {content_sha256(data): None}

    ctx = _Ctx(project, _Search({'old-image': str(stored)}))
    outcome = await job.run(ctx)  # type: ignore[arg-type]

    assert isinstance(outcome, JobResult)
    assert outcome.result['recorded'] == 1  # type: ignore[index]
    again = await svc.ingest_one(data, '/new/location/copy.jpg')
    assert again.status == 'duplicate'
    assert again.image_id == 'old-image'
    assert len(svc_os.images) == 1


async def test_the_backfill_is_idempotent_and_reports_what_it_could_not_read(
    pg_registry, tmp_path: Path
) -> None:
    project = await make_project()
    (tmp_path / 'a.jpg').write_bytes(b'one')
    (tmp_path / 'b.jpg').write_bytes(b'one')  # the same bytes as a: a holds the content
    pairs = [('a', str(tmp_path / 'a.jpg')), ('b', str(tmp_path / 'b.jpg')), ('c', '/gone.jpg')]

    async with scope_for(project)() as session:
        first = await record_contents(session, project, pairs)
    async with scope_for(project)() as session:
        second = await record_contents(session, project, pairs)

    assert first.as_dict() == {
        'recorded': 1,
        'already': 0,
        'same_bytes_as_another': 1,
        'unreadable': 1,
        'changed': 0,
    }
    # a is already recorded; b and c are tried again and give the same answers
    assert (second.recorded, second.already, second.unreadable) == (0, 1, 1)


async def test_a_slice_yields_and_the_next_delivery_resumes_after_the_cursor(
    pg_registry, bound, tmp_path: Path
) -> None:
    project = await make_project()
    bound['project'] = project
    paths = {}
    for i in range(3):
        f = tmp_path / f'{i}.jpg'
        f.write_bytes(f'bytes-{i}'.encode())
        paths[f'img{i}'] = str(f)
    client = _Search(paths)

    assert isinstance(await job.run(_Ctx(project, client, slice_due=True)), Yield)  # type: ignore[arg-type]
    assert client.calls == 0
    assert isinstance(await job.run(_Ctx(project, client, paused=True)), Paused)  # type: ignore[arg-type]

    resumed = _Ctx(
        project, client, resume_from={'after': 'img0', 'counts': {'recorded': 1}, 'scanned': 1}
    )
    outcome = await job.run(resumed)  # type: ignore[arg-type]
    assert isinstance(outcome, JobResult)
    assert outcome.result['scanned'] == 3  # type: ignore[index]
    assert resumed.checkpoints[-1]['after'] == 'img2'
