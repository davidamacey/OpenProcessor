"""item_scores, selection and embedding_viz on the queue: start, busy, the run, status (4A-3, 4A-4)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fixtures.job_rows import make_project, sql

from curation.jobs_a.ctx import FakeCtx
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.curation import embedding_viz
from src.services.curation.item_scores import job as scores
from src.services.curation.selection import job as selection


pytestmark = pytest.mark.db


async def start_scores(project):
    async with async_session_factory()() as session:
        return await scores.start(session, project, ['uniqueness', 'mistakenness'])


async def start_selection(project):
    async with async_session_factory()() as session:
        return await selection.start(
            session,
            project,
            index='p__items',
            query={'match_all': {}},
            k=3,
            seed_crop_id=None,
            scope={'cluster_id': 4},
            max_n=100,
            embedding_field='pe_embedding',
        )


async def start_viz(project):
    async with async_session_factory()() as session:
        return await embedding_viz.start(session, project, scope='residual', max_n=50)


STARTS = {
    'item_scores': start_scores,
    'selection': start_selection,
    'embedding_viz': start_viz,
}


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', sorted(STARTS))
async def test_start_queues_one_row_and_a_second_start_is_busy(
    kind, pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await STARTS[kind](project)
    assert (job.kind, job.status) == (kind, 'queued')
    assert producer.sent() == [str(job.id)]
    with pytest.raises(RequestRejectedError) as busy:
        await STARTS[kind](project)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert producer.sent() == [str(job.id)]


@pytest.mark.asyncio
async def test_a_viz_start_with_a_bad_scope_is_refused_before_a_row_exists(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    async with async_session_factory()() as session:
        with pytest.raises(ValueError, match='cluster_id is required'):
            await embedding_viz.start(session, project, scope='cluster')
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project) == [(0,)]


class FakeScorer:
    pool = 'residual'

    def __init__(self, name):
        self.name = name

    async def score(self, ids, _embeddings, *, opensearch):  # noqa: ARG002
        return SimpleNamespace(fields={i: {'s': 1} for i in ids}, n_scored=len(ids), extra={})


@pytest.mark.asyncio
async def test_the_scoring_run_scores_each_scorer_and_reports_progress(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start_scores(project)
    pools = {'residual': (['a', 'b', 'c'], np.zeros((3, 2), dtype=np.float32))}

    async def build_pools(_os, needed):
        assert needed == {'residual'}
        return pools

    async def bulk(_os, result, **_k):
        return 0

    monkeypatch.setattr(scores, '_build_pools', build_pools)
    monkeypatch.setattr(scores, 'bulk_write_result', bulk)
    monkeypatch.setattr('src.services.curation.item_scores.get_scorer', FakeScorer)
    ctx = FakeCtx(project, job.id, job.params)

    result = await scores.run_scoring_job(ctx, object(), ['uniqueness', 'mistakenness'])

    assert sorted(result['results']) == ['mistakenness', 'uniqueness']
    assert result['results']['uniqueness']['n_scored'] == 3
    assert ctx.progresses[-1][:3] == (6, 6, 'mistakenness')


@pytest.mark.asyncio
async def test_a_cancel_between_scorers_stops_the_scoring_run(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start_scores(project)
    ctx = FakeCtx(project, job.id, job.params)

    async def build_pools(_os, _needed):
        return {'residual': (['a'], np.zeros((1, 2), dtype=np.float32))}

    async def bulk(_os, _result, **_k):
        ctx.cancel_now()  # the cancel arrives while the first scorer writes
        return 0

    monkeypatch.setattr(scores, '_build_pools', build_pools)
    monkeypatch.setattr(scores, 'bulk_write_result', bulk)
    monkeypatch.setattr('src.services.curation.item_scores.get_scorer', FakeScorer)
    result = await scores.run_scoring_job(ctx, object(), ['uniqueness', 'mistakenness'])
    assert list(result['results']) == ['uniqueness']
    assert result['cancelled'] is True


@pytest.mark.asyncio
async def test_the_selection_run_returns_k_diverse_ids_and_status_serves_them(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start_selection(project)
    ids = ['a', 'b', 'c', 'd']
    vecs = np.array([[1, 0], [0.99, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    seen: dict[str, Any] = {}

    async def fetch(_os, index, query, *, cap, embedding_field):
        seen.update(index=index, cap=cap, field=embedding_field)
        return ids, vecs, False

    monkeypatch.setattr('src.services.curation.selection.pool_fetch.fetch_pool_embeddings', fetch)
    result = await selection.run_selection_job(FakeCtx(project, job.id, job.params), object())

    assert seen == {'index': 'p__items', 'cap': 100, 'field': 'pe_embedding'}
    chosen = result['selection']['crop_ids']
    assert len(chosen) == 3
    assert {'a', 'b'} - set(chosen)  # the near-identical pair is not both picked
    assert result['selection']['n_pool'] == 4


@pytest.mark.asyncio
async def test_the_viz_run_fits_and_writes_through_the_existing_helpers(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start_viz(project)
    calls: list[str] = []

    async def space(target):
        return SimpleNamespace(vector_field='pe_embedding')

    async def fetch(_os, *, scope, cluster_id, max_n, embedding_field):
        calls.append(f'fetch:{scope}:{max_n}:{embedding_field}')
        return ['a', 'b'], np.ones((2, 3), dtype=np.float32)

    async def fit(embeddings):
        calls.append('fit')
        return np.zeros((2, 2), dtype=np.float32), '2026-01-01T00:00:00+00:00'

    async def write(_os, ids, xy):
        calls.append('write')
        return len(ids)

    async def meta(**kw):
        calls.append(f'meta:{kw["n_points"]}')

    monkeypatch.setattr(embedding_viz, 'embedding_space', space)
    monkeypatch.setattr(embedding_viz, '_fetch_pool', fetch)
    monkeypatch.setattr(embedding_viz, 'fit_projection', fit)
    monkeypatch.setattr(embedding_viz, '_bulk_write_coordinates', write)
    monkeypatch.setattr(embedding_viz, '_save_run_metadata', meta)

    result = await embedding_viz.run_projection_job(FakeCtx(project, job.id, job.params), object())

    assert calls == ['fetch:residual:50:pe_embedding', 'fit', 'write', 'meta:2']
    assert result == {
        'n_written': 2,
        'projection_version': embedding_viz.VIZ_PROJECTION_VERSION,
    }


@pytest.mark.asyncio
async def test_a_cancel_before_the_fit_skips_the_fit_and_the_write(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start_viz(project)
    ctx = FakeCtx(project, job.id, job.params)
    ctx.cancel_now()

    async def space(target):
        return SimpleNamespace(vector_field='pe_embedding')

    async def fetch(*_a, **_k):
        return ['a'], np.ones((1, 3), dtype=np.float32)

    async def never(*_a, **_k):
        raise AssertionError('a cancelled run must not fit or write')

    monkeypatch.setattr(embedding_viz, 'embedding_space', space)
    monkeypatch.setattr(embedding_viz, '_fetch_pool', fetch)
    monkeypatch.setattr(embedding_viz, 'fit_projection', never)
    monkeypatch.setattr(embedding_viz, '_bulk_write_coordinates', never)
    assert await embedding_viz.run_projection_job(ctx, object()) == {'cancelled': True}


@pytest.mark.asyncio
async def test_status_dicts_serve_the_old_wire_with_idle_before_any_run(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    from src.services.curation.latest_job import latest_job

    project = make_project(pg)
    monkeypatch.setattr('src.services.curation.latest_job.current_project_id', lambda: project)
    async with async_session_factory()() as session:
        assert scores.state_dict(await latest_job(session, 'item_scores'))['status'] == 'idle'
    await start_scores(project)
    async with async_session_factory()() as session:
        shown = scores.state_dict(await latest_job(session, 'item_scores'))
    assert shown['status'] == 'queued'
    assert shown['scorers'] == ['uniqueness', 'mistakenness']
    assert (shown['processed'], shown['total'], shown['started_at']) == (0, 0, 0.0)
