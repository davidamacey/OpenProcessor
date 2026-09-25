"""Tests for the visualization-only UMAP projection overlay.

Never fits a real UMAP here — ``fit_projection`` (the only
function in ``embedding_viz.py`` that imports ``umap``) is monkeypatched
wholesale in every test that exercises the job lifecycle, the same
convention ``test_scores_router.py``'s ``run_scoring_job`` monkeypatch
uses for its own job entrypoint.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest


# =============================================================================
# Own state slot — never the retired clustering reducer's names
# =============================================================================


def test_viz_state_paths_are_distinct_from_retired_clustering_reducer() -> None:
    from src.services.curation import embedding_viz
    from src.services.curation.clustering import embedding_reduce

    assert embedding_viz.VIZ_STATE_JOBLIB_PATH != embedding_reduce.UMAP_STATE_JOBLIB_PATH
    assert embedding_viz.VIZ_STATE_JOBLIB_PATH != embedding_reduce.UMAP_STATE_JOBLIB_PATH_CUML
    assert embedding_viz.UMAP_VIZ_STATE_INDEX != embedding_reduce.UMAP_STATE_INDEX


# =============================================================================
# _subsample — deterministic downsample
# =============================================================================


def test_subsample_noop_when_pool_fits() -> None:
    from src.services.curation.embedding_viz import _subsample

    ids = [f'c{i}' for i in range(5)]
    emb = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    out_ids, out_emb = _subsample(ids, emb, max_n=10)
    assert out_ids == ids
    assert np.array_equal(out_emb, emb)


def test_subsample_deterministic_and_capped() -> None:
    from src.services.curation.embedding_viz import _subsample

    ids = [f'c{i}' for i in range(1000)]
    emb = np.arange(1000 * 4, dtype=np.float32).reshape(1000, 4)
    ids_a, emb_a = _subsample(ids, emb, max_n=50, seed=7)
    ids_b, emb_b = _subsample(ids, emb, max_n=50, seed=7)
    assert ids_a == ids_b
    assert np.array_equal(emb_a, emb_b)
    assert len(ids_a) == 50
    assert set(ids_a) <= set(ids)


# =============================================================================
# start_job — scope validation
# =============================================================================


def test_start_job_rejects_unknown_scope(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    with pytest.raises(ValueError, match='scope'):
        embedding_viz.start_job(AsyncMock(), scope='bogus')


def test_start_job_requires_cluster_id_for_cluster_scope(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    with pytest.raises(ValueError, match='cluster_id'):
        embedding_viz.start_job(AsyncMock(), scope='cluster', cluster_id=None)


# =============================================================================
# run_projection_job — full lifecycle over a synthetic pool, fit mocked
# =============================================================================


async def _await_active_task() -> None:
    """``start_job`` schedules a real ``asyncio.create_task`` -- await it
    directly instead of racing the event loop (these tests don't go
    through a ``TestClient``, so there's no separate loop iteration to
    lean on). A local variable narrows the ``Task[None] | None`` type for
    mypy in a way a bare module-attribute assert doesn't."""
    from src.services.curation import embedding_viz

    task = embedding_viz._active_task
    assert task is not None
    await task


def _fake_residual_fetch(ids_and_embeddings: list[tuple[str, list[float]]]):
    ids = [cid for cid, _ in ids_and_embeddings]
    emb = np.asarray([e for _, e in ids_and_embeddings], dtype=np.float32)

    async def _fake(*args: Any, **kwargs: Any) -> tuple[list[str], np.ndarray]:
        return ids, emb

    return _fake


@pytest.mark.asyncio
async def test_run_projection_job_writes_only_viz_fields_and_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))

    pool = [(f'crop-{i}', [float(i), float(i + 1)]) for i in range(6)]
    monkeypatch.setattr(
        'src.services.curation.clustering.embedding_reduce.fetch_residual_embeddings_parallel',
        _fake_residual_fetch(pool),
    )

    fake_xy = np.arange(len(pool) * 2, dtype=np.float32).reshape(len(pool), 2)

    async def _fake_fit(embeddings):
        return fake_xy, '2026-09-11T00:00:00+00:00'

    monkeypatch.setattr(embedding_viz, 'fit_projection', _fake_fit)

    fake_os = AsyncMock()
    bulk_bodies: list[list[dict[str, Any]]] = []

    async def _capture_bulk(body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        bulk_bodies.append(body)
        return {'errors': False}

    fake_os.bulk = AsyncMock(side_effect=_capture_bulk)
    fake_os.index = AsyncMock(return_value=None)

    state = embedding_viz.start_job(fake_os, scope='residual', max_n=100)
    job_id = state['job_id']
    await _await_active_task()

    final_state = embedding_viz.get_state()
    assert final_state['job_id'] == job_id
    assert final_state['status'] == 'completed'
    assert final_state['n_written'] == len(pool)
    assert final_state['projection_version'] == embedding_viz.VIZ_PROJECTION_VERSION

    # Metadata was saved to embedding_viz's OWN index, never op_umap_state.
    fake_os.index.assert_awaited_once()
    _, index_kwargs = fake_os.index.call_args
    assert index_kwargs['index'] == embedding_viz.UMAP_VIZ_STATE_INDEX
    assert index_kwargs['index'] != 'op_umap_state'

    # Every bulk doc body writes ONLY viz_x/viz_y/viz_projection_version --
    # never cluster_id/cluster_subid/cluster_distance.
    forbidden = {'cluster_id', 'cluster_subid', 'cluster_distance'}
    written_ids: set[str] = set()
    for body in bulk_bodies:
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            assert set(action.keys()) == {'update'}
            written_ids.add(action['update']['_id'])
            assert set(doc['doc'].keys()) == {'viz_x', 'viz_y', 'viz_projection_version'}
            assert not (set(doc['doc'].keys()) & forbidden)
    assert written_ids == {cid for cid, _ in pool}


@pytest.mark.asyncio
async def test_run_projection_job_empty_pool_completes_with_zero_written(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    monkeypatch.setattr(
        'src.services.curation.clustering.embedding_reduce.fetch_residual_embeddings_parallel',
        _fake_residual_fetch([]),
    )

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('fit_projection must not be called for an empty pool')

    monkeypatch.setattr(embedding_viz, 'fit_projection', _boom)

    fake_os = AsyncMock()
    embedding_viz.start_job(fake_os, scope='residual', max_n=100)
    await _await_active_task()

    final_state = embedding_viz.get_state()
    assert final_state['status'] == 'completed'
    assert final_state['n_written'] == 0
    fake_os.bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_double_start_raises_runtime_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))

    hang_forever = asyncio.Event()

    async def _fake_run(job_id, opensearch, *, scope, cluster_id, max_n) -> None:
        embedding_viz._touch_heartbeat()
        await hang_forever.wait()

    monkeypatch.setattr(embedding_viz, 'run_projection_job', _fake_run)

    fake_os = AsyncMock()
    embedding_viz.start_job(fake_os, scope='residual')
    with pytest.raises(RuntimeError, match='already in progress'):
        embedding_viz.start_job(fake_os, scope='residual')

    assert embedding_viz.cancel_job() is True
    hang_forever.set()
    await _await_active_task()


# =============================================================================
# get_cached_projection — the GET-only read path. MUST NEVER trigger a fit.
# =============================================================================


@pytest.mark.asyncio
async def test_get_cached_projection_not_built_when_no_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation import embedding_viz

    fake_os = AsyncMock()
    fake_os.get = AsyncMock(side_effect=Exception('not found'))

    result = await embedding_viz.get_cached_projection(fake_os)
    assert result == {'status': 'not_built'}
    fake_os.search.assert_not_called()


@pytest.mark.asyncio
async def test_get_cached_projection_never_triggers_a_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing assertion: the GET
    read path is genuinely unreachable from the fit function. Monkeypatch
    `fit_projection` to explode if ever called, then drive the
    full GET path with a metadata doc present and confirm no exception."""
    from src.services.curation import embedding_viz

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('get_cached_projection must never call fit_projection')

    monkeypatch.setattr(embedding_viz, 'fit_projection', _boom)
    monkeypatch.setattr(embedding_viz, '_build_reducer', _boom)

    fake_os = AsyncMock()
    fake_os.get = AsyncMock(
        return_value={
            '_source': {
                'projection_version': 'umap_viz_v1',
                'fitted_at': '2026-09-11T00:00:00+00:00',
                'scope': 'residual',
                'cluster_id': None,
                'n_points': 2,
            }
        }
    )
    fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    {
                        '_id': 'crop-a',
                        '_source': {
                            'viz_x': 1.0,
                            'viz_y': 2.0,
                            'cluster_id': 10173,
                            'class_name': '',
                            'class_source': 'vlm_unmatched',
                        },
                    },
                ]
            }
        }
    )
    fake_os.count = AsyncMock(return_value={'count': 0})

    result = await embedding_viz.get_cached_projection(fake_os, max_points=10)
    assert result['projection_version'] == 'umap_viz_v1'
    assert result['fitted_at'] == '2026-09-11T00:00:00+00:00'
    assert result['stale'] is False
    assert result['points'] == [
        {
            'crop_id': 'crop-a',
            'x': 1.0,
            'y': 2.0,
            'cluster_id': 10173,
            'class_name': '',
            'class_source': 'vlm_unmatched',
        }
    ]


@pytest.mark.asyncio
async def test_get_cached_projection_stale_when_uncovered_crops_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation import embedding_viz

    fake_os = AsyncMock()
    fake_os.get = AsyncMock(
        return_value={'_source': {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}}
    )
    fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    fake_os.count = AsyncMock(return_value={'count': 3})  # 3 in-scope crops missing the projection

    result = await embedding_viz.get_cached_projection(fake_os)
    assert result['stale'] is True


@pytest.mark.asyncio
async def test_get_cached_projection_applies_cluster_and_class_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation import embedding_viz

    fake_os = AsyncMock()
    fake_os.get = AsyncMock(
        return_value={'_source': {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}}
    )
    fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    fake_os.count = AsyncMock(return_value={'count': 0})

    await embedding_viz.get_cached_projection(fake_os, cluster_id=10173, class_id=7, max_points=5)

    search_kwargs = fake_os.search.call_args.kwargs
    filter_clauses = search_kwargs['body']['query']['bool']['filter']
    assert {'term': {'cluster_id': 10173}} in filter_clauses
    assert {'term': {'class_id': 7}} in filter_clauses
    assert search_kwargs['body']['size'] == 5


@pytest.mark.asyncio
async def test_get_cached_projection_pages_with_search_after_no_oversized_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenSearch's index.max_result_window is 10000, so a
    single `size: max_points` request always 400s once max_points exceeds
    it. get_cached_projection() must page with search_after in bounded
    chunks instead -- no single search request may ask for size > 10000,
    even when max_points is the router's full 200_000 cap.
    """
    from src.services.curation import embedding_viz

    fake_os = AsyncMock()
    fake_os.get = AsyncMock(
        return_value={'_source': {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}}
    )
    fake_os.count = AsyncMock(return_value={'count': 0})

    page_size = embedding_viz._VIZ_PROJECTION_PAGE_SIZE

    def _page(crop_id: str) -> dict[str, Any]:
        return {
            '_id': crop_id,
            '_source': {
                'viz_x': 1.0,
                'viz_y': 2.0,
                'cluster_id': 1,
                'class_name': 'x',
                'class_source': 'vlm',
            },
            'sort': [crop_id],
        }

    # Two full pages, then a short (final) page -> loop must stop there.
    responses = [
        {'hits': {'hits': [_page(f'a-{i}') for i in range(page_size)]}},
        {'hits': {'hits': [_page(f'b-{i}') for i in range(page_size)]}},
        {'hits': {'hits': [_page('c-0')]}},
    ]
    fake_os.search = AsyncMock(side_effect=responses)

    result = await embedding_viz.get_cached_projection(fake_os, max_points=200_000)

    assert fake_os.search.await_count == 3
    for call in fake_os.search.await_args_list:
        assert call.kwargs['body']['size'] <= 10_000
        assert call.kwargs['body']['sort'] == [{'crop_id': 'asc'}]
        assert call.kwargs['body']['track_total_hits'] is False
    # second and third calls carry search_after from the previous page's last hit
    assert fake_os.search.await_args_list[1].kwargs['body']['search_after'] == [
        f'a-{page_size - 1}'
    ]
    assert fake_os.search.await_args_list[2].kwargs['body']['search_after'] == [
        f'b-{page_size - 1}'
    ]
    assert len(result['points']) == 2 * page_size + 1


# =============================================================================
# Regression guard — never writes cluster fields, mirrors
# test_crop_scores.py::test_no_scorer_writes_cluster_fields /
# test_select_router.py's structural source-scan guard.
# =============================================================================

_FORBIDDEN_CLUSTER_FIELDS = frozenset({'cluster_id', 'cluster_subid', 'cluster_distance'})


def test_bulk_write_coordinates_writes_only_viz_fields() -> None:
    import inspect

    from src.services.curation import embedding_viz

    src = inspect.getsource(embedding_viz._bulk_write_coordinates)
    assert "'viz_x'" in src
    assert "'viz_y'" in src
    assert "'viz_projection_version'" in src
    for forbidden in _FORBIDDEN_CLUSTER_FIELDS:
        assert f"'{forbidden}'" not in src


def test_embedding_viz_source_never_mutates_cluster_fields() -> None:
    """Structural guard over the whole module: no forbidden field name
    ever appears as a bulk/update doc-body key."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / 'src/services/curation/embedding_viz.py').read_text()
    # The only `doc` dict literal in this module is _bulk_write_coordinates's
    # -- already asserted exactly above. This is a coarse belt-and-suspenders
    # scan for any of the forbidden field names appearing anywhere as a
    # quoted dict key (read-side `cluster_id` filters use a bare `{'term':
    # {'cluster_id': ...}}` shape too, so this intentionally doesn't forbid
    # the name outright -- just confirms no *new* write site was added
    # without updating this test).
    doc_body_pattern = re.compile(r"'doc':\s*\{([^}]*)\}", re.DOTALL)
    for match in doc_body_pattern.finditer(src):
        for forbidden in _FORBIDDEN_CLUSTER_FIELDS:
            assert forbidden not in match.group(1), (
                f'embedding_viz.py writes forbidden field {forbidden!r} in a doc body'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
