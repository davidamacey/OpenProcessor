"""Auto-label jobs are addressable by id, and a scoped job writes only in scope.

- ``GET /pipeline/auto_label/status/{job_id}`` answers for the job the
  client started, even after a later job replaced it as the current one;
  an id no job ever had is ``404``.
- A cluster-scoped VLM job's post-VLM ``cluster_id = class_id``
  normalization used to run dataset-wide, moving items that were never in
  the cluster. Every write of a scoped job now stays on the items it
  selected.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch, matches
from curation.test_auto_label_selection import client, job_dir, packs  # noqa: F401 - fixtures
from curation.test_vlm_cluster_scope import CLUSTER, _Labeler, pipeline_env  # noqa: F401
from src.config import get_curation_config


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


ITEMS = get_curation_config().items_index


# =============================================================================
# Job status by id
# =============================================================================


def _finish_current(job_dir: Path, status: str = 'completed') -> None:  # noqa: F811
    """Play the worker: claim the trigger and write a terminal state."""
    (job_dir / 'trigger.json').unlink()
    state = json.loads((job_dir / 'state.json').read_text())
    state.update(status=status, result={'marker': state['job_id']})
    (job_dir / 'state.json').write_text(json.dumps(state))


@pytest.mark.usefixtures('packs')
def test_status_by_id_follows_the_started_job(client: TestClient, job_dir: Path) -> None:  # noqa: F811
    first = client.post('/curation/pipeline/auto_label/start').json()
    job1 = first['job_id']

    r = client.get(f'/curation/pipeline/auto_label/status/{job1}')
    assert r.status_code == 200, r.text
    assert r.json()['job_id'] == job1
    assert r.json()['status'] == 'queued'

    _finish_current(job_dir)
    job2 = client.post('/curation/pipeline/auto_label/start').json()['job_id']
    assert job2 != job1

    # The unscoped status now reports job2; job1 is still answerable.
    assert client.get('/curation/pipeline/auto_label/status').json()['job_id'] == job2
    r = client.get(f'/curation/pipeline/auto_label/status/{job1}')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['job_id'] == job1
    assert body['status'] == 'completed'
    assert body['result'] == {'marker': job1}
    assert client.get(f'/curation/pipeline/auto_label/status/{job2}').json()['job_id'] == job2


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.parametrize('job_id', ['0' * 32, 'not-a-job', '..%2Fstate'])
def test_status_by_unknown_id_is_404(client: TestClient, job_id: str) -> None:  # noqa: F811
    client.post('/curation/pipeline/auto_label/start')
    r = client.get(f'/curation/pipeline/auto_label/status/{job_id}')
    assert r.status_code == 404, r.text


# =============================================================================
# Scoped jobs write only in scope
# =============================================================================


class _Tasks:
    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results

    async def get(self, *, task_id: str) -> dict[str, Any]:
        return {'completed': True, 'response': self._results[task_id]}


class _NormalizingOpenSearch(QueryFakeOpenSearch):
    """Adds ``update_by_query`` with the normalizer's semantics (move a
    classed, non-excluded item into ``cluster_id = class_id``)."""

    def __init__(self, indexes: dict[str, dict[str, dict[str, Any]]]) -> None:
        super().__init__(indexes)
        self._task_results: dict[str, dict[str, Any]] = {}
        self.tasks = _Tasks(self._task_results)
        self.ubq_bodies: list[dict[str, Any]] = []

    async def update_by_query(self, *, index: str, body: dict[str, Any], **_kw: Any) -> Any:
        self.ubq_bodies.append(copy.deepcopy(body))
        updated = 0
        for doc in self.docs(index).values():
            if not matches(doc, body['query']) or doc.get('class_excluded'):
                continue
            if doc.get('class_id') is not None and doc.get('cluster_id') != doc['class_id']:
                doc['cluster_id'] = doc['class_id']
                doc.pop('cluster_subid', None)
                updated += 1
        task = f'task-{len(self.ubq_bodies)}'
        self._task_results[task] = {'updated': updated, 'batches': 1}
        return {'task': task}


def _scope_docs() -> dict[str, dict[str, Any]]:
    base = {'updated_at': '2026-09-01T00:00:00+00:00', 'bbox_norm': [0, 0, 1, 1]}
    return {
        # In the cluster, carrying a class but not yet in its class cluster.
        'in': {**base, 'crop_id': 'in', 'cluster_id': CLUSTER, 'class_id': 3},
        # Outside the cluster, same drift: a scoped job must not touch it.
        'out': {**base, 'crop_id': 'out', 'cluster_id': 10009, 'class_id': 3},
    }


@pytest.mark.asyncio
async def test_cluster_scoped_vlm_job_normalizes_only_its_members(
    pipeline_env: _Labeler,  # noqa: F811
) -> None:
    from src.routers.curation import pipeline

    fake = _NormalizingOpenSearch({ITEMS: _scope_docs()})
    summary = await pipeline.pipeline_auto_label(
        opensearch=fake, train_clusters=False, run_vlm=True, cluster_id=CLUSTER
    )
    docs = fake.docs(ITEMS)
    assert docs['out']['cluster_id'] == 10009
    assert docs['in']['cluster_id'] == 3
    assert summary['stages']['cluster_id_normalize_post_vlm']['updated'] == 1


@pytest.mark.asyncio
async def test_unscoped_vlm_job_still_normalizes_everything(
    pipeline_env: _Labeler,  # noqa: F811
) -> None:
    from src.routers.curation import pipeline

    fake = _NormalizingOpenSearch({ITEMS: _scope_docs()})
    await pipeline.pipeline_auto_label(opensearch=fake, train_clusters=False, run_vlm=True)
    assert fake.docs(ITEMS)['out']['cluster_id'] == 3


@pytest.mark.asyncio
async def test_cluster_scoped_job_skips_dataset_wide_stages(
    pipeline_env: _Labeler,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_clusters / run_auto_promote rewrite cluster ids across the
    whole index; a cluster-scoped job skips them and says so."""
    from unittest.mock import AsyncMock

    from src.routers.curation import pipeline
    from src.services.curation.clustering import orchestrator

    residuals = AsyncMock(return_value={})
    monkeypatch.setattr(orchestrator, 'cluster_residuals', residuals)
    promote = AsyncMock(return_value={})
    monkeypatch.setattr(
        'src.services.curation.clustering.auto_promote.auto_promote_clusters', promote
    )
    fake = _NormalizingOpenSearch({ITEMS: _scope_docs()})
    summary = await pipeline.pipeline_auto_label(
        opensearch=fake,
        train_clusters=True,
        run_auto_promote=True,
        run_vlm=True,
        cluster_id=CLUSTER,
    )
    assert fake.docs(ITEMS)['out']['cluster_id'] == 10009
    residuals.assert_not_awaited()
    promote.assert_not_awaited()
    for stage in ('cluster_id_normalize', 'cluster_residuals', 'auto_promote'):
        assert summary['stages'][stage]['skipped'] is True, stage
        assert 'cluster-scoped' in summary['stages'][stage]['reason'], stage
