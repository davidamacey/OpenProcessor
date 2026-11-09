"""The uniqueness scorer takes its coarse quantizer from the serving item partition."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from _config_fixtures import make_project
from test_incremental_bands import forget_prototypes
from test_ingest_placement import _serve

from src.config.project_context import current_project
from src.services.curation.item_scores.uniqueness import UniquenessScorer


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _bind(monkeypatch: pytest.MonkeyPatch, project: uuid.UUID, slug: str) -> None:
    monkeypatch.setattr(current_project().record, 'id', project, raising=False)
    monkeypatch.setattr(current_project().record, 'slug', slug, raising=False)


async def test_the_scorer_reads_the_serving_partitions_prototypes(
    pg_env, pg, bound_async_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    forget_prototypes()
    slug = f'uq-{uuid.uuid4().hex[:10]}'
    project = make_project(pg, slug)
    _bind(monkeypatch, project, slug)
    await _serve(project)
    centroids = await UniquenessScorer()._load_centroids()
    assert centroids.shape[0] == 4  # the four clusters of the served partition
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)


async def test_a_project_with_no_serving_partition_refuses_the_scorer(
    pg_env, pg, bound_async_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    forget_prototypes()
    slug = f'uq-{uuid.uuid4().hex[:10]}'
    _bind(monkeypatch, make_project(pg, slug), slug)
    with pytest.raises(RuntimeError, match='serving item partition'):
        await UniquenessScorer()._load_centroids()
