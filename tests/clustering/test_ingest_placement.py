"""Ingest stamps an embedded item with its place in the serving partition (4F ``place_new``)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _config_fixtures import make_project
from test_incremental_bands import SPACE, _data, forget_prototypes

from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import activate
from src.services.clustering.run import run_partition
from src.services.curation.item_doc import DetectedItem, build_item_doc


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _serve(project: uuid.UUID) -> Any:
    keys, vectors, centers = _data()
    async with async_session_factory()() as session, project_tx(session, project):
        outcome = await run_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={'n_clusters': 4, 'nredo': 5},
            keys=keys,
            vectors=vectors,
            bands=None,
        )
        await activate(
            session,
            project=project,
            partition_id=outcome.partition_id,
            expected_serving=None,
            expected_revision=None,
            by='test',
        )
    return centers


async def test_an_embedded_item_gets_its_cluster_fields_in_the_doc(
    pg_env, pg, bound_async_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.project_context import current_project
    from src.services.curation import ingest_index

    forget_prototypes()
    slug = f'ip-{uuid.uuid4().hex[:10]}'
    project = make_project(pg, slug)
    monkeypatch.setattr(current_project().record, 'id', project, raising=False)
    monkeypatch.setattr(current_project().record, 'slug', slug, raising=False)
    centers = await _serve(project)
    item = DetectedItem(bbox_pixel=(0, 0, 10, 10), score=0.9, pe_embedding=centers[1] + 0.01)
    bare = DetectedItem(bbox_pixel=(0, 0, 10, 10), score=0.9)
    await ingest_index.place_in_serving_partition([item, bare], ['a', 'b'])
    assert item.placement is not None
    assert item.placement.state == 'assigned'
    assert bare.placement is None
    doc = build_item_doc(
        vector_field='pe_embedding',
        state_field='embedding_state',
        crop_id='a',
        image_id='i',
        image_path='p',
        source='s',
        request_id='r',
        bbox_norm=(0, 0, 0.1, 0.1),
        item=item,
        now='2026-01-01T00:00:00Z',
        crop_area_norm=0.01,
        crop_rank_in_image=1,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )
    assert doc['cluster_assigned_by'] == 'incremental'
    assert doc['cluster_partition_id'] == str(item.placement.partition_id)
    assert doc['cluster_state'] == 'assigned'
