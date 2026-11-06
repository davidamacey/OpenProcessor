"""A new item is placed by the serving partition's prototypes with the calibrated bands."""

from __future__ import annotations

import uuid

import numpy as np
import psycopg
import pytest
from _config_fixtures import make_project

from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import activate
from src.services.clustering.assign import AssignBands, band_state, forget_prototypes, place_new
from src.services.clustering.partitions import PartitionSpace
from src.services.clustering.run import run_partition


pytestmark = [pytest.mark.db, pytest.mark.asyncio]
SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


def test_without_a_calibration_every_item_is_assigned_as_today() -> None:
    assert band_state(0.01, 0.0, None) == 'assigned'


def test_bands_split_assigned_review_and_unassigned() -> None:
    bands = AssignBands(assign_min=0.8, review_min=0.5, min_margin=0.1)
    assert band_state(0.9, 0.3, bands) == 'assigned'
    assert band_state(0.9, 0.05, bands) == 'review'
    assert band_state(0.6, 0.5, bands) == 'review'
    assert band_state(0.4, 0.5, bands) == 'unassigned'


def _data():
    rng = np.random.default_rng(5)
    centers = np.eye(16)[:4] * 5.0
    vectors = np.vstack([c + rng.normal(size=(40, 16)) * 0.1 for c in centers]).astype(np.float32)
    return [f'crop{i:04d}' for i in range(len(vectors))], vectors, centers.astype(np.float32)


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'ib-{uuid.uuid4().hex[:10]}')


async def _serving(project, pg, bands_body: dict | None):
    keys, vectors, centers = _data()
    calibration = None
    if bands_body is not None:
        with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
            calibration = conn.execute(
                'INSERT INTO cluster_calibrations (project_id, target, level, embedding_field, '
                'embedding_model, preprocess_id, kind, fitted_on, body, dev_metrics, created_by) '
                "VALUES (%s, 'item', 'class', 'pe_embedding', 'pe_core@v1', 'pe_core_v1', "
                "'assign_bands', 'pin:x', %s::jsonb, '{}', 'test') RETURNING id",
                (project, __import__('json').dumps(bands_body)),
            ).fetchone()[0]
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
            bands=(calibration, AssignBands(**bands_body)) if bands_body else None,
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


async def test_below_the_review_band_an_item_is_unassigned(pg_env, pg, project) -> None:
    forget_prototypes()
    centers = await _serving(project, pg, {'assign_min': 0.9, 'review_min': 0.5, 'min_margin': 0.0})
    near = centers[0] + 0.05
    far = np.ones(16, dtype=np.float32)  # equidistant from every center: low score
    async with async_session_factory()() as session, project_tx(session, project):
        placed = await place_new(
            session,
            project=project,
            target='item',
            item_ids=['n1', 'n2'],
            vectors=np.vstack([near, far]),
        )
    by_id = {p.item_id: p for p in placed}
    assert by_id['n1'].state == 'assigned'
    assert by_id['n1'].cluster_uid is not None
    assert by_id['n2'].state == 'unassigned'
    assert by_id['n2'].cluster_uid is None
    assert by_id['n1'].assigned_by == 'incremental'


async def test_without_a_calibration_the_nearest_prototype_wins(pg_env, pg, project) -> None:
    forget_prototypes()
    await _serving(project, pg, None)
    async with async_session_factory()() as session, project_tx(session, project):
        placed = await place_new(
            session,
            project=project,
            target='item',
            item_ids=['x'],
            vectors=np.ones((1, 16), dtype=np.float32),
        )
    assert placed[0].state == 'assigned'


async def test_a_project_with_no_serving_partition_places_nothing(pg_env, project) -> None:
    forget_prototypes()
    async with async_session_factory()() as session, project_tx(session, project):
        assert (
            await place_new(
                session,
                project=project,
                target='item',
                item_ids=['x'],
                vectors=np.ones((1, 16), dtype=np.float32),
            )
            == []
        )
