"""
Unit tests for `src.clients.curation_opensearch`.

Mocks the AsyncOpenSearch client; no real OpenSearch instance required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.curation_opensearch.ensure_overlay_fields import (
    ensure_items_probe_fields,
    ensure_items_score_fields,
)
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES, create_curation_indexes
from src.config import IndexRole, get_curation_config


config = get_curation_config()


# =============================================================================
# Schema sanity checks
# =============================================================================


def test_index_bodies_has_exactly_the_remaining_roles() -> None:
    assert set(INDEX_BODIES.keys()) == {IndexRole.IMAGES, IndexRole.ITEMS}


def test_images_has_knn_embedding() -> None:
    body = INDEX_BODIES[IndexRole.IMAGES]
    embedding = body['mappings']['properties']['embedding']
    assert embedding['type'] == 'knn_vector'
    assert embedding['dimension'] == config.embedding_dim == 512
    assert embedding['method']['name'] == 'hnsw'
    assert embedding['method']['space_type'] == 'cosinesimil'
    assert embedding['method']['engine'] == 'lucene'
    assert embedding['method']['parameters'] == {
        'ef_construction': config.hnsw_ef_construction,
        'm': config.hnsw_m,
    }


def test_items_has_required_fields() -> None:
    props = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    required_keys = {
        'crop_id',
        'image_id',
        'image_path',
        'source',
        'bbox_norm',
        'class_id',
        'class_name',
        'class_source',
        'confidence',
        'cluster_id',
        'cluster_distance',
        'cluster_subid',
        'cluster_auto_suggest',
        'label_validated',
        'label_source',
        'test_holdout',
        'probe_pred_class',
        'probe_pred_entropy',
        'created_at',
        'updated_at',
        'pe_embedding',
        # Primary-subject rank + blur quality.
        'crop_rank_in_image',
        'crop_area_norm',
        'blur_lap_ratio',
    }
    assert required_keys.issubset(props.keys()), f'missing: {required_keys - props.keys()}'
    # The PE embedding is a k-NN vector; the retired backbone vector is gone.
    assert props['pe_embedding']['type'] == 'knn_vector'
    assert props['pe_embedding']['dimension'] == config.encoder_embedding_dim
    assert 'backbone_embedding' not in props
    assert 'v6_embedding' not in props


def test_items_has_region_of_interest_fields_via_region_fields() -> None:
    """The region-of-interest sub-annotation fields are keyed by the
    module's RegionFields instance, not hardcoded 'plate_...' literals."""
    from src.clients.curation_opensearch.base import F

    props = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    for attr in ('verified', 'reason', 'detector_chain', 'detected_at', 'boxes', 'box_embeddings'):
        key = getattr(F, attr)
        assert key in props, f'expected region field {key!r} in items mapping'
    # And the region key really is the RegionFields default (region_*), not
    # a hardcoded plate_* literal.
    assert F.status_before_requeue == 'region_status_before_requeue'


def test_items_has_score_and_probe_fields() -> None:
    """Score + probe provenance fields are declared on a freshly-created
    index, not left to dynamic mapping."""
    props = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    score_fields = {
        'uniqueness_score',
        'uniqueness_method',
        'uniqueness_version',
        'uniqueness_scored_at',
        'mistakenness_score',
        'mistakenness_method',
        'mistakenness_version',
        'mistakenness_scored_at',
        'near_dup_group_id',
        'near_dup_group_size',
        'near_dup_is_representative',
    }
    probe_fields = {
        'probe_pred_confidence',
        'probe_disagreement',
        'probe_pred_margin',
        'probe_model_version',
        'probe_scored_at',
    }
    assert score_fields.issubset(props.keys()), f'missing: {score_fields - props.keys()}'
    assert probe_fields.issubset(props.keys()), f'missing: {probe_fields - props.keys()}'
    assert props['uniqueness_score']['type'] == 'float'
    assert props['near_dup_group_size']['type'] == 'integer'
    assert props['probe_pred_margin']['type'] == 'float'


# =============================================================================
# create_curation_indexes — mocked AsyncOpenSearch
# =============================================================================


def _make_mock_client(exists_returns: bool = False) -> MagicMock:
    """Construct a MagicMock that mimics the AsyncOpenSearch surface we use."""
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.exists = AsyncMock(return_value=exists_returns)
    client.indices.create = AsyncMock(return_value={'acknowledged': True})
    client.indices.delete = AsyncMock(return_value={'acknowledged': True})
    client.indices.refresh = AsyncMock(return_value={})
    client.index = AsyncMock(return_value={'result': 'created'})
    client.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    return client


@pytest.mark.asyncio
async def test_create_curation_indexes_creates_all_when_missing() -> None:
    client = _make_mock_client(exists_returns=False)
    results = await create_curation_indexes(client, force_recreate=False)
    assert results == {
        'op_prj_default__images': True,
        'op_prj_default__items': True,
        'op_prj_default__leases': True,
    }
    create_calls = client.indices.create.await_args_list
    assert len(create_calls) == 2
    by_name = {call.kwargs['index']: call.kwargs['body'] for call in create_calls}
    from src.config import index_name

    # Each index was created once, with its own body.
    for role in (IndexRole.IMAGES, IndexRole.ITEMS):
        assert by_name[index_name(config, role)] == INDEX_BODIES[role]
    # No deletes (we did not force-recreate).
    assert client.indices.delete.await_count == 0

    # Literal pin (not re-derived from INDEX_BODIES) on the images index
    # body actually sent over the wire — would catch a content bug that a
    # comparison against INDEX_BODIES itself never could.
    images_body = by_name[index_name(config, IndexRole.IMAGES)]
    images_props = images_body['mappings']['properties']
    assert images_props['image_id'] == {'type': 'keyword'}
    assert images_props['width'] == {'type': 'integer'}
    assert images_props['height'] == {'type': 'integer'}
    assert images_props['indexed_at'] == {'type': 'date'}
    assert images_props['non_jpeg_format_skipped'] == {'type': 'boolean'}


@pytest.mark.asyncio
async def test_create_curation_indexes_idempotent_when_existing() -> None:
    client = _make_mock_client(exists_returns=True)
    results = await create_curation_indexes(client, force_recreate=False)
    assert all(results.values())
    # Every index exists → no creates issued.
    assert client.indices.create.await_count == 0
    assert client.indices.delete.await_count == 0


@pytest.mark.asyncio
async def test_create_curation_indexes_force_recreate_deletes_first() -> None:
    client = _make_mock_client(exists_returns=True)
    results = await create_curation_indexes(client, force_recreate=True)
    assert all(results.values())
    assert client.indices.delete.await_count == 2
    assert client.indices.create.await_count == 2


@pytest.mark.asyncio
async def test_create_curation_indexes_handles_create_failure() -> None:
    client = _make_mock_client(exists_returns=False)
    client.indices.create = AsyncMock(side_effect=RuntimeError('boom'))
    results = await create_curation_indexes(client, force_recreate=False)
    assert all(v is False for v in results.values())


# =============================================================================
# Curation-scoring migrations — additive + idempotent
# =============================================================================


@pytest.mark.asyncio
async def test_ensure_score_fields_migration_matches_body_and_is_additive() -> None:
    client = _make_mock_client()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    result = await ensure_items_score_fields(client)
    assert result['acknowledged'] is True
    assert result['index'] == 'op_prj_default__items'

    call = client.indices.put_mapping.await_args
    assert call is not None
    assert call.kwargs['index'] == 'op_prj_default__items'
    body_props = call.kwargs['body']['properties']
    assert set(body_props) == set(result['fields_added'])

    # Every field this migration PUTs matches the canonical mapping body
    # (a fresh index already has them; this migration just confirms/adds
    # them on long-lived deployments).
    canonical = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    for field_name, field_body in body_props.items():
        assert canonical[field_name] == field_body


@pytest.mark.asyncio
async def test_ensure_score_fields_migration_idempotent() -> None:
    """Calling the migration twice is a no-op the second time — same
    acknowledged PUT, no exception, no growth in fields_added."""
    client = _make_mock_client()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    first = await ensure_items_score_fields(client)
    second = await ensure_items_score_fields(client)
    assert first['fields_added'] == second['fields_added']
    assert client.indices.put_mapping.await_count == 2


@pytest.mark.asyncio
async def test_ensure_score_fields_migration_swallows_field_conflict() -> None:
    """mapper_parsing_exception / 'already exists' is recoverable — logged,
    not raised (matches the existing ensure_items_* pattern)."""
    client = _make_mock_client()
    client.indices.put_mapping = AsyncMock(
        side_effect=RuntimeError('mapper_parsing_exception: field already exists')
    )
    result = await ensure_items_score_fields(client)
    assert result['acknowledged'] is False
    assert 'error' in result


@pytest.mark.asyncio
async def test_ensure_probe_fields_migration_matches_body_and_is_additive() -> None:
    client = _make_mock_client()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    result = await ensure_items_probe_fields(client)
    assert result['acknowledged'] is True

    call = client.indices.put_mapping.await_args
    assert call is not None
    body_props = call.kwargs['body']['properties']
    assert set(body_props) == set(result['fields_added'])
    assert 'probe_pred_margin' in body_props
    assert 'probe_pred_confidence' in body_props
    assert 'probe_disagreement' in body_props

    canonical = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    for field_name, field_body in body_props.items():
        assert canonical[field_name] == field_body


@pytest.mark.asyncio
async def test_ensure_probe_fields_migration_idempotent() -> None:
    client = _make_mock_client()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    first = await ensure_items_probe_fields(client)
    second = await ensure_items_probe_fields(client)
    assert first == second
    assert client.indices.put_mapping.await_count == 2
