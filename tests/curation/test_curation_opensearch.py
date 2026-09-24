"""
Unit tests for `src.clients.curation_opensearch`.

Mocks the AsyncOpenSearch client; no real OpenSearch instance required.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.curation_opensearch import (
    INDEX_BODIES,
    ClassRegistry,
    ClassRegistryError,
    ClassRegistryFile,
    RegistryClassEntry,
    create_curation_indexes,
    ensure_items_probe_fields,
    ensure_items_score_fields,
    get_curation_index_settings,
)
from src.config import IndexRole, get_curation_config


if TYPE_CHECKING:
    from pathlib import Path


config = get_curation_config()


# =============================================================================
# Schema sanity checks
# =============================================================================


def test_index_bodies_has_all_five_roles() -> None:
    assert set(INDEX_BODIES.keys()) == {
        IndexRole.IMAGES,
        IndexRole.ITEMS,
        IndexRole.LABELS_CONFIRMED,
        IndexRole.CLASSES,
        IndexRole.SETTINGS,
    }


def test_images_has_knn_embedding() -> None:
    body = INDEX_BODIES[IndexRole.IMAGES]
    embedding = body['mappings']['properties']['embedding']
    assert embedding['type'] == 'knn_vector'
    assert embedding['dimension'] == config.embedding_dim == 512
    assert embedding['method']['name'] == 'hnsw'
    assert embedding['method']['space_type'] == 'cosinesimil'
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
        'hdd_source',
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
        'v6_embedding',
        # Primary-subject rank + blur quality.
        'crop_rank_in_image',
        'crop_area_norm',
        'blur_lap_ratio',
    }
    assert required_keys.issubset(props.keys()), f'missing: {required_keys - props.keys()}'
    # PE + v6 embeddings are k-NN vectors.
    assert props['pe_embedding']['type'] == 'knn_vector'
    assert props['pe_embedding']['dimension'] == config.encoder_embedding_dim
    assert props['v6_embedding']['type'] == 'knn_vector'


def test_items_has_region_of_interest_fields_via_region_fields() -> None:
    """The region-of-interest sub-annotation fields are keyed by the
    module's RegionFields instance, not hardcoded 'plate_...' literals."""
    from src.clients.curation_opensearch import F

    props = INDEX_BODIES[IndexRole.ITEMS]['mappings']['properties']
    for attr in ('bbox_norm', 'score', 'verified', 'reason', 'detector', 'detected_at'):
        key = getattr(F, attr)
        assert key in props, f'expected region field {key!r} in items mapping'
    # And the region key really is the RegionFields default (region_*), not
    # a hardcoded plate_* literal.
    assert F.status_legacy == 'region_status_legacy'


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
        'dup_group_id',
        'dup_group_size',
        'dup_is_representative',
        'dup_threshold',
        'dup_method',
        'dup_scored_at',
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
    assert props['dup_group_size']['type'] == 'integer'
    assert props['probe_pred_margin']['type'] == 'float'


def test_labels_confirmed_has_no_embedding() -> None:
    props = INDEX_BODIES[IndexRole.LABELS_CONFIRMED]['mappings']['properties']
    assert 'embedding' not in props
    for k in (
        'label_id',
        'image_path',
        'bbox_norm',
        'class_id',
        'class_name',
        'label_source',
        'confirmed_at',
        'crop_id',
    ):
        assert k in props


def test_classes_mirrors_registry_schema() -> None:
    props = INDEX_BODIES[IndexRole.CLASSES]['mappings']['properties']
    assert 'embedding' not in props
    for k in (
        'class_id',
        'class_name',
        'group',
        'sample_count',
        'validated_count',
        'added_at',
        'deprecated',
        'notes',
    ):
        assert k in props


@pytest.mark.asyncio
async def test_get_curation_index_settings_returns_string_keyed_dict() -> None:
    out = await get_curation_index_settings()
    assert set(out.keys()) == {
        'op_images',
        'op_items',
        'op_labels_confirmed',
        'op_classes',
        'op_curation_settings',
    }


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
        'op_images': True,
        'op_items': True,
        'op_labels_confirmed': True,
        'op_classes': True,
        'op_curation_settings': True,
    }
    # Each index was created exactly once with the right body.
    create_calls = client.indices.create.await_args_list
    assert len(create_calls) == 5
    seen = {call.kwargs['index'] for call in create_calls}
    assert seen == {
        'op_images',
        'op_items',
        'op_labels_confirmed',
        'op_classes',
        'op_curation_settings',
    }
    # Each index name got the body for its OWN role, not a mismatched one
    # (catches a role<->index swap bug) — comparing against INDEX_BODIES
    # itself only proves wiring, not content, so also pin one body's
    # actual shape against literal expected values below.
    by_name = {call.kwargs['index']: call.kwargs['body'] for call in create_calls}
    from src.config import index_name

    for role in IndexRole:
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
    # All four exist → no creates issued.
    assert client.indices.create.await_count == 0
    assert client.indices.delete.await_count == 0


@pytest.mark.asyncio
async def test_create_curation_indexes_force_recreate_deletes_first() -> None:
    client = _make_mock_client(exists_returns=True)
    results = await create_curation_indexes(client, force_recreate=True)
    assert all(results.values())
    assert client.indices.delete.await_count == 5
    assert client.indices.create.await_count == 5


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
    assert result['index'] == 'op_items'

    call = client.indices.put_mapping.await_args
    assert call is not None
    assert call.kwargs['index'] == 'op_items'
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


# =============================================================================
# ClassRegistry — file-system backed, tmp_path fixture
# =============================================================================


@pytest.fixture
def tmp_registry(tmp_path: Path) -> ClassRegistry:
    """ClassRegistry pointed at an empty tmp file."""
    path = tmp_path / 'class_registry.json'
    return ClassRegistry(path=path)


def _seeded_registry(path: Path) -> ClassRegistry:
    """Helper: write a seed file with one class, return registry."""
    seed = ClassRegistryFile(
        classes=[RegistryClassEntry(class_id=0, class_name='corvette', group='chevrolet')]
    )
    path.write_text(seed.model_dump_json(indent=2), encoding='utf-8')
    return ClassRegistry(path=path)


def test_registry_load_empty_when_missing(tmp_registry: ClassRegistry) -> None:
    reg = tmp_registry.load()
    assert isinstance(reg, ClassRegistryFile)
    assert reg.classes == []
    assert tmp_registry.next_id() == 0


def test_registry_load_caches_until_mtime_changes(tmp_path: Path) -> None:
    path = tmp_path / 'class_registry.json'
    reg = _seeded_registry(path)
    a = reg.load()
    b = reg.load()
    assert a is b  # cache hit


def test_registry_add_class_assigns_next_id_and_writes(tmp_registry: ClassRegistry) -> None:
    new_id = tmp_registry.add_class('corvette', group='chevrolet', notes='seed')
    assert new_id == 0
    assert tmp_registry.path.exists()
    raw = json.loads(tmp_registry.path.read_text(encoding='utf-8'))
    assert raw['classes'][0]['class_name'] == 'corvette'
    assert raw['classes'][0]['class_id'] == 0
    # next id increments.
    next_id = tmp_registry.add_class('camaro', group='chevrolet')
    assert next_id == 1


def test_registry_add_class_creates_snapshot_on_second_write(
    tmp_registry: ClassRegistry,
) -> None:
    tmp_registry.add_class('corvette', group='chevrolet')
    parent = tmp_registry.path.parent
    snaps_before = sorted(parent.glob('class_registry.*.json'))
    # First add with no prior file → no snapshot.
    assert len(snaps_before) == 0
    tmp_registry.add_class('camaro', group='chevrolet')
    snaps_after = sorted(parent.glob('class_registry.*.json'))
    # Second add — registry already existed → exactly one snapshot.
    assert len(snaps_after) == 1, f'snapshot files found: {snaps_after}'
    snap = json.loads(snaps_after[0].read_text(encoding='utf-8'))
    # Snapshot is the *prior* state — only 'corvette'.
    assert [c['class_name'] for c in snap['classes']] == ['corvette']


def test_registry_add_class_rejects_duplicate_name(tmp_registry: ClassRegistry) -> None:
    tmp_registry.add_class('corvette', group='chevrolet')
    with pytest.raises(ClassRegistryError, match='duplicate class_name'):
        tmp_registry.add_class('corvette', group='chevrolet')


def test_registry_add_class_rejects_empty_name(tmp_registry: ClassRegistry) -> None:
    with pytest.raises(ClassRegistryError, match='non-empty'):
        tmp_registry.add_class('   ', group='chevrolet')


def test_registry_rename_keeps_id_and_updates_name(tmp_registry: ClassRegistry) -> None:
    cid = tmp_registry.add_class('vette', group='chevrolet')
    tmp_registry.rename_class(cid, 'corvette')
    after = tmp_registry.load()
    entry = next(c for c in after.classes if c.class_id == cid)
    assert entry.class_name == 'corvette'


def test_registry_rename_rejects_collision(tmp_registry: ClassRegistry) -> None:
    a = tmp_registry.add_class('corvette', group='chevrolet')
    b = tmp_registry.add_class('camaro', group='chevrolet')
    with pytest.raises(ClassRegistryError, match='already in use'):
        tmp_registry.rename_class(b, 'corvette')
    # a unchanged, b unchanged
    after = tmp_registry.load()
    names = {c.class_id: c.class_name for c in after.classes}
    assert names[a] == 'corvette'
    assert names[b] == 'camaro'


def test_registry_rename_unknown_id_raises(tmp_registry: ClassRegistry) -> None:
    with pytest.raises(ClassRegistryError, match='not found'):
        tmp_registry.rename_class(999, 'whatever')


def test_registry_merge_class_deprecates_source_and_keeps_id(
    tmp_registry: ClassRegistry,
) -> None:
    src = tmp_registry.add_class('corvette_c5', group='chevrolet')
    tgt = tmp_registry.add_class('corvette', group='chevrolet')
    out = tmp_registry.merge_class(src, tgt)
    assert out['source_id'] == src
    assert out['target_id'] == tgt
    assert out['deprecated'] is True

    after = tmp_registry.load()
    src_entry = next(c for c in after.classes if c.class_id == src)
    tgt_entry = next(c for c in after.classes if c.class_id == tgt)
    # Source is still present (ID burned), but deprecated.
    assert src_entry.deprecated is True
    assert src_entry.merged_into == tgt
    # Target stays valid.
    assert tgt_entry.deprecated is False
    # validate_id reflects the deprecation.
    assert tmp_registry.validate_id(src) is False
    assert tmp_registry.validate_id(tgt) is True
    # next_id still increments past the burned source id.
    next_id = tmp_registry.next_id()
    assert next_id == max(src, tgt) + 1


def test_registry_merge_self_raises(tmp_registry: ClassRegistry) -> None:
    cid = tmp_registry.add_class('corvette', group='chevrolet')
    with pytest.raises(ClassRegistryError, match='itself'):
        tmp_registry.merge_class(cid, cid)


def test_registry_merge_into_deprecated_raises(tmp_registry: ClassRegistry) -> None:
    a = tmp_registry.add_class('a', group='g')
    b = tmp_registry.add_class('b', group='g')
    c = tmp_registry.add_class('c', group='g')
    tmp_registry.merge_class(b, c)  # b deprecated
    with pytest.raises(ClassRegistryError, match='deprecated'):
        tmp_registry.merge_class(a, b)


@pytest.mark.asyncio
async def test_registry_sync_to_opensearch_indexes_each_class(
    tmp_registry: ClassRegistry,
) -> None:
    tmp_registry.add_class('a', group='g')
    tmp_registry.add_class('b', group='g')
    client = _make_mock_client(exists_returns=True)
    out = await tmp_registry.sync_to_opensearch(client)
    assert out == {'upserted': 2, 'n_classes': 2}
    # F-26: one bulk() call for both classes (not one index() per class),
    # plus 1 refresh.
    assert client.bulk.await_count == 1
    assert client.indices.refresh.await_count == 1
    bulk_body = client.bulk.await_args.kwargs['body']
    indexed_ids = {action['index']['_id'] for action in bulk_body[0::2]}
    assert indexed_ids == {'0', '1'}
