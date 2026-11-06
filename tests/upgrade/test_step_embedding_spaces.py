"""2E-8: an upgraded faiss project records ``engine: faiss`` and a Lucene one ``engine: lucene``."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.services.upgrade.from_060.steps import embedding_spaces as step


def test_the_method_is_read_from_the_mapping_entry() -> None:
    field = {
        'type': 'knn_vector',
        'dimension': 1024,
        'method': {
            'name': 'hnsw',
            'space_type': 'cosinesimil',
            'engine': 'faiss',
            'parameters': {'m': 24, 'ef_construction': 200},
        },
    }
    method = step.knn_method_of(field)
    assert (method.engine, method.m, method.ef_construction) == ('faiss', 24, 200)


def test_a_method_the_registry_cannot_describe_is_refused() -> None:
    field = {'type': 'knn_vector', 'dimension': 8, 'method': {'engine': 'nmslib'}}
    with pytest.raises(step.MappingError, match='nmslib'):
        step.knn_method_of(field)


def test_a_missing_field_is_named() -> None:
    with pytest.raises(step.MappingError, match=r'region_box_embeddings\.embedding'):
        step.field_of({'properties': {'pe_embedding': {}}}, 'region_box_embeddings.embedding')


@pytest.mark.db
@pytest.mark.asyncio
async def test_each_loaded_project_gets_three_spaces_with_the_live_engine(
    make_runner, clean_db
) -> None:
    await make_runner().run()
    with clean_db.engine('super').connect() as conn:
        rows = conn.execute(
            text(
                'SELECT p.slug, s.target, s.index_role, s.vector_field, s.state_field, s.nested_path, '
                "s.knn_method ->> 'engine', s.role, s.status, s.created_by, s.space_key, s.dim "
                'FROM embedding_spaces s JOIN projects p ON p.id = s.project_id ORDER BY 1, 2'
            )
        ).all()
    assert rows == [
        ('default', 'box', 'items', 'region_box_embeddings.embedding', None, 'region_box_embeddings', 'faiss', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
        ('default', 'frame', 'images', 'pe_embedding', None, None, 'faiss', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
        ('default', 'item', 'items', 'pe_embedding', 'embedding_state', None, 'faiss', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
        ('demo', 'box', 'items', 'region_box_embeddings.embedding', None, 'region_box_embeddings', 'lucene', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
        ('demo', 'frame', 'images', 'pe_embedding', None, None, 'lucene', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
        ('demo', 'item', 'items', 'pe_embedding', 'embedding_state', None, 'lucene', 'primary', 'ready', 'upgrade', 'pe_core_l14_336_v1', 1024),
    ]  # fmt: skip
