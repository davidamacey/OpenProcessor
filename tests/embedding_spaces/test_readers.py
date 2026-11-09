"""What the readers ask the registry for: the item vector field, the write fields, the excludes."""

from __future__ import annotations

import pytest
from _embedding_registry_fake import new_space

from src.config.cluster_profile import default_profile_body
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.services.embedding_spaces import readers


@pytest.mark.asyncio
async def test_the_default_project_reads_the_zero_six_fields() -> None:
    assert await readers.item_vector_field() == 'pe_embedding'
    fields = await readers.item_write_fields()
    assert (fields.vector_field, fields.state_field) == ('pe_embedding', 'embedding_state')
    assert await readers.frame_vector_field() == 'pe_embedding'


@pytest.mark.asyncio
async def test_a_flipped_primary_moves_every_reader_to_the_new_field(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'), primary=True)
    assert await readers.item_vector_field() == 'vec_mobileclip2_s2_v1'
    fields = await readers.item_write_fields()
    assert (fields.vector_field, fields.state_field) == (
        'vec_mobileclip2_s2_v1',
        'vec_mobileclip2_s2_v1_state',
    )


@pytest.mark.asyncio
async def test_the_excludes_cover_every_live_vector_of_the_items_index(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'))
    embedding_registry.add(new_space(Target.BOX, 'mobileclip2_s2'))
    assert await readers.vector_source_excludes() == [
        'boxvec_mobileclip2_s2_v1',  # a box space is excluded whole, by its nested path
        'pe_embedding',
        'region_box_embeddings',
        'vec_mobileclip2_s2_v1',
    ]


@pytest.mark.asyncio
async def test_a_project_without_a_primary_space_refuses(embedding_registry) -> None:
    embedding_registry.primary.pop(Target.ITEM)
    with pytest.raises(EmbeddingSpaceError) as excinfo:
        await readers.item_vector_field()
    assert excinfo.value.code == 'embedding_space_not_found'


def test_the_default_cluster_profile_names_the_space_it_clusters() -> None:
    body = default_profile_body('item', 'vec_x_v1')
    assert body['embedding_field'] == 'vec_x_v1'
    assert default_profile_body('item', 'pe_embedding')['embedding_field'] == 'pe_embedding'


@pytest.mark.asyncio
async def test_the_item_state_fields_are_those_of_every_live_item_space(embedding_registry) -> None:
    assert await readers.item_state_fields() == {'pe_core_l14_336_v1': 'embedding_state'}
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'))
    embedding_registry.add(new_space(Target.BOX, 'mobileclip2_s2'))
    assert await readers.item_state_fields() == {
        'pe_core_l14_336_v1': 'embedding_state',
        'mobileclip2_s2_v1': 'vec_mobileclip2_s2_v1_state',
    }


@pytest.mark.asyncio
async def test_the_images_excludes_are_the_frame_vectors_only(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.FRAME, 'mobileclip2_s2'))
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'))
    assert await readers.vector_source_excludes('images') == [
        'pe_embedding',
        'vec_mobileclip2_s2_v1',
    ]
