"""``knn_field_body`` renders today's vector fields byte for byte and honours new spec members."""

from __future__ import annotations

import json

from src.clients.curation_opensearch.base import config, default_knn_spec, knn_field_body
from src.clients.curation_opensearch.bodies_core import _images_body, _items_body
from src.domain.embedding_spaces import KnnMethodSpec


def _legacy(dim: int) -> dict:
    """The body the fixed ``_knn_field`` built before the spec existed."""
    return {
        'type': 'knn_vector',
        'dimension': dim,
        'method': {
            'name': 'hnsw',
            'space_type': 'cosinesimil',
            'engine': 'lucene',
            'parameters': {'ef_construction': config.hnsw_ef_construction, 'm': config.hnsw_m},
        },
    }


def _dump(body: dict) -> str:
    return json.dumps(body)  # insertion order is part of "byte-identical"


def test_todays_two_fields_render_byte_identical() -> None:
    for dim in (config.embedding_dim, config.encoder_embedding_dim):
        assert _dump(knn_field_body(default_knn_spec(), dim)) == _dump(_legacy(dim))


def test_the_index_bodies_carry_the_same_fields() -> None:
    images = _images_body()['mappings']['properties']
    items = _items_body()['mappings']['properties']
    assert _dump(images['embedding']) == _dump(_legacy(config.embedding_dim))
    assert _dump(images['pe_embedding']) == _dump(_legacy(config.encoder_embedding_dim))
    assert _dump(items['pe_embedding']) == _dump(_legacy(config.encoder_embedding_dim))
    box = items['region_box_embeddings']['properties']['embedding']
    assert _dump(box) == _dump(_legacy(config.encoder_embedding_dim))


def test_optional_spec_members_appear_only_when_set() -> None:
    plain = knn_field_body(KnnMethodSpec(), 64)
    assert not {'mode', 'compression_level', 'data_type'} & plain.keys()
    rich = knn_field_body(
        KnnMethodSpec(engine='faiss', m=24, mode='on_disk', compression_level='8x'), 64
    )
    assert rich['mode'] == 'on_disk'
    assert rich['compression_level'] == '8x'
    assert rich['method']['engine'] == 'faiss'
    assert rich['method']['parameters'] == {'ef_construction': 128, 'm': 24}
