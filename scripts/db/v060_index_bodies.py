"""The 0.6.0 index bodies of the upgrade fixture (``make_v060_fixture.py``).

The deployment-wide control indexes are literals: the code that created them is gone from the
tree, and the fixture records what a 0.6.0 install held. The per-project bodies come from the
helpers that still build them.
"""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.bodies_other import _configs_body as _folded_configs_body


DIM = 1024


PROJECTS_INDEX_BODY: dict[str, Any] = {
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0}},
    'mappings': {
        'dynamic': False,
        'properties': {
            'slug': {'type': 'keyword'},
            'status': {'type': 'keyword'},
            'revision': {'type': 'long'},
            'created_at': {'type': 'date'},
            'updated_at': {'type': 'date'},
        },
    },
}
GLOBAL_CONFIGS_INDEX_BODY: dict[str, Any] = {
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0}},
    'mappings': {
        'dynamic': False,
        'properties': {
            'doc_type': {'type': 'keyword'},
            'kind': {'type': 'keyword'},
            'name': {'type': 'keyword'},
            'revision': {'type': 'integer'},
            'body': {'type': 'object', 'enabled': False},
            'description': {'type': 'keyword', 'ignore_above': 512, 'index': False},
            'created_at': {'type': 'date'},
            'updated_at': {'type': 'date'},
            'updated_by': {'type': 'keyword'},
            'cloned_from': {'type': 'keyword'},
            'axis': {'type': 'keyword'},
            'previous': {'type': 'object', 'enabled': False},
            'config_revision': {'type': 'long'},
            'process': {'type': 'keyword'},
            'applied_at': {'type': 'date'},
        },
    },
}


def configs_body() -> dict[str, Any]:
    """A 0.6.0 ``__configs`` index also mapped the config-store rows the folded roles share it with."""
    body = _folded_configs_body()
    body['mappings']['properties'].update(GLOBAL_CONFIGS_INDEX_BODY['mappings']['properties'])
    return body


def knn_field(engine: str) -> dict[str, Any]:
    method: dict[str, Any] = {
        'name': 'hnsw',
        'space_type': 'cosinesimil',
        'engine': engine,
        'parameters': {'ef_construction': 128, 'm': 16},
    }
    return {'type': 'knn_vector', 'dimension': DIM, 'method': method}


def items_body(engine: str) -> dict[str, Any]:
    return {
        'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0, 'knn': True}},
        'mappings': {
            'properties': {
                'crop_id': {'type': 'keyword'},
                'class_id': {'type': 'integer'},
                'class_name': {'type': 'keyword'},
                'embedding_state': {'type': 'keyword'},
                'pe_embedding': knn_field(engine),
                'region_box_embeddings': {
                    'type': 'nested',
                    'properties': {'box_id': {'type': 'keyword'}, 'embedding': knn_field(engine)},
                },
            }
        },
    }


def images_body(engine: str) -> dict[str, Any]:
    return {
        'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0, 'knn': True}},
        'mappings': {
            'properties': {'image_id': {'type': 'keyword'}, 'pe_embedding': knn_field(engine)}
        },
    }


def index(body: dict[str, Any], docs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {'mappings': body['mappings'], 'settings': body['settings'], 'docs': docs or {}}
