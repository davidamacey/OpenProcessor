"""The 0.6.0 index bodies of the upgrade fixture (``make_v060_fixture.py``).

Every control index body is a literal: the code that created them is gone from the tree, and the
fixture records what a 0.6.0 install held.
"""

from __future__ import annotations

from typing import Any


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


PLAIN_SETTINGS: dict[str, Any] = {'index': {'number_of_shards': 1, 'number_of_replicas': 0}}
OPEN_OBJECT: dict[str, Any] = {'type': 'object', 'enabled': False}

LABELS_CONFIRMED_BODY: dict[str, Any] = {
    'settings': PLAIN_SETTINGS,
    'mappings': {
        'properties': {
            'label_id': {'type': 'keyword'},
            'image_path': {'type': 'keyword'},
            'bbox_norm': {'type': 'float'},
            'class_id': {'type': 'integer'},
            'class_name': {'type': 'keyword'},
            'label_source': {'type': 'keyword'},
            'confirmed_at': {'type': 'date'},
            'crop_id': {'type': 'keyword'},
            'class_mismatch': {'type': 'boolean'},
            'detector_class_id': {'type': 'integer'},
            'detector_class_name': {'type': 'keyword'},
            'detector_confidence': {'type': 'float'},
            'class_source': {'type': 'keyword'},
            'updated_at': {'type': 'date'},
        }
    },
}
CLASSES_BODY: dict[str, Any] = {
    'settings': PLAIN_SETTINGS,
    'mappings': {
        'properties': {
            'class_id': {'type': 'integer'},
            'class_name': {'type': 'keyword'},
            'group': {'type': 'keyword'},
            'sample_count': {'type': 'long'},
            'validated_count': {'type': 'long'},
            'added_at': {'type': 'date'},
            'deprecated': {'type': 'boolean'},
            'notes': {'type': 'text'},
        }
    },
}
_SETTINGS_DOC_PROPERTIES: dict[str, Any] = {
    'defaults': OPEN_OBJECT,
    'ingest_policy': OPEN_OBJECT,
    'vlm_policy': OPEN_OBJECT,
    'vlm_scope_reps': OPEN_OBJECT,
    'cluster_refresh': OPEN_OBJECT,
    'lease': OPEN_OBJECT,
    'class_registry': OPEN_OBJECT,
    'updated_at': {'type': 'date'},
    'updated_by': {'type': 'keyword'},
}
_VIZ_STATE_PROPERTIES: dict[str, Any] = {
    'state_id': {'type': 'keyword'},
    'projection_version': {'type': 'keyword'},
    'scope': {'type': 'keyword'},
    'cluster_id': {'type': 'integer'},
    'n_points': {'type': 'integer'},
    'fitted_at': {'type': 'date'},
    'n_components': {'type': 'integer'},
    'metric': {'type': 'keyword'},
}
SETTINGS_BODY: dict[str, Any] = {
    'settings': PLAIN_SETTINGS,
    'mappings': {'properties': dict(_SETTINGS_DOC_PROPERTIES)},
}
UMAP_STATE_BODY: dict[str, Any] = {
    'settings': PLAIN_SETTINGS,
    'mappings': {
        'dynamic': False,
        'properties': {
            'state_id': {'type': 'keyword'},
            'reducer_b64': {'type': 'binary'},
            'n_components': {'type': 'integer'},
            'metric': {'type': 'keyword'},
        },
    },
}
UMAP_VIZ_STATE_BODY: dict[str, Any] = {
    'settings': PLAIN_SETTINGS,
    'mappings': {'dynamic': False, 'properties': dict(_VIZ_STATE_PROPERTIES)},
}


def configs_body() -> dict[str, Any]:
    """A 0.6.0 ``__configs`` index carried the folded settings and viz-state roles and the
    config-store rows."""
    return {
        'settings': PLAIN_SETTINGS,
        'mappings': {
            'dynamic': False,
            'properties': {
                **_SETTINGS_DOC_PROPERTIES,
                **_VIZ_STATE_PROPERTIES,
                **GLOBAL_CONFIGS_INDEX_BODY['mappings']['properties'],
            },
        },
    }


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
