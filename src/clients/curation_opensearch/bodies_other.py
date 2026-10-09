"""Index bodies for every role other than ``images`` and ``items``."""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.base import _plain_settings
from src.clients.curation_opensearch.items_extra import POLICY_DOC_MAPPING


# Mismatch-provenance fields the now-deleted label_import.py used to write,
# plus fields written by the class-merge relabel script (classes.py,
# :506-518). W10 (any_domain_plan.md I13): the labels_confirmed index and
# its mapping stay -- mappings are never dropped -- but nothing writes it
# anymore; the dataset-import ledger is the record now. Kept
# additive-only; see `ensure_labels_confirmed_fields`.
LABELS_CONFIRMED_EXTRA_MAPPING: dict[str, dict[str, Any]] = {
    'class_mismatch': {'type': 'boolean'},
    'detector_class_id': {'type': 'integer'},
    'detector_class_name': {'type': 'keyword'},
    'detector_confidence': {'type': 'float'},
    'class_source': {'type': 'keyword'},
    'updated_at': {'type': 'date'},
}


def _labels_confirmed_body() -> dict[str, Any]:
    return {
        'settings': _plain_settings(),
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
                **LABELS_CONFIRMED_EXTRA_MAPPING,
            }
        },
    }


def _classes_body() -> dict[str, Any]:
    return {
        'settings': _plain_settings(),
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


def _settings_body() -> dict[str, Any]:
    """Curation-strategy shared-defaults document (one row, doc id
    :data:`CURATION_SETTINGS_DOC_ID`) -- backs ``GET/PUT /curation/settings``
    and :func:`~src.services.curation.strategy_registry.resolve_effective_default`.

    ``defaults`` is deliberately ``enabled: false`` (stored, never
    indexed/searchable) rather than a strict per-axis mapping: it is an
    OPEN map keyed by axis id (a future axis must not require a mapping
    change / reindex), and nothing ever queries into it -- every read is
    a single ``GET`` by the fixed doc id, never a search. OpenSearch's
    partial ``update`` API still does its normal recursive object merge
    against `_source` regardless of ``enabled``, which is exactly what a
    partial ``PUT /curation/settings`` needs (merge one axis in without
    clobbering the others).
    """
    return {
        'settings': _plain_settings(),
        'mappings': {
            'properties': {
                'defaults': {'type': 'object', 'enabled': False},
                **POLICY_DOC_MAPPING,
                'updated_at': {'type': 'date'},
                'updated_by': {'type': 'keyword'},
            }
        },
    }


def _umap_state_body() -> dict[str, Any]:
    """The retired clustering reducer's fitted-manifold cache
    (``clustering/embedding_reduce.py``). ``reducer_b64`` is a pickled
    UMAP reducer, base64-encoded, up to ~60 MB (see
    ``_OPENSEARCH_PERSIST_MAX_BYTES``) -- mapped ``binary`` (stored,
    never analyzed/indexed) rather than left to dynamic mapping, which
    tokenized it as ``text``."""
    return {
        'settings': _plain_settings(),
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


def _umap_viz_state_body() -> dict[str, Any]:
    """Visualization-only projection's own metadata slot
    (``src/services/curation/embedding_viz.py``) -- deliberately
    distinct from :func:`_umap_state_body`. Metadata only, no pickled
    blob."""
    return {
        'settings': _plain_settings(),
        'mappings': {
            'dynamic': False,
            'properties': {
                'state_id': {'type': 'keyword'},
                'projection_version': {'type': 'keyword'},
                'scope': {'type': 'keyword'},
                'cluster_id': {'type': 'integer'},
                'n_points': {'type': 'integer'},
                'fitted_at': {'type': 'date'},
                'n_components': {'type': 'integer'},
                'metric': {'type': 'keyword'},
            },
        },
    }


def _configs_body() -> dict[str, Any]:
    """The config store (W2, any_domain_plan.md §3.1): prompt packs,
    region profiles, activations, activation history and the
    cross-process revision counter -- one doc per row, ``doc_type``
    discriminates the shape (``config`` | ``revision`` | ``activation``
    | ``activation_event`` | ``meta`` | ``runtime``).

    For a project created after this wave, ``resources_for_new`` also
    points the ``SETTINGS`` and ``UMAP_VIZ_STATE`` roles at this same
    index name (shard folding, owner D4, projects_plan.md §2.3), so this
    mapping carries their properties too (``_settings_body()`` /
    ``_umap_viz_state_body()``, field-for-field identical types --
    verified conflict-free in ``test_configs_mapping_union.py``). Those
    two roles are read/written by fixed doc id only (``default`` /
    ``current``), never searched, and every config-store doc has a
    ``doc_type`` while the folded docs have none, so a
    ``doc_type``-filtered config-store query never sees them (see
    ``test_folded_roles_by_id_only.py``).
    """
    return {
        'settings': _plain_settings(),
        'mappings': {
            'dynamic': False,
            'properties': {
                # -- config-store rows --
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
                # -- folded SETTINGS (op_curation_settings doc `default`) --
                'defaults': {'type': 'object', 'enabled': False},
                **POLICY_DOC_MAPPING,
                # -- folded UMAP_VIZ_STATE (doc `current`) --
                'state_id': {'type': 'keyword'},
                'projection_version': {'type': 'keyword'},
                'scope': {'type': 'keyword'},
                'cluster_id': {'type': 'integer'},
                'n_points': {'type': 'integer'},
                'fitted_at': {'type': 'date'},
                'n_components': {'type': 'integer'},
                'metric': {'type': 'keyword'},
            },
        },
    }
