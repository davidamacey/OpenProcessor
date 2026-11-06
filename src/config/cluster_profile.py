"""The clustering profile's defaults and axes (research clustering plan 5.4).

The body model is ``ClusterProfileBody`` (``src/schemas/curation/research_config.py``); a project
that never activated a ``cluster_profile`` gets the defaults below, which are the values the
retired ``OP_CLUSTER_REFRESH_*`` and ``OP_IVF_RETRAIN_*`` variables had. ``OP_CLUSTER_REFRESH_INTERVAL_S``
stays an environment variable: it is the beat's clock, not a per-project rule.
"""

from __future__ import annotations

from typing import Any, Final


TARGET_AXES: Final[dict[str, str]] = {
    'item': 'clustering_item',
    'region_box': 'clustering_region_box',
}
DEFAULT_REFRESH: Final[dict[str, Any]] = {
    'growth_threshold': 200,
    'max_deferral_s': 1800,
    'retry_backoff_s': 600,
}
_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    'item': {
        'target': 'item',
        'level': 'class',
        'embedding_field': 'pe_embedding',
        'method': 'ivf',
        'params': {'n_clusters': 512},
        'refresh': DEFAULT_REFRESH,
        'activation': 'human',
        'backend': 'cpu',
    },
    'region_box': {
        'target': 'region_box',
        'level': 'class',
        'embedding_field': 'region_box_embeddings.embedding',
        'method': 'minibatch_kmeans',
        'params': {'target_bucket_size': 200},
        'refresh': DEFAULT_REFRESH,
        'activation': 'human',
        'backend': 'cpu',
    },
}


def default_profile_body(target: str) -> dict[str, Any]:
    """The profile body of a project that activated none (a copy; callers may validate it)."""
    try:
        body = _DEFAULTS[target]
    except KeyError:
        raise ValueError(f'no cluster profile for target {target!r}') from None
    return {**body, 'params': dict(body['params']), 'refresh': dict(body['refresh'])}
