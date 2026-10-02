"""Follow-up actions a finished job report offers the client.

Every ``next_steps`` entry a job report serves is built here, so the paths
live in one place. A path is relative to the project mount
(``/curation/projects/{project}``); ``tests/curation/test_next_steps_paths.py``
resolves each one against the published OpenAPI.
"""

from __future__ import annotations

from typing import Any


def _step(action: str, path: str, reason: str) -> dict[str, Any]:
    return {'action': action, 'method': 'POST', 'path': path, 'reason': reason}


def recluster_items() -> dict[str, Any]:
    return _step(
        'recluster',
        '/cluster/umap/rebuild',
        'Source clusters were not copied; recluster the combined items.',
    )


def cluster_regions() -> dict[str, Any]:
    return _step(
        'cluster_regions',
        '/regions/cluster',
        'Group the imported region boxes with their visual neighbours.',
    )


ALL_STEPS = (recluster_items, cluster_regions)
