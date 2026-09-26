"""Read-only OpenSearch shard/heap capacity check, served on
``GET /projects`` so Cropwright can show it before create (see
docs/design/openprocessor_internal/projects_plan.md §2.3).

P1 only serves this; P3 adds the create-time 409 enforcement. There is
no fixed project-count cap (owner decision D4) -- the real limits are
the cluster's hard shard limit and its heap, checked here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal


CapacityStatus = Literal['ok', 'warn', 'blocked']

_CACHE_TTL_SECONDS = 10.0
_cache: tuple[float, ProjectCapacity | None] | None = None


@dataclass(frozen=True)
class ProjectCapacity:
    status: CapacityStatus
    active_shards: int
    per_project_shards: int
    soft_limit: int
    hard_limit: int
    heap_max_bytes: int
    max_shards_per_node: int
    data_nodes: int
    projects_until_soft_limit: int
    message: str
    labels: dict[str, str]

    def to_wire(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'active_shards': self.active_shards,
            'per_project_shards': self.per_project_shards,
            'soft_limit': self.soft_limit,
            'hard_limit': self.hard_limit,
            'heap_max_bytes': self.heap_max_bytes,
            'max_shards_per_node': self.max_shards_per_node,
            'data_nodes': self.data_nodes,
            'projects_until_soft_limit': self.projects_until_soft_limit,
            'message': self.message,
            'labels': self.labels,
        }


_LABELS = {
    'ok': 'Room for more projects',
    'warn': 'Near the recommended shard budget',
    'blocked': 'No room for another project',
}


def per_project_shards() -> int:
    """The number of distinct index names a brand-new project creates
    today -- computed from ``resources_for_new`` rather than
    hardcoded, so this tracks reality as roles are added/folded (W2
    folds two roles into ``configs``; that fold has not landed in this
    codebase yet, so this is currently 7, not 6 -- see the PR notes)."""
    from src.config.curation import base_curation_config
    from src.config.projects import resources_for_new

    resources = resources_for_new('__capacity_probe__', base_curation_config())
    return len(set(resources.indexes.values()))


def _shards_per_heap_gb() -> float:
    return float(os.environ.get('OP_SHARDS_PER_HEAP_GB', '20'))


async def _fetch_cluster_facts(opensearch: Any) -> tuple[int, int, int, int]:
    """Returns (active_shards, data_nodes, max_shards_per_node,
    heap_max_bytes_total). Every call here hits a global (non-index)
    endpoint, so it needs no project bound."""
    health = await opensearch.transport.perform_request('GET', '/_cluster/health')
    active_shards = int(health.get('active_shards', 0))
    data_nodes = int(health.get('number_of_data_nodes', 1)) or 1

    settings = await opensearch.transport.perform_request(
        'GET',
        '/_cluster/settings',
        params={'include_defaults': 'true', 'flat_settings': 'true'},
    )
    max_shards_per_node = None
    for scope in ('transient', 'persistent', 'defaults'):
        value = (settings.get(scope) or {}).get('cluster.max_shards_per_node')
        if value is not None:
            max_shards_per_node = int(value)
            break
    if max_shards_per_node is None:
        max_shards_per_node = 1000  # OpenSearch's own default

    stats = await opensearch.transport.perform_request('GET', '/_nodes/stats/jvm')
    heap_max_bytes_total = sum(
        int(((node.get('jvm') or {}).get('mem') or {}).get('heap_max_in_bytes', 0))
        for node in (stats.get('nodes') or {}).values()
    )
    return active_shards, data_nodes, max_shards_per_node, heap_max_bytes_total


def _build_capacity(
    *,
    active_shards: int,
    data_nodes: int,
    max_shards_per_node: int,
    heap_max_bytes_total: int,
    extra_shards: int,
) -> ProjectCapacity:
    hard_limit = max_shards_per_node * data_nodes
    heap_gb_total = heap_max_bytes_total / (1024**3)
    soft_limit = min(hard_limit, int(heap_gb_total * _shards_per_heap_gb()))
    shards = per_project_shards()

    projected = active_shards + extra_shards
    if projected > hard_limit:
        status: CapacityStatus = 'blocked'
        message = (
            f'Creating this project needs {extra_shards} more OpenSearch shards, but the cluster '
            f'is at {active_shards} of its {hard_limit}-shard limit (cluster.max_shards_per_node). '
            'Delete a project you no longer need, or raise cluster.max_shards_per_node and the '
            'heap together.'
        )
    elif projected > soft_limit:
        status = 'warn'
        message = (
            f'OpenSearch has a {heap_gb_total:.0f} GB heap, which serves about {soft_limit} '
            f'shards well; this project brings the total to {projected}. Searches may slow down. '
            f'Raise OPENSEARCH_HEAP in .env (about 1 GB per {int(_shards_per_heap_gb())} shards) '
            'and restart opensearch.'
        )
    else:
        status = 'ok'
        message = 'Plenty of shard/heap headroom for another project.'

    projects_until_soft_limit = max(0, (soft_limit - active_shards) // shards) if shards else 0

    return ProjectCapacity(
        status=status,
        active_shards=active_shards,
        per_project_shards=shards,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        heap_max_bytes=heap_max_bytes_total,
        max_shards_per_node=max_shards_per_node,
        data_nodes=data_nodes,
        projects_until_soft_limit=projects_until_soft_limit,
        message=message,
        labels=dict(_LABELS),
    )


async def capacity_status(
    opensearch: Any, *, extra_shards: int | None = None
) -> ProjectCapacity | None:
    """Cached (10s/process) capacity read. Returns ``None`` only when
    OpenSearch is unreachable -- callers serve ``capacity: null`` in that
    case rather than a fabricated status."""
    global _cache  # noqa: PLW0603 - process-wide 10s cache

    if extra_shards is None:
        extra_shards = per_project_shards()

    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]

    try:
        (
            active_shards,
            data_nodes,
            max_shards_per_node,
            heap_max_bytes_total,
        ) = await _fetch_cluster_facts(opensearch)
    except Exception:
        _cache = (now, None)
        return None

    result = _build_capacity(
        active_shards=active_shards,
        data_nodes=data_nodes,
        max_shards_per_node=max_shards_per_node,
        heap_max_bytes_total=heap_max_bytes_total,
        extra_shards=extra_shards,
    )
    _cache = (now, result)
    return result
