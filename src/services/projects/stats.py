"""``GET {prefix}/stats`` (§4, scoped): per-project counts, index sizes
and disk usage, cached 5 min per project (disk usage is the expensive
part -- a full ``rglob`` walk -- so it is computed in a thread and
reused for that window)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from src.clients.curation_opensearch.registry import ClassRegistry
from src.config.project_context import current_project
from src.core.logging import get_logger
from src.services.curation.embedding_state import embedded_clause
from src.services.training.model_classes import owned_model_names


logger = get_logger(__name__)


_DISK_CACHE_TTL_SECONDS = 300.0
_disk_cache: dict[str, tuple[float, dict[str, Any]]] = {}


async def _term_count(client: Any, index: str, field: str, value: Any) -> int:
    try:
        resp = await client.count(index=index, body={'query': {'term': {field: value}}})
        return int(resp.get('count') or 0)
    except Exception:
        return 0


# "Validated" everywhere a project's counts are served: a human-confirmed
# class (the same filter the pipeline-health rollup counts).
VALIDATED_ITEMS_QUERY: dict[str, Any] = {'term': {'class_validated': True}}


async def _query_count(
    client: Any, items_index: str, query: dict[str, Any], what: str
) -> int | None:
    try:
        resp = await client.count(index=items_index, body={'query': query})
        return int(resp.get('count') or 0)
    except Exception as exc:
        logger.warning('project_count_unavailable', what=what, index=items_index, error=str(exc))
        return None


async def validated_count(client: Any, items_index: str) -> int | None:
    """Validated items in ``items_index``, or ``None`` when it could not
    be counted -- never a made-up 0. Caller binds the owning project."""
    return await _query_count(client, items_index, VALIDATED_ITEMS_QUERY, 'validated')


async def embedded_count(client: Any, items_index: str) -> int | None:
    """Items with a vector (the working set), or ``None`` when uncountable."""
    return await _query_count(client, items_index, embedded_clause(), 'embedded')


async def index_count(client: Any, index: str) -> int:
    try:
        resp = await client.count(index=index)
        return int(resp.get('count') or 0)
    except Exception:
        return 0


def _dir_bytes(path: Any) -> int:
    try:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    except OSError:
        return 0


def _disk_usage(slug: str, resources: Any) -> dict[str, Any]:
    now = time.monotonic()
    cached = _disk_cache.get(slug)
    if cached is not None and now - cached[0] < _DISK_CACHE_TTL_SECONDS:
        return cached[1]
    result = {
        'exports_bytes': _dir_bytes(resources.export_root),
        'uploads_bytes': _dir_bytes(resources.upload_root),
        'runs_bytes': _dir_bytes(resources.train_jobs_dir),
        'computed_at': datetime.now(UTC).isoformat(),
    }
    _disk_cache[slug] = (now, result)
    return result


async def project_stats(client: Any) -> dict[str, Any]:
    """Must run inside a bound project (the route is scoped)."""
    from src.config.curation import images_index, items_index

    bound = current_project()
    items_idx = items_index()
    images_idx = images_index()

    images_count = await index_count(client, images_idx)
    items_count = await index_count(client, items_idx)
    validated = await validated_count(client, items_idx)
    items_embedded = await embedded_count(client, items_idx)
    from src.config.region_state import RegionStatus

    pending_detection = await _term_count(
        client, items_idx, 'region_status', RegionStatus.PENDING_DETECTION.value
    )
    holdout_items = await _term_count(client, items_idx, 'test_holdout', True)
    classes_count = len(
        ClassRegistry(path=bound.record.resources.class_registry_path).load().classes
    )

    indexes = []
    for role, name in bound.record.resources.indexes.items():
        indexes.append({'role': role.value, 'name': name})

    from src.services.projects.lifecycle import running_jobs as _running_jobs

    jobs = await _running_jobs(bound.record)

    return {
        'counts': {
            'images': images_count,
            'items': items_count,
            'validated': validated,
            'items_embedded': items_embedded,
            'pending_detection': pending_detection,
            'holdout_items': holdout_items,
            'classes': classes_count,
            'promoted_models': len(owned_model_names(bound.record.slug)),
        },
        'indexes': indexes,
        'disk': _disk_usage(bound.record.slug, bound.record.resources),
        'jobs': {'running': [j.to_wire() for j in jobs]},
        'last_ingest_at': None,
    }
