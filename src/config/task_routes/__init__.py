"""The one task-name -> queue table, aggregated from one module per lane.

A decorator never passes ``queue=``: Celery merges the route under the task's own options,
so a decorator queue would silently override this table.
"""

from __future__ import annotations

from src.config.task_routes import (
    b2_embedding,
    b4_fanout,
    b4_project,
    clustering,
    families_a,
    families_b,
    families_c,
    gpu_eval,
    gpu_export,
    gpu_train,
    infra,
    near_dup,
    near_dup_core,
    periodic,
    promote,
    region_text,
    source_backfill,
    trigger,
)


_MODULES = (
    infra,
    periodic,
    families_a,
    families_b,
    families_c,
    trigger,
    gpu_train,
    gpu_export,
    gpu_eval,
    promote,
    b4_fanout,
    b4_project,
    clustering,
    b2_embedding,
    near_dup,
    near_dup_core,
    source_backfill,
    region_text,
)


def _collect() -> dict[str, str]:
    routes: dict[str, str] = {}
    for module in _MODULES:
        for task_name, queue in module.ROUTES.items():
            if task_name in routes:
                raise ValueError(f'task {task_name!r} routed twice')
            routes[task_name] = queue
    return routes


ROUTES: dict[str, str] = _collect()


def celery_task_routes() -> dict[str, dict[str, str]]:
    """The ``task_routes`` setting."""
    return {name: {'queue': queue} for name, queue in ROUTES.items()}


__all__ = ['ROUTES', 'celery_task_routes']
