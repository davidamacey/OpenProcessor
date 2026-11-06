"""Every job kind, aggregated from one module per lane (so lanes never edit the same file)."""

from __future__ import annotations

from src.config.job_kinds import (
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
    promote,
    region_text,
    source_backfill,
    trigger,
)
from src.config.job_kinds.kind import JobKind


_MODULES = (
    infra,
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


def _collect() -> dict[str, JobKind]:
    kinds: dict[str, JobKind] = {}
    names: set[str] = set()
    for module in _MODULES:
        for kind in module.KINDS:
            if kind.name in kinds:
                raise ValueError(f'job kind {kind.name!r} declared twice')
            if kind.task_name in names:
                raise ValueError(f'task name {kind.task_name!r} used by two job kinds')
            kinds[kind.name] = kind
            names.add(kind.task_name)
    return kinds


KINDS: dict[str, JobKind] = _collect()
KIND_BY_TASK: dict[str, JobKind] = {kind.task_name: kind for kind in KINDS.values()}

__all__ = ['KINDS', 'KIND_BY_TASK', 'JobKind']
