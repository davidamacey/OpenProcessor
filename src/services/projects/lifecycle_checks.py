"""Lifecycle preconditions: the running-job inventory that blocks archive and
delete, the last-active-project rule, and the status and revision checks.

Split out of ``lifecycle.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.projects.registry import get_project_registry
from src.services.rejections import rejection


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


@dataclass(frozen=True)
class JobRef:
    """A running job blocking a lifecycle action (delta 11)."""

    kind: str
    kind_label: str
    id: str
    label: str
    started_at: str | None

    def to_wire(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'kind_label': self.kind_label,
            'id': self.id,
            'label': self.label,
            'started_at': self.started_at,
        }


_KIND_LABELS = {
    'train': 'Training run',
    'bakeoff': 'Bake-off',
    'autolabel': 'Auto-label',
    'export': 'Export',
    'detection_worker': 'Detection worker',
}


async def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """The project's real running-job inventory, via
    ``src.services.projects.busy.running_jobs`` -- the single source of
    truth for "is this project busy" across every job-producing
    subsystem (§5.4). Delete/archive never re-implement their own file
    scan; they only adapt ``busy.JobRef`` (``kind``/``job_id``) to this
    module's wire-shaped ``JobRef`` (delta 11).

    P3F pass-3 m-b: ``label`` is the job source's own genuine human
    label (``busy.JobRef.label``, e.g. a train run's submitted
    ``mlflow_run_name``) when the source recorded one, falling back to
    the internal ``job_id`` ONLY when it didn't -- documented here
    rather than silently treating ``job_id`` as if it were always a
    human-meaningful name."""
    from src.services.projects import busy

    return [
        JobRef(
            kind=j.kind,
            kind_label=_KIND_LABELS.get(j.kind, j.kind),
            id=j.job_id,
            label=j.label or j.job_id,
            started_at=j.started_at,
        )
        for j in busy.running_jobs(record)
    ]


def _other_active_slugs(snapshot: dict[str, ProjectRecord], slug: str) -> list[str]:
    """Every *other* project that is ``active``. Archived, building,
    failed, deleting and deleted projects do not count: none of them can
    take writes. The one rule both the delete dry run and the real
    archive/delete guards use, so the dry run never disagrees."""
    return [s for s, r in snapshot.items() if s != slug and r.status == 'active']


_LAST_ACTIVE_MESSAGE = 'this is the only active project; leave at least one'


async def _last_active_check(record: ProjectRecord) -> None:
    """Refuse an archive or delete that would leave no active project.

    P3F m4 (documented, not fixed): this reads the registry snapshot and
    the caller's own OCC write happens later, so two concurrent
    archives/deletes of the last two active projects can both pass this
    check and both succeed -- the same class of race as two concurrent
    creates racing the capacity check. A real fix needs a single
    serialization point (a lock doc with its own OCC, or a distributed
    lock) this codebase has no infra for yet; every other OCC guard here
    protects one document's own read-modify-write, not an invariant
    spanning every document in the registry. Left as documented
    best-effort until that infra exists."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    if not _other_active_slugs(dict(registry.snapshot()), record.slug):
        raise rejection(409, 'last_active_project', _LAST_ACTIVE_MESSAGE, project=record.slug)


def _require_transition(record: ProjectRecord, action: str, allowed: frozenset[str]) -> None:
    if record.status not in allowed:
        raise rejection(
            409,
            'invalid_transition',
            f"cannot {action} '{record.slug}' while it is {record.status}",
            project=record.slug,
            project_status=record.status,
            action=action,
        )


def _require_revision(record: ProjectRecord, expected_revision: int) -> None:
    if record.revision != expected_revision:
        raise rejection(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=record.slug,
            current_revision=record.revision,
        )
