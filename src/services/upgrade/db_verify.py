"""``openprocessor db verify``: the database against OpenSearch and storage. Report only.

* every ``project_indexes.index_name`` exists in OpenSearch;
* the D-47 class reconciliation over each project's items (the same code the upgrade's verify
  step runs): items whose ``class_id`` is not in the registry, or whose ``class_name`` disagrees;
* the storage keys that promotions (``models.artifact_storage_key``) and training runs
  (``training_runs.artifact_prefix``) point at exist.

Restoring a dump to an older point than the OpenSearch data can name a class id the registry does
not have; this report is how that shows (the docs say to restore both from the same time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.db.models import Model, Project, ProjectClass, ProjectIndex, TrainingRun
from src.services.upgrade.class_reconcile import reconcile_classes


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.storage.base import BlobStore
    from src.services.upgrade.from_060.admin_client import ControlReader


@dataclass
class DbVerifyReport:
    missing_indexes: list[dict[str, str]] = field(default_factory=list)
    class_drift: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_storage: list[dict[str, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.missing_indexes
            or self.missing_storage
            or any(d['unknown_class_id'] or d['name_disagrees'] for d in self.class_drift.values())
        )

    def to_json(self) -> dict[str, Any]:
        return {
            'clean': self.clean,
            'missing_indexes': self.missing_indexes,
            'class_drift': self.class_drift,
            'missing_storage': self.missing_storage,
        }


def _exists(store: BlobStore, key: str) -> bool:
    try:
        return bool(store.exists(key) or store.list_prefix(key))
    except (OSError, ValueError):
        return False


async def verify_database(
    session: AsyncSession, reader: ControlReader, store: BlobStore
) -> DbVerifyReport:
    report = DbVerifyReport()
    projects: dict[uuid.UUID, str] = dict(
        (await session.execute(select(Project.id, Project.slug))).tuples().all()
    )
    indexes: dict[str, dict[str, str]] = {}
    for pid, role, name in await session.execute(
        select(ProjectIndex.project_id, ProjectIndex.role, ProjectIndex.index_name)
    ):
        indexes.setdefault(projects[pid], {})[role] = name
    registries: dict[str, dict[int, str]] = {slug: {} for slug in projects.values()}
    for pid, class_id, name in await session.execute(
        select(ProjectClass.project_id, ProjectClass.class_id, ProjectClass.name)
    ):
        registries[projects[pid]][class_id] = name
    for slug in sorted(indexes):
        missing = False
        for role, name in sorted(indexes[slug].items()):
            if not await reader.index_exists(name):
                report.missing_indexes.append({'project': slug, 'role': role, 'index': name})
                missing = True
        if not missing and 'items' in indexes[slug]:
            drift = await reconcile_classes(reader, indexes[slug]['items'], registries[slug])
            report.class_drift[slug] = drift.to_json()
    for model_project, key in await session.execute(
        select(Model.project_id, Model.artifact_storage_key)
    ):
        if model_project is not None and not _exists(store, key):
            report.missing_storage.append(
                {'project': projects[model_project], 'kind': 'model', 'key': key}
            )
    for pid, key in await session.execute(
        select(TrainingRun.project_id, TrainingRun.artifact_prefix)
    ):
        if not _exists(store, key):
            report.missing_storage.append(
                {'project': projects[pid], 'kind': 'training_run', 'key': key}
            )
    return report
