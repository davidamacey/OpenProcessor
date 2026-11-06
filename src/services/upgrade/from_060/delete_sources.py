"""Step 4, ``delete_sources``: remove what the load moved, once verify has passed.

Every deletion tolerates "already gone", so a step that stopped half way runs again to the end.
The sources are the control indexes of OpenSearch, the class-registry files (renamed, never
deleted) and the state and job files the bundle holds. Export directories and the served-model
files of the Triton repository are artifacts, not control data, and stay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.bundle import iter_records


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.upgrade.from_060.admin_client import ControlWriter
    from src.services.upgrade.from_060.layout import Area, Layout

# Areas whose files are artifacts or serving state: never deleted by the upgrade.
KEPT_AREA_PREFIXES = ('exports:', 'models')


@dataclass
class DeleteReport:
    indexes_deleted: list[str] = field(default_factory=list)
    registries_renamed: list[str] = field(default_factory=list)
    files_deleted: int = 0
    directories_removed: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            'indexes_deleted': self.indexes_deleted,
            'registries_renamed': self.registries_renamed,
            'files_deleted': self.files_deleted,
            'directories_removed': self.directories_removed,
        }


def _area_index(layout: Layout, records: list[dict[str, object]]) -> dict[str, Area]:
    return {area.label: area for area in layout.areas(records)}


def _prune_empty(directory: Path, area_root: Path) -> int:
    """Remove ``directory`` and its parents while they are empty, up to and including the area's
    own root (the per-project job directory the files left empty), never above it."""
    removed = 0
    while (
        directory.is_dir() and directory.is_relative_to(area_root) and not any(directory.iterdir())
    ):
        directory.rmdir()
        removed += 1
        directory = directory.parent
    return removed


async def delete_sources(bundle: Path, writer: ControlWriter, layout: Layout) -> DeleteReport:
    """Delete the control indexes and the migrated files named by the bundle."""
    report = DeleteReport()
    project_records: list[dict[str, object]] = []
    indexes: list[str] = []
    files: list[tuple[str, str]] = []
    for record in iter_records(bundle):
        if record['t'] == 'index':
            indexes.append(record['name'])
        elif record['t'] == 'doc' and record['index'] == layout.projects_index:
            if str(record['id']).startswith(legacy.PROJECT_DOC_PREFIX):
                project_records.append(record['source'])
        elif record['t'] == 'file':
            files.append((record['area'], record['rel']))
        elif record['t'] == 'blob':
            files.append((record['source_area'], record['source_rel']))
    for name in indexes:
        if await writer.delete_index(name):
            report.indexes_deleted.append(name)
    areas = _area_index(layout, project_records)
    for label, rel in files:
        area = areas.get(label)
        if area is None or label.startswith(KEPT_AREA_PREFIXES):
            continue
        target = area.target(rel)
        if label.startswith('class_registry:'):
            if rel == legacy.CLASS_REGISTRY_FILE and target.is_file():
                target.replace(target.with_name(legacy.CLASS_REGISTRY_RENAMED))
                report.registries_renamed.append(label.partition(':')[2])
            continue
        if target.is_file():
            target.unlink()
            report.files_deleted += 1
        report.directories_removed += _prune_empty(target.parent, area.root)
    return report
