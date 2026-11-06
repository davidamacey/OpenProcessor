"""The file areas of a 0.6.0 install: which directories hold control data, and which files in
them the bundle keeps.

An :class:`Area` is a directory plus glob patterns relative to it. Per-project directories come
from each project record's stored ``resources`` (absolute paths, computed at create time in
0.6.0); the shared job directories and the model repository come from the same environment
variables 0.6.0 read.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from src.config.env import env_path
from src.services.upgrade.from_060 import sources as legacy


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

# A file under these globs (import ledgers, ingest queue databases, label-operation member lists)
# is streamed by the load instead of held in memory; a queue database is read as an SQLite image.
BULK_PATTERNS = ('*/ledger/*.jsonl', '*queue.sqlite', 'label_ops/*/members.json')


@dataclass(frozen=True)
class Area:
    """``label`` names the area in the bundle; ``root`` is the directory; ``include`` are globs
    (``fnmatch`` on the posix path relative to ``root``, ``*`` crossing ``/``)."""

    label: str
    root: Path
    include: tuple[str, ...] = ('*',)
    slug: str | None = None
    family: str = ''

    def files(self) -> Iterator[tuple[str, Path]]:
        """``(relative posix path, path)`` of every kept file, sorted."""
        if not self.root.is_dir():
            return
        found: list[tuple[str, Path]] = []
        for path in self.root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(self.root).as_posix()
            if any(fnmatch.fnmatchcase(rel, pattern) for pattern in self.include):
                found.append((rel, path))
        yield from sorted(found)

    def is_bulk(self, rel: str) -> bool:
        return any(fnmatch.fnmatchcase(rel, pattern) for pattern in BULK_PATTERNS)

    def target(self, rel: str) -> Path:
        """Where a restored ``rel`` goes; refuses a path that leaves the area."""
        pure = PurePosixPath(rel)
        if pure.is_absolute() or '..' in pure.parts:
            msg = f'{self.label}: unsafe relative path {rel!r}'
            raise ValueError(msg)
        return self.root.joinpath(*pure.parts)


@dataclass(frozen=True)
class Layout:
    """The shared directories of a 0.6.0 install."""

    job_family_dirs: Mapping[str, Path]
    combine_dir: Path
    model_repo: Path
    projects_index: str = legacy.PROJECTS_INDEX
    global_configs_index: str = legacy.GLOBAL_CONFIGS_INDEX
    extra_areas: Sequence[Area] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> Layout:
        """The directories 0.6.0 resolved from its environment variables."""
        return cls(
            job_family_dirs={
                name: env_path(var, default) for name, var, default in legacy.JOB_FAMILIES
            },
            combine_dir=env_path(*legacy.COMBINE_ENV),
            model_repo=env_path(*legacy.MODEL_REPO_ENV),
        )

    def areas(self, records: Sequence[Mapping[str, Any]]) -> list[Area]:
        """Every area for the given project records (the ``op_projects`` document sources)."""
        areas: list[Area] = []
        for record in sorted(records, key=lambda r: r['slug']):
            areas.extend(self._project_areas(record))
        areas.append(Area('combine', self.combine_dir))
        areas.append(
            Area('models', self.model_repo, ('*/promote.json', '*/class_remap.json')),
        )
        areas.extend(self.extra_areas)
        return areas

    def _project_areas(self, record: Mapping[str, Any]) -> list[Area]:
        slug = str(record['slug'])
        res = record['resources']
        out = [
            Area(
                f'class_registry:{slug}',
                Path(res['class_registry_path']).parent,
                (legacy.CLASS_REGISTRY_FILE, legacy.CLASS_REGISTRY_FILE + '.migrated'),
                slug,
            ),
            Area(
                f'state:{slug}',
                Path(res['project_state_dir']),
                (
                    legacy.PAUSE_FLAG,
                    f'{legacy.HOLDOUT_DIR}/*',
                    f'{legacy.LABEL_OPS_DIR}/*',
                    legacy.LAST_CLUSTERING_FILE,
                ),
                slug,
            ),
            Area(f'exports:{slug}', Path(res['export_root']), ('*/manifest.json',), slug),
            Area(
                f'train:{slug}',
                Path(res['train_jobs_dir']),
                (
                    '*.job.json',
                    '*.status.json',
                    '*.manifest.json',
                    '*.registry_snapshot.json',
                    '*.run.log',
                    '*.cancel',
                ),
                slug,
                'training',
            ),
            Area(f'autolabel:{slug}', Path(res['autolabel_dir']), ('*',), slug, 'autolabel'),
        ]
        out.extend(
            Area(f'{family}:{slug}', base / 'projects' / slug, ('*',), slug, family)
            for family, base in self.job_family_dirs.items()
        )
        return out
