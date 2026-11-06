"""A bound project on a migrated database, a storage root in ``tmp_path`` and a fake registry.

The training services read the bound project, the class registry and the export manifest from
places a database test does not stand up; each is replaced here by the smallest fake that keeps
the code under test on its real path (the database, the job service and the queue seam).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fixtures.job_fixtures import FakeProducer, job_env, kinds, producer, queue_hooks, wakes
from fixtures.job_rows import make_project, sql

from src.clients.storage.roots import RootedStore
from src.config import project_context
from src.services.training import job_manifest, jobs, lineage


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ['FakeProducer', 'job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']

REGISTRY_CLASSES = [(0, 'car'), (1, 'bus'), (2, 'person'), (3, 'bike')]


class _Registry:
    classes = [
        SimpleNamespace(class_id=i, class_name=n, deprecated=False) for i, n in REGISTRY_CLASSES
    ]

    def model_dump_json(self, indent: int = 0) -> str:
        del indent
        return '{"classes": []}'


@pytest.fixture
def run_env(pg, job_env, producer, wakes, monkeypatch, tmp_path: Path) -> Iterator[SimpleNamespace]:
    project = make_project(pg)
    [(slug,)] = sql(pg, 'SELECT slug FROM projects WHERE id = %s', project)
    store = RootedStore(
        {
            'projects': tmp_path / 'projects',
            'source': tmp_path,
            'models': tmp_path,
            'cache': tmp_path,
            'upgrade': tmp_path,
        }
    )
    monkeypatch.setattr(jobs, 'run_store', lambda: store)
    monkeypatch.setattr(job_manifest, 'run_store', lambda: store)
    monkeypatch.setattr(
        jobs,
        'get_curation_config',
        lambda: SimpleNamespace(
            project_slug=slug, export_root=tmp_path / 'exports', mlflow_experiment=slug
        ),
    )

    async def registry() -> _Registry:
        return _Registry()

    monkeypatch.setattr('src.services.curation.registry_access.current_registry', registry)
    monkeypatch.setattr(
        lineage,
        'read_export_identity',
        lambda _dir: lineage.ExportIdentity('ds-sha', 'frozen-sha', 'label-sha', 'v1'),
    )
    record = SimpleNamespace(id=project, slug=slug)
    with project_context.bind_project(record):  # type: ignore[arg-type]
        yield SimpleNamespace(pg=pg, project=project, slug=slug, store=store, producer=producer)
