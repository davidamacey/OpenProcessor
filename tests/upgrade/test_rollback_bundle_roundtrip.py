"""2E-4: bundle -> delete the sources -> rollback -> the recreated documents and files equal the fixture."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
from typing import TYPE_CHECKING

import pytest

from src.services.upgrade.from_060.delete_sources import delete_sources
from src.services.upgrade.from_060.export import export_bundle
from src.services.upgrade.from_060.rollback import RollbackError, rollback


if TYPE_CHECKING:
    from tests.upgrade._v060 import V060Install


NOW = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)
CONTROL = ('op_projects', 'op_global_configs')


def _files(v060: V060Install) -> dict[str, str]:
    return {
        str(p.relative_to(v060.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in v060.root.rglob('*')
        if p.is_file() and 'storage' not in p.relative_to(v060.root).parts
    }


def _control_docs(v060: V060Install) -> dict[str, dict[str, object]]:
    return {
        name: copy.deepcopy(v060.os.docs(name))
        for name in v060.os.indexes
        if name.startswith(CONTROL)
        or name.rsplit('__', 1)[-1]
        in {'configs', 'settings', 'classes', 'umap_state', 'umap_viz_state', 'labels_confirmed'}
    }


@pytest.mark.asyncio
async def test_rollback_restores_the_fixture_after_the_sources_are_deleted(v060, tmp_path) -> None:
    bundle = tmp_path / 'bundle'
    await export_bundle(v060.reader, v060.layout, v060.store, bundle, now=NOW)
    files_before = _files(v060)
    # the transient documents (leases, runtime reports) are not exported or restored
    docs_before = {
        name: {k: v for k, v in docs.items() if not k.startswith(('lease:', 'runtime:'))}
        for name, docs in _control_docs(v060).items()
    }
    deleted = await delete_sources(bundle, v060.writer, v060.layout)
    assert 'op_projects' in deleted.indexes_deleted
    assert not _control_docs(v060)
    assert (v060.path('data/projects/default/class_registry.json.migrated-0.7.0')).is_file()
    assert not v060.path('data/projects/default/class_registry.json').exists()

    report = await rollback(bundle, v060.reader, v060.writer, v060.layout, v060.store)

    assert report.ok, report.problems
    assert _control_docs(v060) == docs_before
    assert _files(v060) == files_before  # including the run log, restored from its storage key
    assert report.registries_renamed_back == ['default']
    assert not v060.path('data/projects/default/class_registry.json.migrated-0.7.0').exists()
    # the kept indexes were never touched
    assert 'op_prj_demo__items' in v060.os.indexes


@pytest.mark.asyncio
async def test_rollback_is_idempotent(v060, tmp_path) -> None:
    bundle = tmp_path / 'bundle'
    await export_bundle(v060.reader, v060.layout, v060.store, bundle, now=NOW)
    await delete_sources(bundle, v060.writer, v060.layout)
    await rollback(bundle, v060.reader, v060.writer, v060.layout, v060.store)
    again = await rollback(bundle, v060.reader, v060.writer, v060.layout, v060.store)
    assert again.ok
    assert again.files_restored == 0


@pytest.mark.asyncio
async def test_a_foreign_index_of_the_same_name_is_refused_not_overwritten(v060, tmp_path) -> None:
    bundle = tmp_path / 'bundle'
    await export_bundle(v060.reader, v060.layout, v060.store, bundle, now=NOW)
    await delete_sources(bundle, v060.writer, v060.layout)
    await v060.writer.create_index('op_projects', {'properties': {}}, {'index': {}})
    await v060.writer.bulk_index('op_projects', [('project:other', {'slug': 'other'})])
    with pytest.raises(RollbackError, match='op_projects'):
        await rollback(bundle, v060.reader, v060.writer, v060.layout, v060.store)
