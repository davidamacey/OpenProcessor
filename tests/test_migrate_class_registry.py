"""``scripts/curation/migrate_class_registry.py``: class_registry.json <-> registry document.

The real registry store runs against an in-memory document store that enforces
``op_type=create`` and ``if_seq_no``; only OpenSearch is faked.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest
from _registry_doc_fake import FakeDocOpenSearch

from scripts.curation import migrate_class_registry as migrate
from src.clients.curation_opensearch import registry as registry_module
from src.clients.curation_opensearch.registry import get_class_registry
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, new_project_record


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _real_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, '_registries', {})


def _record(tmp_path: Path, slug: str = 'demo') -> ProjectRecord:
    record = new_project_record(slug, base_curation_config())
    res = dataclasses.replace(
        record.resources,
        class_registry_path=tmp_path / slug / 'class_registry.json',
        project_state_dir=tmp_path / 'state' / slug,
    )
    return dataclasses.replace(record, resources=res)


# ids with a gap, a deprecated class and a hotkey: what a long-lived 0.6.0 project looks like
LEGACY_FILE = {
    'version': 1,
    'updated_at': '2026-01-01T00:00:00+00:00',
    'classes': [
        {'id': 0, 'name': 'car', 'group': 'vehicle', 'hotkey_letter': 'c'},
        {'id': 2, 'name': 'truck', 'group': 'vehicle'},
        {'id': 5, 'name': 'old_van', 'deprecated': True, 'merged_into': 2},
    ],
}


def _write_file(record: ProjectRecord, payload: dict[str, Any] = LEGACY_FILE) -> Path:
    path = record.resources.class_registry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


async def _snapshot(record: ProjectRecord, client: Any) -> Any:
    with bind_project(record):
        return await get_class_registry().snapshot(client)


async def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)

    report = await migrate.migrate_project(client, record, dry_run=True, reverse=False)

    assert report['action'] == 'seed'
    assert report['applied'] is False
    assert path.exists()
    assert client.docs == {}


async def test_migration_preserves_every_id_and_renames_the_file(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'seed'
    assert report['applied'] is True
    assert not path.exists()
    assert path.with_name('class_registry.json.migrated').exists()
    doc = await _snapshot(record, client)
    assert [(c.class_id, c.class_name, c.deprecated, c.merged_into) for c in doc.classes] == [
        (0, 'car', False, None),
        (2, 'truck', False, None),
        (5, 'old_van', True, 2),
    ]
    assert doc.classes[0].hotkey_letter == 'c'
    assert doc.next_class_id == 6
    assert doc.migrated_from_file_at is not None
    with bind_project(record):
        new_id = await get_class_registry().add_class(client, 'bus')
    assert new_id == 6


async def test_a_rerun_is_a_no_op(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    _write_file(record)
    await migrate.migrate_project(client, record, dry_run=False, reverse=False)
    before = json.dumps(client.docs, sort_keys=True)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'nothing_to_do'
    assert json.dumps(client.docs, sort_keys=True) == before


async def test_an_equal_document_only_renames_the_file(tmp_path: Path) -> None:
    """The first run created the document and died before the rename."""
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)
    await migrate.migrate_project(client, record, dry_run=False, reverse=False)
    path.with_name('class_registry.json.migrated').replace(path)
    before = json.dumps(client.docs, sort_keys=True)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'rename_only'
    assert not path.exists()
    assert json.dumps(client.docs, sort_keys=True) == before


async def test_a_conflicting_document_is_an_error_and_writes_nothing(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)
    await migrate.migrate_project(client, record, dry_run=False, reverse=False)
    with bind_project(record):
        await get_class_registry().add_class(client, 'bus')  # the document moves on
    path.write_text(json.dumps(LEGACY_FILE), encoding='utf-8')  # a stale file shows up again
    before = json.dumps(client.docs, sort_keys=True)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'conflict'
    assert 'differ' in report['error']
    assert path.exists()
    assert json.dumps(client.docs, sort_keys=True) == before


async def test_a_project_with_no_file_has_nothing_to_do(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'nothing_to_do'
    assert client.docs == {}


async def test_reverse_writes_the_current_classes_back_to_the_file(tmp_path: Path) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)
    await migrate.migrate_project(client, record, dry_run=False, reverse=False)
    with bind_project(record):
        await get_class_registry().add_class(client, 'bus')  # added after the migration

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=True)

    assert report['action'] == 'reverse'
    on_disk = json.loads(path.read_text(encoding='utf-8'))
    assert [c['class_id'] for c in on_disk['classes']] == [0, 2, 5, 6]
    assert on_disk['classes'][3]['class_name'] == 'bus'
    assert any(p.name.startswith('class_registry.json.migrated.') for p in path.parent.iterdir())
    doc = await _snapshot(record, client)
    assert doc.reversed_to_file_at is not None
    assert len(doc.classes) == 4


async def test_forward_after_reverse_reseeds_from_the_file_with_ids_preserved(
    tmp_path: Path,
) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    path = _write_file(record)
    await migrate.migrate_project(client, record, dry_run=False, reverse=False)
    await migrate.migrate_project(client, record, dry_run=False, reverse=True)
    # the older version ran for a while and added a class to the file
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['classes'].append({'class_id': 7, 'class_name': 'tractor'})
    path.write_text(json.dumps(payload), encoding='utf-8')

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'reseed_after_reverse'
    assert not path.exists()
    doc = await _snapshot(record, client)
    assert [c.class_id for c in doc.classes] == [0, 2, 5, 7]
    assert doc.next_class_id == 8
    assert doc.reversed_to_file_at is None


async def test_the_untouched_empty_document_of_a_new_project_is_replaced_by_a_copied_in_file(
    tmp_path: Path,
) -> None:
    """The worked example: a project is created (empty document), the example file is copied in."""
    record, client = _record(tmp_path), FakeDocOpenSearch()
    await _snapshot(record, client)  # first use creates the empty document
    path = _write_file(record)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'reseed_empty'
    assert not path.exists()
    assert [c.class_id for c in (await _snapshot(record, client)).classes] == [0, 2, 5]


async def test_a_document_that_has_been_written_to_is_never_replaced_by_a_copied_in_file(
    tmp_path: Path,
) -> None:
    record, client = _record(tmp_path), FakeDocOpenSearch()
    with bind_project(record):
        await get_class_registry().add_class(client, 'mine')
        await get_class_registry().set_deprecated(client, 0, True)  # emptied by use, not untouched
    _write_file(record)

    report = await migrate.migrate_project(client, record, dry_run=False, reverse=False)

    assert report['action'] == 'conflict'
