"""P3F item 6 m1: the delete path guard for ``train_jobs_dir``
used to be checked against a root DERIVED FROM the
same path (``path.parent.parent``), which can never refuse anything --
any path is trivially "within" its own grandparent. A corrupted or
hand-edited registry doc pointing the field outside the real shared
root must now be refused with ``path_escape``, not silently followed.

P3F pass-3 m-a: the prior pass's fix covered only these 2 of the
project's 8 dirs -- the other 6 still went through the old
``_rm_dir_guarded``/``_path_within``, which accepted ``path ==
shared_root`` itself. A corrupted resources record pointing one of
those 6 fields straight at the shared multi-project root would have
wiped every sibling project's directory tree (the review's own probe
did exactly this). The tests below reproduce that for each of the 6,
and also prove the path-escape check now runs BEFORE the irreversible
index delete (previously it ran only inside dir removal, itself after
indexes were already gone).
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import replace

import pytest

from src.core.exceptions import RequestRejectedError
from src.services.projects import delete as delete_mod, lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry()
    set_project_registry(registry)
    return registry


def _corrupt_stored(monkeypatch, corrupted) -> None:
    """Directories derive from the slug and the deployment roots, so a corrupted record can only
    come from a misread row: make the delete's read hand back this one for its project."""
    real = delete_mod.record_from_row

    def read(row):
        record = real(row)
        if row.slug != corrupted.slug:
            return record
        return dataclasses.replace(record, resources=corrupted.resources)

    monkeypatch.setattr(delete_mod, 'record_from_row', read)


def test_delete_refuses_path_escape_for_corrupted_train_jobs_dir(tmp_path, monkeypatch) -> None:
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(registry.ensure_fresh())

    record = registry.get('alpha')
    assert record is not None
    escaped_dir = tmp_path / 'escaped_outside_root'
    escaped_dir.mkdir(parents=True)
    (escaped_dir / 'canary.txt').write_text('do not delete me', encoding='utf-8')
    corrupted = replace(record, resources=replace(record.resources, train_jobs_dir=escaped_dir))
    _corrupt_stored(monkeypatch, corrupted)

    deleting = asyncio.run(lifecycle.delete_project(slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    with pytest.raises(RequestRejectedError) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'path_escape'
    assert exc_info.value.status_code == 500
    assert escaped_dir.is_dir(), 'the escaped directory must survive, untouched'
    assert (escaped_dir / 'canary.txt').is_file()


_PREVIOUSLY_UNGUARDED_FIELDS = (
    'export_root',
    'class_registry_path',
    'bakeoff_eval_root',
    'project_state_dir',
    'upload_root',
    'bakeoff_jobs_dir',
)


def _shared_root_for(field_name: str):
    from src.config.curation_accessors import base_curation_config
    from src.config.projects import projects_data_root

    if field_name in ('export_root', 'class_registry_path', 'bakeoff_eval_root'):
        return projects_data_root()
    return base_curation_config().state_dir / 'projects'


@pytest.mark.parametrize('field_name', _PREVIOUSLY_UNGUARDED_FIELDS)
def test_delete_refuses_path_escape_when_dir_points_at_the_shared_root_itself(
    field_name: str,
    monkeypatch,
) -> None:
    """m-a: reproduce the reviewer's exact probe for each of the 6 dirs
    the prior pass left unguarded -- point the field straight at the
    shared multi-project root (never a per-project subdir) and confirm
    a sibling project's own directory under that root survives
    untouched, first showing that it WOULD be destroyed pre-fix (this
    docstring's claim is verified by reverting the fix per the task's
    red-first requirement, not re-asserted at runtime here)."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(lifecycle.create_project(client, slug='beta', display_name='Beta'))
    asyncio.run(registry.ensure_fresh())

    record = registry.get('alpha')
    beta = registry.get('beta')
    assert record is not None
    assert beta is not None

    shared_root = _shared_root_for(field_name)
    shared_root.mkdir(parents=True, exist_ok=True)

    beta_path = getattr(beta.resources, field_name)
    if field_name == 'class_registry_path':
        beta_path = beta_path.parent
    beta_path.mkdir(parents=True, exist_ok=True)
    (beta_path / 'canary.txt').write_text('do not delete me', encoding='utf-8')

    corrupted_value = (
        shared_root / 'class_registry.json' if field_name == 'class_registry_path' else shared_root
    )
    corrupted = dataclasses.replace(
        record,
        resources=dataclasses.replace(record.resources, **{field_name: corrupted_value}),
    )
    _corrupt_stored(monkeypatch, corrupted)

    deleting = asyncio.run(lifecycle.delete_project(slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    with pytest.raises(RequestRejectedError) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'path_escape'
    assert exc_info.value.status_code == 500

    assert (beta_path / 'canary.txt').is_file(), (
        f"a sibling project's dir must survive a corrupted '{field_name}' escape"
    )


def test_path_escape_check_runs_before_the_irreversible_index_delete(monkeypatch) -> None:
    """m-a ordering: a path_escape must be raised BEFORE
    ``_delete_indexes`` runs -- previously the path check lived only
    inside dir removal, itself called AFTER index deletion, so a
    path_escape left the record wedged 'deleting' with the indexes
    already irreversibly gone (every retry hits the same escape
    again)."""
    client = FakeLifecycleOpenSearch()
    registry = _registry_for(client)
    asyncio.run(seed_default_project(client))
    asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    asyncio.run(registry.ensure_fresh())

    record = registry.get('alpha')
    assert record is not None
    escaped_dir = _shared_root_for('export_root')
    corrupted = dataclasses.replace(
        record, resources=dataclasses.replace(record.resources, export_root=escaped_dir)
    )
    _corrupt_stored(monkeypatch, corrupted)

    deleting = asyncio.run(lifecycle.delete_project(slug='alpha', confirm='alpha'))
    assert deleting.status == 'deleting'

    with pytest.raises(RequestRejectedError) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'path_escape'

    assert client.deleted_indexes == [], (
        'path validation must run before any index is irreversibly deleted'
    )
