"""Clone, import, bulk ingest and reprocess directory isolation between projects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED
from curation.cross_project_leak._seeders import _stored_prompt_pack, _stored_region_profile


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


def test_from_project_clone_reads_source_index_only_under_the_real_guard(
    leak_env: LeakEnv,
) -> None:
    """Isolation gap (b): the leak sweep's fixed
    route table never sends `from_project`, so a `read_only=True`
    regression on either clone route's source bind was invisible to it
    (M1b in the review: dropping `read_only=True` left the whole 151-test
    pack/profile/leak/clone selection green). This drives `from_project`
    through the real app + real `ProjectGuardedTransport`-backed transport
    directly, and pins that the source project's `configs` index is only
    ever read, never written, while the new pack/profile lands only in
    the target.
    """
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    from src.config.curation import IndexRole

    _stored_prompt_pack(leak_env, 'alpha')
    _stored_region_profile(leak_env, 'alpha')
    source_configs_index = leak_env.records['alpha'].resources.indexes[IndexRole.CONFIGS]
    target_configs_index = leak_env.records['beta'].resources.indexes[IndexRole.CONFIGS]

    writes_before = list(leak_env.transport.writes)

    r = client.post(
        f'{SCOPED.format(project="beta")}/prompt_packs/alpha-model/clone',
        json={'new_name': 'from-alpha-pack', 'from_project': 'alpha'},
    )
    assert r.status_code == 201, r.text

    r2 = client.post(
        f'{SCOPED.format(project="beta")}/region_profiles/alpha-model/clone',
        json={'new_name': 'from-alpha-profile', 'from_project': 'alpha'},
    )
    assert r2.status_code == 201, r2.text

    new_writes = leak_env.transport.writes[len(writes_before) :]
    assert source_configs_index not in new_writes, (
        "from_project clone wrote to the SOURCE project's configs index -- "
        'the read_only bind on the source is not being honored'
    )
    assert target_configs_index in new_writes, (
        'from_project clone never wrote to the target -- the clone did not actually happen'
    )

    # The cloned docs must exist in the target, not the source.
    target_docs = leak_env.transport.store.get(target_configs_index, {})
    source_docs = leak_env.transport.store.get(source_configs_index, {})
    assert 'pack:from-alpha-pack' in target_docs
    assert 'pack:from-alpha-pack' not in source_docs
    assert 'profile:from-alpha-profile' in target_docs
    assert 'profile:from-alpha-profile' not in source_docs


def test_import_state_of_alpha_is_invisible_to_beta_and_a_misplaced_dir_is_refused(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Imports, uploads and reprocess jobs are filesystem state the route
    sweep cannot see leak. Each project's dir carries a ``.project`` marker."""
    import shutil

    from src.config.project_context import ProjectDirMismatchError, bind_project
    from src.services.curation.dataset_import.store import (
        ImportStore,
        imports_root,
        list_stores,
        open_store,
    )

    monkeypatch.setenv('OP_DATASET_IMPORTS_DIR', str(leak_env.root / 'imports'))
    import_id = 'imp_20260101T000000_0123abcd'
    with bind_project(leak_env.records['alpha']):
        ImportStore(imports_root() / import_id).job.write({'status': 'completed'})
        assert open_store(import_id) is not None
        alpha_dir = imports_root()
        assert (alpha_dir / '.project').read_text() == 'alpha'
    with bind_project(leak_env.records['beta']):
        assert open_store(import_id) is None
        assert list_stores() == []
        beta_dir = imports_root()
    assert beta_dir != alpha_dir
    # alpha's directory copied under beta's path: refused, not served.
    shutil.copytree(alpha_dir, beta_dir)
    with bind_project(leak_env.records['beta']), pytest.raises(ProjectDirMismatchError):
        open_store(import_id)


def test_bulk_ingest_runs_of_alpha_are_invisible_to_beta_and_a_misplaced_dir_is_refused(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from src.config.project_context import ProjectDirMismatchError, bind_project
    from src.services.curation.bulk_ingest.store import (
        BulkStore,
        list_stores,
        open_store,
        runs_root,
    )

    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(leak_env.root / 'bulk'))
    run_id = 'bulk_20260101T000000_0123abcd'
    with bind_project(leak_env.records['alpha']):
        BulkStore(runs_root() / run_id).job.write({'status': 'completed'})
        assert open_store(run_id) is not None
        alpha_dir = runs_root()
        assert (alpha_dir / '.project').read_text() == 'alpha'
    with bind_project(leak_env.records['beta']):
        assert open_store(run_id) is None
        assert list_stores() == []
        beta_dir = runs_root()
    assert beta_dir != alpha_dir
    shutil.copytree(alpha_dir, beta_dir)
    with bind_project(leak_env.records['beta']), pytest.raises(ProjectDirMismatchError):
        open_store(run_id)


def test_reprocess_and_upload_dirs_are_marked_with_their_project(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    from src.config.project_context import ProjectDirMismatchError, bind_project
    from src.services.curation import reprocess_job
    from src.services.curation.dataset_import.upload import datasets_root, receive_archive

    monkeypatch.setenv('OP_REPROCESS_JOBS_DIR', str(leak_env.root / 'reprocess'))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('ds/data.yaml', 'names: [a]')

    async def upload(record: Any) -> None:
        async def body() -> Any:
            yield buf.getvalue()

        await receive_archive(body(), upload_root=record.resources.upload_root)

    with bind_project(leak_env.records['alpha']):
        reprocess_job.jobs_root().mkdir(parents=True)
        assert (reprocess_job.jobs_root() / '.project').read_text() == 'alpha'
        asyncio.run(upload(leak_env.records['alpha']))
        alpha_uploads = datasets_root(leak_env.records['alpha'].resources.upload_root)
        assert (alpha_uploads / '.project').read_text() == 'alpha'
    # beta handed alpha's upload root: refused
    with bind_project(leak_env.records['beta']), pytest.raises(ProjectDirMismatchError):
        asyncio.run(upload(leak_env.records['alpha']))
