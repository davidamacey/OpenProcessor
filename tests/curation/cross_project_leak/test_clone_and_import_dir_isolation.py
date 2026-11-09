"""Clone and upload directory isolation between projects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED
from curation.cross_project_leak._seeders import _stored_prompt_pack, _stored_region_profile


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


pytestmark = pytest.mark.db


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
    _stored_prompt_pack(leak_env, 'alpha')
    _stored_region_profile(leak_env, 'alpha')
    source_indexes = set(leak_env.records['alpha'].resources.indexes.values())

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
    # the config store is PostgreSQL: a clone writes no OpenSearch document of the source
    assert not source_indexes & set(new_writes)

    alpha = SCOPED.format(project='alpha')
    beta = SCOPED.format(project='beta')
    for kind, source_name, new_name in (
        ('prompt_packs', 'alpha-model', 'from-alpha-pack'),
        ('region_profiles', 'alpha-model', 'from-alpha-profile'),
    ):
        # the clone landed in the target only, and the source was only read
        assert client.get(f'{beta}/{kind}/{new_name}').status_code == 200
        assert client.get(f'{alpha}/{kind}/{new_name}').status_code == 404
        source = client.get(f'{alpha}/{kind}/{source_name}')
        assert source.status_code == 200
        assert source.json()['revision'] == 1


def test_upload_dirs_are_marked_with_their_project(leak_env: LeakEnv) -> None:
    import io
    import zipfile

    from curation.dataset_import.upload_support import receive_archive
    from src.config.project_context import ProjectDirMismatchError, bind_project
    from src.services.curation.dataset_import.upload import datasets_root

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('ds/data.yaml', 'names: [a]')

    async def upload(record: Any) -> None:
        async def body() -> Any:
            yield buf.getvalue()

        await receive_archive(body(), upload_root=record.resources.upload_root)

    with bind_project(leak_env.records['alpha']):
        asyncio.run(upload(leak_env.records['alpha']))
        alpha_uploads = datasets_root(leak_env.records['alpha'].resources.upload_root)
        assert (alpha_uploads / '.project').read_text() == 'alpha'
    # beta handed alpha's upload root: refused
    with bind_project(leak_env.records['beta']), pytest.raises(ProjectDirMismatchError):
        asyncio.run(upload(leak_env.records['alpha']))
