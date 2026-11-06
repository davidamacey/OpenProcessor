"""A registered config (a built-in pack) is activated by name and keeps its wire shape, and the
applied-config reports of the processes that follow a project."""

from __future__ import annotations

import pytest
from _config_fixtures import make_project

from src.db.repos import config as repo
from src.services.config_store import documents, get_config_store
from src.services.config_store.documents import Registered
from src.services.config_store.store import reset_config_stores
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.fixture(autouse=True)
def _fresh_stores():
    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.asyncio
async def test_activating_a_registered_pack_stores_a_system_revision_the_wire_does_not_show(
    pg_env,
) -> None:
    name = GENERIC_ITEM_PACK.name
    body = {'class_system': GENERIC_ITEM_PACK.class_system}
    activated = await documents.activate(
        axis='prompt_pack',
        name=name,
        revision=None,
        expected_active=None,
        registered=Registered(body=body, description='built-in pack'),
    )
    assert activated.ref == (name, activated.ref[1])  # type: ignore[index]

    async with documents.tx('inspect') as session:
        stored = await repo.get_config(
            session,
            project_id=await documents.bound_project_id(),
            kind='prompt_pack',
            name=name,
            revision=None,
        )
    assert stored is not None  # the activation's foreign key points at a real document
    assert stored.updated_by == 'system'
    assert stored.body == body

    store = get_config_store()
    await store.refresh()
    assert name not in store.current.packs  # not a stored config
    assert documents.wire_revision('prompt_pack', name, stored.revision) is None


@pytest.mark.asyncio
async def test_a_runtime_report_is_refreshed_not_duplicated(pg_env) -> None:
    await documents.report_runtime(process='detection', host='h1', config_epoch=3)
    await documents.report_runtime(process='detection', host='h1', config_epoch=5)
    await documents.report_runtime(process='detection', host='h2', config_epoch=4)
    reports = await documents.live_runtime_reports(process='detection')
    assert {(r.host, r.config_epoch) for r in reports} == {('h1', 5), ('h2', 4)}
    assert await documents.live_runtime_reports(process='other') == []
