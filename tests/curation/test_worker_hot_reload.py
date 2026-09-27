"""W2 sec 4.5: the detection worker's hot-reloadable runtime --
``scripts/curation/worker/runtime.py`` (``build_runtime``,
``RuntimeHolder``, ``quiesce_and_swap``), and the write-path stamping
(``RegionFields.profile``/``profile_revision``, ``vlm_prompt_pack``) in
``scripts/curation/worker/bulk_writer.py``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.curation.region_worker_main as worker
from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.config.project_context import bind_project, current_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store, reset_config_stores
from src.services.detection import profile_registry


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _record(slug: str, revision: int = 1) -> ProjectRecord:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=revision,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    yield
    reset_config_stores()


# =============================================================================
# RuntimeHolder isolation: "alpha activation does not swap beta"
# =============================================================================


@pytest.mark.asyncio
async def test_alpha_activation_does_not_swap_beta() -> None:
    """Activating a profile in project alpha's config store must not
    make ``config_wants_swap`` report ``True`` for beta's store -- each
    project's activation is tracked in its own, independently-refreshed
    ``ConfigStore`` (keyed by slug), so a change in one is invisible to
    the other."""
    from scripts.curation.worker.runtime import config_wants_swap

    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    beta = _record('beta')

    with bind_project(alpha):
        alpha_store = get_config_store(mode='pinned')
        await alpha_store.refresh(client)
    with bind_project(beta):
        beta_store = get_config_store(mode='pinned')
        await beta_store.refresh(client)

    # Neither has activated anything yet -- no swap wanted for either
    # relative to a runtime baseline of "nothing active" (None, None).
    assert config_wants_swap(alpha_store, (None, None)) is False
    assert config_wants_swap(beta_store, (None, None)) is False

    with bind_project(alpha):
        idx = alpha_store.index
        doc = await save_config(
            client, idx, kind='region_profile', name='wheel', body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
        await alpha_store.refresh(client)
        alpha_store.pin_active()

    # Alpha's own store now wants a swap...
    assert config_wants_swap(alpha_store, (None, None)) is True
    # ...but beta's store, never refreshed against alpha's index, is untouched.
    await beta_store.refresh(client)
    assert config_wants_swap(beta_store, (None, None)) is False


# =============================================================================
# build_runtime / quiesce_and_swap
# =============================================================================


def _fake_args() -> Any:
    ns = MagicMock()
    ns.segmenter_url = ''
    ns.vlm_url = ''
    return ns


@pytest.mark.asyncio
async def test_build_runtime_returns_refs() -> None:
    from scripts.curation.worker.runtime import build_runtime

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    rt = await build_runtime(
        pool=MagicMock(),
        profile=profile,
        pack=pack,
        args=_fake_args(),
        profile_revision=3,
        pack_revision=None,
    )
    assert rt.profile_ref == (profile.name, 3)
    assert rt.pack_ref == (pack.name, None)
    assert rt.detector is not None
    assert rt.vlm is None  # no vlm_url


@pytest.mark.asyncio
async def test_quiesce_and_swap_drains_queues_before_building() -> None:
    from scripts.curation.worker.runtime import RuntimeHolder, quiesce_and_swap

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    q: asyncio.Queue[Any] = asyncio.Queue()
    await q.put('item')
    holder = RuntimeHolder()
    drained_before_build = False

    async def _drain_soon() -> None:
        nonlocal drained_before_build
        await asyncio.sleep(0.01)
        drained_before_build = True
        q.task_done()

    asyncio.get_event_loop().create_task(_drain_soon())
    rt = await quiesce_and_swap(
        queues=[q],
        holder=holder,
        slug='alpha',
        pool=MagicMock(),
        profile=profile,
        pack=pack,
        args=_fake_args(),
    )
    assert drained_before_build
    assert holder.get('alpha') is rt
    assert holder.get('beta') is None


# =============================================================================
# Write-path stamping
# =============================================================================


def _make_task(*, crop_id: str, status: str | None = 'pending') -> _ItemTask:
    return _ItemTask(
        project=current_project().record,
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status=status,
        class_name='audi',
    )


@pytest.mark.asyncio
async def test_bulk_write_stamps_region_profile_and_pack() -> None:
    F = get_region_fields()
    a = _make_task(crop_id='a')
    a.update_doc = {F.status: 'detected', F.score: 0.9}

    async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
        found = {d['_id']: {F.status: 'pending'} for d in body['docs']}
        return make_mget_response(found)

    async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        items = [
            make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
        ]
        return make_bulk_response(items)

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_fake_mget)
    opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

    n_written, _ = await worker._bulk_update(opensearch, [a])
    assert n_written == 1

    assert opensearch.bulk.await_args is not None
    bulk_body = opensearch.bulk.await_args.kwargs['body']
    written_doc = bulk_body[1]['doc']
    active_profile = profile_registry.get_active_region_profile()
    assert active_profile is not None
    assert written_doc[F.profile] == active_profile.name
    # env-registered (never activated through the store) -- no revision.
    assert written_doc[F.profile_revision] is None
    assert written_doc['vlm_prompt_pack']


@pytest.mark.asyncio
async def test_bulk_write_stamps_store_activated_profile_revision() -> None:
    """A profile activated through the store (with a revision) stamps
    that revision, not ``None`` -- distinguishing a stored activation
    from the env/file default."""
    F = get_region_fields()
    client = FakeConfigOpenSearch()
    project = _record('wheel-project')

    active_profile = profile_registry.get_active_region_profile()
    assert active_profile is not None

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        doc = await save_config(
            client,
            idx,
            kind='region_profile',
            name=active_profile.name,
            body={},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name=doc['name'],
            revision=doc['revision'],
            expected_active=None,
        )
        await store.refresh(client)
        store.pin_active()

        a = _make_task(crop_id='a')
        a.update_doc = {F.status: 'detected', F.score: 0.9}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            found = {d['_id']: {F.status: 'pending'} for d in body['docs']}
            return make_mget_response(found)

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, _ = await worker._bulk_update(opensearch, [a])
        assert n_written == 1
        assert opensearch.bulk.await_args is not None
        bulk_body = opensearch.bulk.await_args.kwargs['body']
        written_doc = bulk_body[1]['doc']
        assert written_doc[F.profile_revision] == doc['revision']
