"""Write-path stamping of ``RegionFields.profile``, ``profile_revision`` and ``vlm_prompt_pack`` in ``bulk_writer.py``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from curation.worker_hot_reload._helpers import _record
from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.config.project_context import bind_project, current_project
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store
from src.services.detection import profile_registry


pytestmark = pytest.mark.usefixtures('reference_region_profile')


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
    a.update_doc = {F.status: 'detected', F.max_score: 0.9}
    # the pack stamp is per-TASK, gated on whether a
    # VLM call actually contributed to this task's write this pass.
    a.vlm_called = True

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

    n_written, _ = await _bulk_update(opensearch, [a])
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
async def test_bulk_write_does_not_stamp_pack_when_no_vlm_call_happened() -> None:
    """a write whose task never actually involved a
    VLM call (no VLM configured, or a write path that skipped it, e.g.
    the high-confidence secondary-segmenter auto-skip) must not carry
    ``vlm_prompt_pack`` -- that would claim a VLM ran when it didn't."""
    F = get_region_fields()
    a = _make_task(crop_id='a')
    a.update_doc = {F.status: 'detected', F.max_score: 0.9}
    assert a.vlm_called is False  # the default

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

    n_written, _ = await _bulk_update(opensearch, [a])
    assert n_written == 1

    assert opensearch.bulk.await_args is not None
    bulk_body = opensearch.bulk.await_args.kwargs['body']
    written_doc = bulk_body[1]['doc']
    assert 'vlm_prompt_pack' not in written_doc


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
        a.update_doc = {F.status: 'detected', F.max_score: 0.9}

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

        n_written, _ = await _bulk_update(opensearch, [a])
        assert n_written == 1
        assert opensearch.bulk.await_args is not None
        bulk_body = opensearch.bulk.await_args.kwargs['body']
        written_doc = bulk_body[1]['doc']
        assert written_doc[F.profile_revision] == doc['revision']
