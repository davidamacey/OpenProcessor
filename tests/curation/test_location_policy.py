"""The location_policy route and the purge it starts (CM-4), against PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from near_duplicates.conftest import job_env, kinds, make_project, producer, queue_hooks, wakes

from src.core.exceptions import RequestRejectedError
from src.db.repos.config import RevisionConflictError
from src.db.session import async_session_factory
from src.services.curation import location_policy as service, location_policy_store as store


if TYPE_CHECKING:
    import uuid
    from typing import Any

pytestmark = [pytest.mark.db, pytest.mark.asyncio]
__all__ = ['job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']


@pytest_asyncio.fixture(autouse=True)
async def bound(pg_registry: Any, monkeypatch: pytest.MonkeyPatch) -> uuid.UUID:
    project = await make_project()

    async def project_id() -> uuid.UUID:
        return project

    monkeypatch.setattr(store, 'bound_project_id', project_id)
    monkeypatch.setattr(service, 'current_project_id', lambda: project)
    store._cache.clear()
    return project


async def _put(policy: str, revision: int | None = None) -> Any:
    """Change the policy; at the current revision unless one is given."""
    if revision is None:
        revision = (await service.get_policy()).revision
    async with async_session_factory()() as session:
        return await service.change_policy(session, policy, expected_revision=revision)


async def test_a_new_project_has_the_default_policy() -> None:
    got = await service.get_policy()
    assert (got.location_policy, got.purge_job_id) == ('store', None)


async def test_a_switch_to_drop_stores_the_policy_and_starts_the_purge_step(producer: Any) -> None:
    before = (await service.get_policy()).revision
    changed = await _put('drop')

    assert (changed.location_policy, changed.revision) == ('drop', before + 1)
    assert changed.purge_job_id is not None
    assert producer.sent() == [changed.purge_job_id]
    assert (await service.get_policy()).location_policy == 'drop'


async def test_a_stale_revision_changes_nothing_and_starts_nothing(producer: Any) -> None:
    stale = (await service.get_policy()).revision
    await _put('store_coarse')
    with pytest.raises(RevisionConflictError):
        await _put('drop', stale)
    assert (await service.get_policy()).location_policy == 'store_coarse'
    assert len(producer.sent()) == 1


async def test_loosening_starts_no_purge(producer: Any) -> None:
    await _put('drop')
    sent = len(producer.sent())
    loosened = await _put('store')
    assert (loosened.location_policy, loosened.purge_job_id) == ('store', None)
    assert len(producer.sent()) == sent


async def test_tightening_from_coarse_to_drop_and_resending_the_same_policy_both_purge(
    producer: Any,
) -> None:
    first = await _put('store_coarse')
    assert first.purge_job_id is not None
    assert len(producer.sent()) == 1


async def test_a_busy_backfill_refuses_the_start_after_the_policy_was_saved(producer: Any) -> None:
    await _put('drop')
    with pytest.raises(RequestRejectedError) as busy:
        await _put('drop')
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert 'location_purge' in busy.value.detail['message']
    assert (await service.get_policy()).revision == busy.value.detail['revision']
    assert len(producer.sent()) == 1


async def test_an_unknown_policy_is_refused_before_anything_is_written() -> None:
    before = await service.get_policy()
    with pytest.raises(ValueError, match='location_policy'):
        await _put('keep')
    assert await service.get_policy() == before
