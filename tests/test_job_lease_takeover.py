"""Acquire, renew, release and takeover of a lease against the version rules of the server."""

from __future__ import annotations

import threading

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch

from src.clients import job_lease
from src.domain.job_lease import Acquired, Busy, HeldLease, Lost, Renewed
from src.utils.upstream_errors import UpstreamError


pytestmark = pytest.mark.real_lease_io


async def _renewed(client: FakeLeaseOpenSearch, lease: HeldLease) -> HeldLease:
    result = await job_lease.renew(client, lease)
    assert isinstance(result, Renewed)
    return result.lease


def _client() -> FakeLeaseOpenSearch:
    return FakeLeaseOpenSearch({}, threading.Lock())


@pytest.mark.asyncio
async def test_sequential_acquire_release_acquire_gives_generations_one_and_two() -> None:
    client = _client()
    first = await job_lease.acquire(client, 'probe', None, 'j1')
    assert isinstance(first, Acquired)
    assert first.lease.generation == 1
    assert await job_lease.release(client, first.lease) is True

    second = await job_lease.acquire(client, 'probe', None, 'j2')

    assert isinstance(second, Acquired)
    assert second.lease.generation == 2


@pytest.mark.asyncio
async def test_a_live_holder_makes_the_second_acquire_busy() -> None:
    client = _client()
    held = await job_lease.acquire(client, 'probe', None, 'j1')
    assert isinstance(held, Acquired)

    again = await job_lease.acquire(client, 'probe', None, 'j2')

    assert isinstance(again, Busy)
    assert again.job_id == 'j1'


@pytest.mark.asyncio
async def test_scopes_are_independent_leases() -> None:
    client = _client()
    assert isinstance(await job_lease.acquire(client, 'label_op', 'a', 'j1'), Acquired)
    assert isinstance(await job_lease.acquire(client, 'label_op', 'b', 'j2'), Acquired)


@pytest.mark.asyncio
async def test_a_renew_after_a_takeover_is_lost_and_the_old_release_overwrites_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    old = await job_lease.acquire(client, 'probe', None, 'j1')
    assert isinstance(old, Acquired)
    old = Acquired(await _renewed(client, old.lease))
    # The holder stalls past the TTL; a sibling takes the lease over.
    real_time = job_lease.time.time
    monkeypatch.setattr(job_lease.time, 'time', lambda: real_time() + 500)
    taken = await job_lease.takeover_expired(client, 'probe', None, 'j2')
    monkeypatch.setattr(job_lease.time, 'time', real_time)
    assert isinstance(taken, Acquired)
    assert taken.lease.generation == 2

    assert isinstance(await job_lease.renew(client, old.lease), Lost)
    assert await job_lease.release(client, old.lease) is False

    now_held = await job_lease.acquire(client, 'probe', None, 'j3')
    assert isinstance(now_held, Busy)
    assert now_held.job_id == 'j2'


@pytest.mark.asyncio
async def test_takeover_expired_leaves_a_fresh_lease_alone() -> None:
    client = _client()
    held = await job_lease.acquire(client, 'probe', None, 'j1')
    assert isinstance(held, Acquired)

    result = await job_lease.takeover_expired(client, 'probe', None, 'j2')

    assert isinstance(result, Busy)
    assert isinstance(await job_lease.renew(client, held.lease), Renewed)


@pytest.mark.asyncio
async def test_a_transport_error_on_get_raises_instead_of_proceeding() -> None:
    client = _client()
    client.fail_get = ConnectionError('down')

    with pytest.raises(UpstreamError):
        await job_lease.acquire(client, 'probe', None, 'j1')


@pytest.mark.asyncio
async def test_a_deleted_lease_document_makes_the_renew_lost() -> None:
    client = _client()
    held = await job_lease.acquire(client, 'probe', None, 'j1')
    assert isinstance(held, Acquired)
    client.store.clear()

    assert isinstance(await job_lease.renew(client, held.lease), Lost)
