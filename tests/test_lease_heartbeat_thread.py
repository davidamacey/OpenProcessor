"""The lease heartbeat runs on its own thread: a blocked event loop does not cost the lease."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch

from src.clients import job_lease
from src.domain.job_lease import Acquired, Busy
from src.services.curation import lease_heartbeat


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.real_lease_io


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> FakeLeaseOpenSearch:
    monkeypatch.setattr(lease_heartbeat, 'LEASE_TTL_S', 2.0)
    monkeypatch.setattr(lease_heartbeat, 'HEARTBEAT_TICK_S', 0.2)
    monkeypatch.setattr(lease_heartbeat, 'FENCE_MARGIN_S', 0.5)
    return FakeLeaseOpenSearch({}, threading.Lock())


async def _acquire(client: FakeLeaseOpenSearch, ttl: float = 2.0) -> Acquired:
    result = await job_lease.acquire(client, 'probe', None, 'j1', ttl_s=ttl)
    assert isinstance(result, Acquired)
    return result


@pytest.mark.asyncio
async def test_a_blocked_event_loop_keeps_its_lease(fast: FakeLeaseOpenSearch) -> None:
    held = await _acquire(fast)
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, client_factory=lambda: fast)
    beat.start()

    time.sleep(3.0)  # blocks this loop for longer than the TTL: an asyncio ticker would starve

    other = await job_lease.acquire(fast, 'probe', None, 'j2')
    assert isinstance(other, Busy)
    assert beat.may_write()
    await beat.release(fast)


@pytest.mark.asyncio
async def test_stop_and_release_lets_a_new_holder_in_with_the_next_generation(
    fast: FakeLeaseOpenSearch,
) -> None:
    held = await _acquire(fast)
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, client_factory=lambda: fast)
    beat.start()
    await beat.release(fast)

    second = await job_lease.acquire(fast, 'probe', None, 'j2')

    assert isinstance(second, Acquired)
    assert second.lease.generation == 2


@pytest.mark.asyncio
async def test_a_taken_over_lease_sets_lost_and_blocks_writes(
    fast: FakeLeaseOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = await _acquire(fast)
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, client_factory=lambda: fast)
    real = time.time
    # A sibling sees the lease expired (clock jump) and takes it over before the next renew.
    monkeypatch.setattr(job_lease.time, 'time', lambda: real() + 500)
    taken = await job_lease.acquire(fast, 'probe', None, 'j2')
    monkeypatch.setattr(job_lease.time, 'time', real)
    assert isinstance(taken, Acquired)

    beat.start()

    assert beat.lost.wait(5)
    assert not beat.may_write()
    assert await beat.confirm(fast) is False


@pytest.mark.asyncio
async def test_confirm_renews_a_live_lease(fast: FakeLeaseOpenSearch) -> None:
    held = await _acquire(fast)
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, client_factory=lambda: fast)
    beat.start()

    assert await beat.confirm(fast) is True


@pytest.mark.asyncio
async def test_unreachable_opensearch_self_fences_before_the_ttl(
    fast: FakeLeaseOpenSearch,
) -> None:
    held = await _acquire(fast)
    broken = FakeLeaseOpenSearch(fast.store, fast.lock)

    async def boom(*_: object, **__: object) -> dict[str, object]:
        raise ConnectionError('opensearch down')

    broken.index = boom  # type: ignore[method-assign]
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, client_factory=lambda: broken)
    beat.start()

    assert beat.lost.wait(5)


@pytest.mark.asyncio
async def test_a_lost_lease_fences_the_jobs_state_writes(
    fast: FakeLeaseOpenSearch, tmp_path: Path
) -> None:
    from src.services.curation.file_job import FileJob, LeaseLostError

    held = await _acquire(fast)
    job = FileJob(tmp_path / 'job')
    beat = lease_heartbeat.LeaseHeartbeat(held.lease, job=job, client_factory=lambda: fast)
    beat.start()
    job.write({'status': 'running'})

    beat.lost.set()

    with pytest.raises(LeaseLostError):
        job.update(status='completed')
    assert job.read()['status'] == 'running'
    beat.stop()
    job.update(status='completed')  # the fence is cleared once the holder is done
