"""SSE resume across api replicas (plan 05 section 4B, review item R-15; WHATWG SSE).

A client reads the global event stream from replica 1, drops, and reconnects to replica 2 with the
``Last-Event-ID`` it last saw. The stream is a view of the ``events`` table, so replica 2 must
replay exactly the events after that id: none missing, none repeated, and no ``reset``. Spec:
https://html.spec.whatwg.org/multipage/server-sent-events.html#the-last-event-id-header

Needs the scale overlay (``docker/test/compose.scale.yml``, two api replicas); see ``_scale.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.live import _compose
from tests.live._jobs import start_infra_job, wait_for_job
from tests.live._scale import REPLICA1_URL, REPLICA2_URL, require_replicas
from tests.live._sse import SseCollector
from tests.live.conftest import API_PREFIX


pytestmark = pytest.mark.live

TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}


def _stored_seqs(*job_ids: str) -> list[int]:
    quoted = ', '.join(f"'{j}'" for j in job_ids)
    rows = _compose.psql(f'SELECT id FROM events WHERE job_id IN ({quoted}) ORDER BY id')
    return [int(r) for r in rows.splitlines()]


def _seq(event_id: str) -> int:
    return int(event_id.split('-')[1])


def test_resume_on_the_other_replica_has_no_gap_and_no_duplicate(api: Any) -> None:
    require_replicas()
    events_a = f'{REPLICA1_URL}{API_PREFIX}/events'
    events_b = f'{REPLICA2_URL}{API_PREFIX}/events'

    first = SseCollector(events_a).start()
    try:
        running = start_infra_job('sleep_noop', seconds=15)
        assert first.wait_for(
            lambda ev: (
                sum(
                    1
                    for e in ev
                    if e.get('event') == 'job.progress' and e['data']['job_id'] == running
                )
                >= 3
            ),
            timeout_s=60,
        ), f'no progress on replica 1: {first.for_job(running)} error={first.error}'
    finally:
        first.close()
    cursor = first.events[-1]['id']
    seen_before = [e['id'] for e in first.events if e.get('id')]

    # Events published while no client is connected must reach the client on the other replica.
    away = start_infra_job('noop')
    assert wait_for_job(api, away, TERMINAL, timeout_s=60) is not None

    second = SseCollector(events_b, last_event_id=cursor).start()
    try:
        assert second.wait_for(
            lambda ev: any(
                e.get('event') == 'job.state'
                and e['data']['job_id'] == running
                and e['data']['status'] == 'completed'
                for e in ev
            ),
            timeout_s=90,
        ), f'running job never completed on replica 2: {second.for_job(running)} {second.error}'
    finally:
        second.close()

    assert not any(e.get('event') == 'reset' for e in second.events), second.events[:3]
    after = [e['id'] for e in second.events if e.get('id')]
    assert after, 'replica 2 replayed nothing'
    assert all(_seq(i) > _seq(cursor) for i in after), (cursor, after)
    assert len(after) == len(set(after)), f'duplicates on replica 2: {after}'
    assert not set(after) & set(seen_before), 'an event delivered on both replicas'

    mine = {running, away}
    got = sorted(
        _seq(e['id']) for e in [*first.events, *second.events] if e['data'].get('job_id') in mine
    )
    assert got == sorted(_stored_seqs(*mine)), 'events lost or invented across replicas'
