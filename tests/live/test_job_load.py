"""The 1,000-job load check (plan 03_b2_overview step 24, owner decision D-83).

``VERIFY_LOAD_JOBS`` sets the job count (default 1000; use 50 for a smoke run of the same path).
It queues that many ``noop`` jobs against 4 ``jobs-worker`` processes (2 replicas, concurrency 2)
and requires: every job ends ``completed``; none is lost or run twice (one ``job_attempts`` row
per job); no deadlock in the worker logs; the peak of ``pg_stat_activity`` stays inside the
connection budget (``max_connections - 13``, 02b section 8); the run ends inside the bound of
1.2 s per job (20 minutes for 1,000). The timings are printed and written to
``docker/test/verify-data/job_load_<N>.json`` (git-ignored) for the report.

The infra ``noop`` is deployment-wide, so the plan's "spread over 10 projects" fairness clause
(each project's first job inside 60 s) has no project to measure here; the claim latency is
recorded for the whole set instead (``claim_p50_s``, ``claim_max_s``).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from tests.live import _compose
from tests.live._jobs import start_noop_burst
from tests.live.conftest import VERIFY_DATA_DIR


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.live

COUNT = int(os.environ.get('VERIFY_LOAD_JOBS', '1000'))
REPLICAS = 2
BOUND_S_PER_JOB = 1.2
ACTIVE = ('queued', 'running', 'cancelling')
CONNECTION_RESERVE = 13


class _ConnectionSampler:
    """Polls the connection count of the database every 2 s for the length of the run."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            out = _compose.psql(
                'SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()'
            )
            self.peak = max(self.peak, int(out))
            self._stop.wait(2.0)

    def __enter__(self) -> _ConnectionSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


@pytest.fixture
def four_job_workers() -> Iterator[None]:
    """2 jobs-worker replicas at concurrency 2, put back to one afterwards."""
    _compose.compose(
        'up', '-d', '--no-deps', '--no-build', '--wait', '--scale', f'jobs-worker={REPLICAS}',
        'jobs-worker', timeout=240,
    )  # fmt: skip
    yield
    _compose.compose(
        'up', '-d', '--no-deps', '--no-build', '--wait', '--scale', 'jobs-worker=1', 'jobs-worker',
        timeout=240,
    )  # fmt: skip


def _by_status(run: str) -> dict[str, int]:
    rows = _compose.psql(
        f"SELECT status, count(*) FROM jobs WHERE params->>'load_run' = '{run}' GROUP BY 1"
    )
    return {k: int(v) for k, v in (line.split('|') for line in rows.splitlines() if line)}


def test_a_burst_of_small_jobs_all_complete_once(four_job_workers: None) -> None:
    run = uuid.uuid4().hex
    max_connections = int(_compose.psql('SHOW max_connections'))
    with _ConnectionSampler() as sampler:
        began = time.monotonic()
        enqueue_s = start_noop_burst(COUNT, run)
        bound_s = max(120.0, COUNT * BOUND_S_PER_JOB)
        status: dict[str, int] = {}
        while time.monotonic() - began < bound_s:
            status = _by_status(run)
            if sum(status.values()) == COUNT and not any(status.get(s) for s in ACTIVE):
                break
            time.sleep(3.0)
        total_s = time.monotonic() - began

    assert status == {'completed': COUNT}, (
        f'after {total_s:.0f} s (bound {bound_s:.0f} s): {status}'
    )
    attempts = _compose.psql(
        'SELECT count(*), count(DISTINCT a.job_id), max(a.attempt) FROM job_attempts a '
        f"JOIN jobs j ON j.id = a.job_id WHERE j.params->>'load_run' = '{run}'"
    )
    assert attempts == f'{COUNT}|{COUNT}|1', f'a job ran twice or was lost: {attempts}'
    claim = _compose.psql(
        'SELECT extract(epoch FROM percentile_cont(0.5) WITHIN GROUP (ORDER BY a.started_at - j.created_at)), '
        'extract(epoch FROM max(a.started_at - j.created_at)) FROM job_attempts a '
        f"JOIN jobs j ON j.id = a.job_id WHERE j.params->>'load_run' = '{run}'"
    )
    p50, worst = (float(x) for x in claim.split('|'))
    logs = _compose.compose('logs', '--no-log-prefix', 'jobs-worker', timeout=300).stdout
    bad = [ln for ln in logs.splitlines() if 'deadlock' in ln.lower()]
    assert not bad, bad[:5]
    assert sampler.peak <= max_connections - CONNECTION_RESERVE, (
        f'peak {sampler.peak} connections of {max_connections}'
    )
    report: dict[str, Any] = {
        'jobs': COUNT,
        'workers': f'{REPLICAS} replicas x concurrency 2',
        'enqueue_s': round(enqueue_s, 1),
        'total_s': round(total_s, 1),
        'jobs_per_s': round(COUNT / total_s, 2),
        'claim_p50_s': round(p50, 2),
        'claim_max_s': round(worst, 2),
        'peak_connections': sampler.peak,
        'max_connections': max_connections,
    }
    (VERIFY_DATA_DIR / f'job_load_{COUNT}.json').write_text(json.dumps(report, indent=2))
    print(f'JOB_LOAD {json.dumps(report)}')
