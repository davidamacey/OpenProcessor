"""Live: an API restart in the middle of a bulk run leaves no file dead (plan 13a, lane L7).

A bulk run sends its batches to the API over loopback. While the API restarts, a send either finds
no connection (``ConnectError``: nothing was processed, so it is retried and, when the retries run
out, recorded as the transient ``busy``, which uses no attempt) or meets a connection that breaks
after the request went out (``batch_error``: it may be a file that crashes the worker, so it uses
one attempt of three). So a restart costs each in-flight file at most one attempt, never makes one
dead, and every file is stored after the run or one resume.

Dead rows are never claimed again, so "every file stored after the resume" proves none is dead.
Runs in the harness project only (``docker/test/compose.yml``).
"""

from __future__ import annotations

from typing import Any

import pytest

from . import _compose
from ._ingest_support import IngestProject, count, ingest_project, picture
from .conftest import wait_for_state, wait_until


pytestmark = pytest.mark.live

FILES = 60
BULK_DONE = ('completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')


def _colour(i: int) -> tuple[int, int, int]:
    return (i * 4 % 256, (i * 7 + 40) % 256, (i * 13 + 90) % 256)


def _api_answers(api: Any) -> bool:
    try:
        return api.get('/live').status_code == 200
    except Exception:  # the API is still restarting
        return False


def test_an_api_restart_in_the_middle_of_a_bulk_run_leaves_no_file_dead_and_every_file_is_stored_after_the_run_or_a_resume(
    api: Any, opensearch: Any
) -> None:
    with ingest_project(api, opensearch, 'bulkrestart') as project:
        _run_and_restart(api, opensearch, project)


def _run_and_restart(api: Any, opensearch: Any, project: IngestProject) -> None:
    folder = project.host_dir / 'restart'
    folder.mkdir(parents=True)
    for i in range(FILES):
        (folder / f'{i:03d}.jpg').write_bytes(picture(_colour(i)))
    body = {
        'folder': f'{project.container_dir}/restart',
        'source': 'restart_live',
        'batch_size': 2,
        'concurrency': 2,
    }
    started = api.post(f'{project.prefix}/ingest/bulk', json=body)
    assert started.status_code == 202, started.text
    run_id = started.json()['run_id']

    def fetch() -> dict[str, Any]:
        return api.get(f'{project.prefix}/ingest/bulk/{run_id}').json()

    assert wait_until(lambda: fetch()['progress']['ingested'] >= 4, timeout=120), fetch()
    _compose.compose('restart', 'api', timeout=240)
    assert wait_until(lambda: _api_answers(api), timeout=180)

    run = wait_for_state(fetch, lambda r: r['status'] in BULK_DONE, timeout=600)
    if count(opensearch, project.images_index) < FILES:
        # the run gave up or ended with files failed: a resume continues the same source
        assert run['status'] != 'cancelled', run
        resumed = api.post(f'{project.prefix}/ingest/bulk', json=body)
        assert resumed.status_code == 202, resumed.text
        assert resumed.json()['resumed'] is True, resumed.text
        run_id = resumed.json()['run_id']
        run = wait_for_state(fetch, lambda r: r['status'] in BULK_DONE, timeout=600)

    assert run['status'] == 'completed', run
    assert count(opensearch, project.images_index) == FILES  # a dead row would not be stored
    failures = api.get(f'{project.prefix}/ingest/bulk/{run_id}/failures').json()
    assert failures['total'] == 0, failures
