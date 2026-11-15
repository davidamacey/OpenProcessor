"""Live: an ingest while the frame engine cannot take a photo (plan 13a: 503 ``engine_unavailable``,
and a bulk run that waits instead of giving up).

Every photo of an ingest goes to the frame engine; the harness's engine (``fake_frame_engine.py``)
refuses on request (``tests/live/_engine.py``). What a user must never meet: a half-stored photo, a
duplicated object, or a bulk run that stops silently.

* An upload while the engine refuses answers 503 ``engine_unavailable`` with ``Retry-After``,
  stores nothing and claims nothing: the same upload after the engine is back stores the photo.
* A bulk run started while the engine refuses does not fail its files: they wait for their retry
  time (the run reports how many), no attempt is used, nothing is stored, and the run completes
  with every file stored after the engine is back.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from . import _compose
from ._engine import Refusal, engine_refuses
from ._ingest_support import IngestProject, count, ingest_project, picture
from .conftest import wait_for_state, wait_until


pytestmark = pytest.mark.live

FILES = 4
BULK_DONE = ('completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')
RED = (200, 30, 30)


@pytest.fixture(scope='module')
def project(api: Any, opensearch: Any) -> Any:
    with ingest_project(api, opensearch, 'engdown') as handle:
        yield handle


def _upload(api: Any, project: IngestProject, colour: tuple[int, int, int]) -> Any:
    return api.post(
        f'{project.prefix}/ingest/upload',
        files=[('images', ('photo.jpg', picture(colour), 'image/jpeg'))],
        data={'image_paths': json.dumps(['phone/photo.jpg'])},
    )


@pytest.mark.parametrize('mode', ['not_ready', 'unavailable'])
def test_an_upload_while_the_engine_refuses_answers_503_and_stores_nothing(
    api: Any, opensearch: Any, project: IngestProject, mode: Refusal
) -> None:
    before = (count(opensearch, project.images_index), count(opensearch, project.items_index))

    with engine_refuses(mode):
        refused = _upload(api, project, RED)

    assert refused.status_code == 503, refused.text
    assert refused.json()['code'] == 'engine_unavailable', refused.text
    assert int(refused.headers['Retry-After']) >= 1, dict(refused.headers)
    assert (
        count(opensearch, project.images_index),
        count(opensearch, project.items_index),
    ) == before, 'a refused photo left documents behind'

    # resending is safe: the photo was not claimed, so it is stored, not a duplicate
    again = _upload(api, project, RED)
    assert again.status_code == 200, again.text
    assert [r['status'] for r in again.json()['results']] == ['success'], again.text
    assert count(opensearch, project.images_index) == before[0] + 1


def _ledger(run_id: str) -> list[tuple[str, int, bool]]:
    """``(status, attempts, has a retry time)`` of every file of the run's source."""
    out = _compose.psql(
        'SELECT f.status, f.attempts, f.retry_at IS NOT NULL FROM ingest_source_files f '
        "WHERE f.source_id = (SELECT (params->>'source_id')::uuid FROM jobs "
        f"WHERE id = '{run_id}') ORDER BY f.file_key"
    )
    rows = [line.split('|') for line in out.splitlines()]
    return [(status, int(attempts), retry == 't') for status, attempts, retry in rows]


def test_a_bulk_run_started_while_the_engine_refuses_waits_and_completes_when_it_is_back(
    api: Any, opensearch: Any
) -> None:
    with ingest_project(api, opensearch, 'engwait') as project:
        folder = project.host_dir / 'wait'
        folder.mkdir(parents=True)
        for i in range(FILES):
            (folder / f'{i}.jpg').write_bytes(picture((i * 50 + 10, 90, 200 - i * 40)))
        body = {
            'folder': f'{project.container_dir}/wait',
            'source': 'engine_wait',
            'batch_size': 2,
            'concurrency': 1,
        }

        def fetch() -> dict[str, Any]:
            return api.get(f'{project.prefix}/ingest/bulk/{run_id}').json()

        with engine_refuses('not_ready'):
            started = api.post(f'{project.prefix}/ingest/bulk', json=body)
            assert started.status_code == 202, started.text
            run_id = started.json()['run_id']
            # the sender retries each batch's 503 for about 40 s, then records the files as
            # waiting; the run reports them and goes on running
            assert wait_until(
                lambda: fetch()['progress'].get('waiting') == FILES, timeout=240, interval=2.0
            ), fetch()
            assert fetch()['status'] == 'running', fetch()
            assert _ledger(run_id) == [('failed', 0, True)] * FILES, (
                'a waiting file uses no attempt'
            )
            assert count(opensearch, project.images_index) == 0

        run = wait_for_state(fetch, lambda r: r['status'] in BULK_DONE, timeout=300)

        assert run['status'] == 'completed', run
        assert count(opensearch, project.images_index) == FILES
        assert _ledger(run_id) == [('done', 0, False)] * FILES
        failures = api.get(f'{project.prefix}/ingest/bulk/{run_id}/failures').json()
        assert failures['total'] == 0, failures
