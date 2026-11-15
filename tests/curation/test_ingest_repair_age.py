"""``REPAIR_AFTER_S`` is derived from the real OpenSearch client, not guessed.

An incomplete photo is repaired by the next ingest only when no live writer can still be writing
it (its writer marked the failure, or its doc is older than the longest a writer can live). The
constant is the product of the requests a unit makes in a row and the longest one request can take
(``src/services/curation/ingest_writer.py`` holds the derivation); each input is pinned here
against the library, the settings and the code, so a change to any of them fails this test instead
of silently shortening the bound.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from opensearchpy import AsyncTransport

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.config.settings import Settings
from src.services.curation import image_completeness, ingest_writer as w
from src.services.near_duplicates import content_key
from src.tasks import b4_project, jobs_c


def test_the_attempts_of_one_request_are_the_transports_retries_plus_one() -> None:
    params = inspect.signature(AsyncTransport.__init__).parameters
    assert params['max_retries'].default + 1 == w.OPENSEARCH_ATTEMPTS
    # a timeout is not retried: the 4 attempts are of connection errors and 502/503/504
    assert params['retry_on_timeout'].default is False


def test_the_client_timeout_is_the_slowest_client_that_writes_images() -> None:
    clients = [
        Settings.model_fields['opensearch_timeout'].default,
        jobs_c.OPENSEARCH_TIMEOUT_S,
        b4_project.OPENSEARCH_TIMEOUT_S,
    ]
    assert max(clients) == w.CLIENT_TIMEOUT_S


def test_the_request_counts_are_the_retry_rounds_of_the_writes() -> None:
    assert w.LEASE_WRITE_REQUESTS == 2 * (image_completeness.LIST_WRITE_RETRIES + 1)
    assert w.MARK_WRITE_REQUESTS == 2 * (image_completeness.MARK_WRITE_RETRIES + 1)
    assert w.LEASE_RETURN_REQUESTS == w.MARK_WRITE_REQUESTS  # the same write, the same rounds
    assert w.NEW_PHOTO_REQUESTS == (
        w.IMAGES_STEP_REQUESTS + w.ITEMS_STEP_REQUESTS + w.MARK_WRITE_REQUESTS
    )
    assert w.REPAIR_REQUESTS == (
        w.IMAGES_STEP_REQUESTS
        + w.MATCH_REQUESTS
        + w.LEASE_WRITE_REQUESTS
        + w.REPAIR_CREATE_REQUESTS
        + w.LEASE_RETURN_REQUESTS
        + w.MARK_WRITE_REQUESTS
    )
    assert max(w.NEW_PHOTO_REQUESTS, w.REPAIR_REQUESTS) == w.SEQUENTIAL_REQUESTS


def test_the_age_covers_the_requests_pauses_and_attempts() -> None:
    pauses = 2 * sum(w.RETRY_429_PAUSES_S)
    assert w.SEQUENTIAL_REQUESTS * 4 * 60 + pauses == w.REPAIR_AFTER_S
    assert w.REPAIR_AFTER_S > 3600, 'an hour of the slowest possible client'


@pytest.mark.asyncio
async def test_a_unit_that_hits_every_conflict_makes_no_more_requests_than_the_bound_counts() -> (
    None
):
    """The densest path the fake can drive: a 409 on the images create against another copy's
    incomplete old doc (settle, completeness read), a conflict on one item create (refetch and
    update), and the list write. Its request count stays under the constants'."""
    svc, os_fake, _ = _make_service(
        detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
    )
    data = _jpeg_bytes(seed=51)
    await svc.ingest_one(data, '/p/a.jpg')
    (image_id,) = os_fake.images
    first_item = os_fake.images[image_id]['item_ids'][0]
    os_fake.items.clear()
    os_fake.images[image_id]['indexed_at'] = '2020-01-01T00:00:00+00:00'
    os_fake.images[image_id].pop('ingest_incomplete', None)
    os_fake._create_conflict_pending.add(first_item)

    counted = {'n': 0}

    def counting(name: str) -> Any:
        real = getattr(os_fake, name)

        async def call(*a: Any, **k: Any) -> Any:
            counted['n'] += 1
            return await real(*a, **k)

        return call

    for name in ('bulk', 'mget', 'search'):
        setattr(os_fake, name, counting(name))

    # No content row: this ingest claims and creates, and its create meets the doc (the 409 path).
    await content_key.release_bound([image_id])
    before = counted['n']
    await asyncio.wait_for(svc.ingest_one(data, '/p/b.jpg'), timeout=20)

    assert counted['n'] - before <= w.SEQUENTIAL_REQUESTS
