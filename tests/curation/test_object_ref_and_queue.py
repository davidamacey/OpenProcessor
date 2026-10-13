"""The object reference stored as ``source_identifier``, the fail-closed allow-list and
the durable queue's idempotency and retry rules."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from src.services.curation.bulk_ingest.object_queue import MAX_ATTEMPTS, ObjectQueue, QueueEntry
from src.services.curation.bulk_ingest.object_store import (
    ObjectRef,
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
)


if TYPE_CHECKING:
    from pathlib import Path

HOSTILE_KEYS = [
    'plain/a.jpg',
    'with space/and+plus.jpg',
    'hash#and?query=1/x.jpg',
    'pct%20/already%2Fencoded.jpg',
    'ünï/códe/日本.png',
    'trailing//double//slash.jpg',
]


@pytest.mark.parametrize('key', HOSTILE_KEYS)
def test_a_reference_round_trips_whatever_the_key(key: str) -> None:
    ref = ObjectRef('bkt', key, 'abc123-4', 'v1.2/3', 10)
    parsed = ObjectRef.parse(ref.to_uri())
    assert (parsed.bucket, parsed.key, parsed.etag, parsed.version_id) == (
        'bkt',
        key,
        'abc123-4',
        'v1.2/3',
    )
    assert ref.to_uri().count('#') == 1  # a '#' in the key cannot split the reference


def test_a_different_etag_is_a_different_reference() -> None:
    assert ObjectRef('b', 'k.jpg', 'one').to_uri() != ObjectRef('b', 'k.jpg', 'two').to_uri()


def test_non_references_are_rejected() -> None:
    for bad in ('http://x/y.jpg', 's3://', 's3://bucket-only'):
        with pytest.raises(ValueError, match='s3'):
            ObjectRef.parse(bad)


def test_s3_is_off_unless_configured_and_the_allow_list_binds_bucket_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_S3_ALLOWED', raising=False)
    with pytest.raises(S3NotConfiguredError):
        S3Config.from_env().check('any', 'k.jpg')

    monkeypatch.setenv('OP_S3_ALLOWED', 'whole, scoped/in/ ,s3://uri/form')
    config = S3Config.from_env()
    config.check('whole', 'anything/at/all.jpg')
    config.check('scoped', 'in/deep/x.jpg')
    config.check('uri', 'form/k')
    for bucket, key in (('scoped', 'out/x.jpg'), ('scoped', ''), ('unlisted', 'in/x.jpg')):
        with pytest.raises(S3NotAllowedError):
            config.check(bucket, key)


def _queue(tmp_path: Path) -> ObjectQueue:
    return ObjectQueue(tmp_path / 'q' / 'queue.sqlite')


def test_the_queue_is_idempotent_by_key_and_etag(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    assert queue.enqueue([QueueEntry('a.jpg', 'e1'), QueueEntry('b.jpg', 'e1')]) == (2, 0)
    assert queue.enqueue([QueueEntry('a.jpg', 'e1'), QueueEntry('a.jpg', 'e2')]) == (1, 1)
    assert [(e.key, e.etag) for e in queue.iter_pending()] == [
        ('a.jpg', 'e1'),
        ('b.jpg', 'e1'),
        ('a.jpg', 'e2'),
    ]
    queue.mark_done([('a.jpg', 'e1')])
    # A replay of a finished object queues nothing and leaves it done.
    assert queue.enqueue([QueueEntry('a.jpg', 'e1')]) == (0, 1)
    assert queue.counts() == {'pending': 2, 'done': 1, 'failed': 0, 'dead': 0}


def test_failures_retry_up_to_the_limit_and_terminal_kinds_give_up_at_once(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    queue.enqueue([QueueEntry('flaky.jpg', 'e'), QueueEntry('gone.jpg', 'e')])
    queue.mark_failed([('gone.jpg', 'e', 'NoSuchKey', 'gone')])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        queue.mark_failed([('flaky.jpg', 'e', '503', 'fetch_error')])
        expected = 'dead' if attempt == MAX_ATTEMPTS else 'failed'
        assert queue.counts()[expected] >= 1
    assert queue.counts() == {'pending': 0, 'done': 0, 'failed': 0, 'dead': 2}
    assert list(queue.iter_pending()) == []
    assert queue.retry_dead() == 2
    assert queue.counts()['pending'] == 2


def test_a_failed_row_is_offered_again_but_a_done_row_never_goes_back(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue([QueueEntry('a.jpg', 'e'), QueueEntry('b.jpg', 'e')])
    queue.mark_failed([('a.jpg', 'e', 'x', 'fetch_error')])
    queue.mark_done([('b.jpg', 'e')])
    queue.mark_failed([('b.jpg', 'e', 'late failure report', 'fetch_error')])
    assert [e.key for e in queue.iter_pending()] == ['a.jpg']
    assert queue.counts() == {'pending': 0, 'done': 1, 'failed': 1, 'dead': 0}


def test_concurrent_writers_lose_nothing(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    def writer(n: int) -> None:
        for i in range(30):
            queue.enqueue([QueueEntry(f'w{n}/{i}.jpg', 'e'), QueueEntry('shared.jpg', 'e')])
            queue.add_counters({'events.received': 1})

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert queue.counts()['pending'] == 6 * 30 + 1
    assert queue.meta_get('events.received') == 6 * 30
