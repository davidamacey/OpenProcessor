"""One scan pass of an S3 prefix or a folder into a source's queue: paging, parallel
sub-prefixes, the modification-time watermark and resume after a crash."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from _fake_object_store import FakeObjectStore
from src.services.curation.bulk_ingest.folder_scan import SETTLE_S, scan_folder
from src.services.curation.bulk_ingest.object_queue import ObjectQueue
from src.services.curation.bulk_ingest.object_store import ObjectFetchError
from src.services.curation.bulk_ingest.s3_listing import WATERMARK_OVERLAP_S, scan_s3


if TYPE_CHECKING:
    from pathlib import Path

EXT = ('.jpg', '.png')
T0 = 1_700_000_000.0


def _filled(store: FakeObjectStore, n_dirs: int = 3, per_dir: int = 5) -> int:
    for d in range(n_dirs):
        for i in range(per_dir):
            store.put('b', f'p/d{d}/{i}.jpg', b'x%d%d' % (d, i), modified=T0)
    store.put('b', 'p/top.png', b'top', modified=T0)
    store.put('b', 'p/d0/readme.txt', b'no', modified=T0)
    store.put('b', 'p/d0/', b'', modified=T0)  # a "folder" placeholder
    return n_dirs * per_dir + 1


def _scan(store: FakeObjectStore, queue: ObjectQueue, **kw: object) -> object:
    return scan_s3(store, queue, bucket='b', prefix='p/', extensions=EXT, workers=3, **kw)  # type: ignore[arg-type]


def test_a_first_pass_queues_every_image_across_pages_and_partitions(tmp_path: Path) -> None:
    store, queue = FakeObjectStore(page_size=2), ObjectQueue(tmp_path / 'q.sqlite')
    expected = _filled(store)
    result = _scan(store, queue, now=T0 + 10)
    assert result.queued == expected  # type: ignore[attr-defined]
    assert queue.counts()['pending'] == expected
    keys = {e.key for e in queue.iter_pending()}
    assert 'p/d0/readme.txt' not in keys and 'p/d0/' not in keys
    assert result.pages > 3  # type: ignore[attr-defined]


def test_a_later_pass_queues_only_what_arrived_since(tmp_path: Path) -> None:
    store, queue = FakeObjectStore(), ObjectQueue(tmp_path / 'q.sqlite')
    _filled(store)
    _scan(store, queue, now=T0 + 1000)
    queue.mark_done([(e.key, e.etag) for e in list(queue.iter_pending())])

    store.put('b', 'p/d1/new.jpg', b'new', modified=T0 + 1100)
    store.put('b', 'p/d9/also.jpg', b'also', modified=T0 + 1100)
    again = _scan(store, queue, now=T0 + 2000)
    assert {e.key for e in queue.iter_pending()} == {'p/d1/new.jpg', 'p/d9/also.jpg'}
    assert again.queued == 2  # type: ignore[attr-defined]


def test_an_object_that_landed_inside_the_overlap_is_not_missed(tmp_path: Path) -> None:
    store, queue = FakeObjectStore(), ObjectQueue(tmp_path / 'q.sqlite')
    _filled(store)
    _scan(store, queue, now=T0 + 1000)
    # Became visible late but carries a timestamp from just before the first pass started.
    store.put('b', 'p/late.jpg', b'late', modified=T0 + 1000 - WATERMARK_OVERLAP_S / 2)
    _scan(store, queue, now=T0 + 2000)
    assert 'p/late.jpg' in {e.key for e in queue.iter_pending()}


def test_a_crash_mid_pass_resumes_each_partition_where_it_stopped(tmp_path: Path) -> None:
    store, queue = FakeObjectStore(page_size=2), ObjectQueue(tmp_path / 'q.sqlite')
    expected = _filled(store, n_dirs=2, per_dir=6)
    store.fail_listing_after = 4
    with pytest.raises(ObjectFetchError):
        _scan(store, queue, now=T0 + 10)
    partial = queue.counts()['pending']
    assert 0 < partial < expected

    store.fail_listing_after = None
    store.list_calls.clear()
    result = _scan(store, queue, now=T0 + 20)
    assert result.resumed is True  # type: ignore[attr-defined]
    assert queue.counts()['pending'] == expected
    # No page listed before the crash is listed again: every call after the root carries a
    # continuation token or belongs to a partition that had not started.
    relisted_from_start = [c for c in store.list_calls if c[2] is None and c[3] is None]
    assert len(relisted_from_start) <= 2


def test_a_stop_request_leaves_the_checkpoint(tmp_path: Path) -> None:
    store, queue = FakeObjectStore(page_size=2), ObjectQueue(tmp_path / 'q.sqlite')
    _filled(store)
    calls = {'n': 0}

    def stop() -> bool:
        calls['n'] += 1
        return calls['n'] > 3

    from src.services.curation.bulk_ingest.s3_listing import ScanStoppedError

    with pytest.raises(ScanStoppedError):
        _scan(store, queue, should_stop=stop, now=T0)
    assert queue.meta_get('scan') is not None
    assert queue.meta_get('watermark') is None  # a pass that did not finish moves nothing


def test_the_folder_watcher_skips_unsettled_files_and_picks_them_up_next_pass(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'watched'
    root.mkdir()
    old, fresh = root / 'old.jpg', root / 'fresh.jpg'
    old.write_bytes(b'o')
    fresh.write_bytes(b'f')
    now = 2_000_000_000.0
    os.utime(old, (now - 600, now - 600))
    os.utime(fresh, (now - 1, now - 1))
    (root / 'skip.txt').write_text('no')

    queue = ObjectQueue(tmp_path / 'q.sqlite')
    first = scan_folder(queue, root, extensions=EXT, now=now)
    assert first.queued == 1
    assert [e.key for e in queue.iter_pending()] == [str(old)]

    second = scan_folder(queue, root, extensions=EXT, now=now + SETTLE_S + 1)
    assert second.queued == 1
    assert {e.key for e in queue.iter_pending()} == {str(old), str(fresh)}
    # Nothing changed: a third pass queues nothing.
    assert scan_folder(queue, root, extensions=EXT, now=now + 100).queued == 0
