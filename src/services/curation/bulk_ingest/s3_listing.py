"""Scan an S3 prefix into a source's queue: paged, parallel over sub-prefixes, resumable.

A *pass* lists every object under the source prefix and queues those that are new:
modified at or after the previous pass's watermark and not yet in the queue by
``(key, etag)``. The first pass has no watermark and queues everything. Progress is
written to the queue's ``meta`` after each page's rows are committed, so a restart
resumes each unfinished sub-prefix at its last continuation token instead of listing
from the start; a pass that completes moves the watermark to its own start minus
:data:`WATERMARK_OVERLAP_S` (clock skew and uploads that were in flight).

The top level is listed with a delimiter: objects directly under the prefix are
queued, and each child prefix becomes a partition listed by its own worker. A flat
bucket is one partition and no faster; the webhook trigger is the answer to a
listing that is too slow.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.bulk_ingest.object_queue import QueueEntry


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.services.curation.bulk_ingest.object_queue import ObjectQueue
    from src.services.curation.bulk_ingest.object_store import ListedObject, ObjectStore

WATERMARK_OVERLAP_S = 120.0
SCAN_META = 'scan'
WATERMARK_META = 'watermark'
LAST_SCAN_META = 'last_scan'
_ROOT = ''
_DONE = 'done'


class ScanStoppedError(Exception):
    """The scan was asked to stop (its source was removed or paused)."""


@dataclass
class ScanResult:
    pages: int = 0
    listed: int = 0
    queued: int = 0
    duplicates: int = 0
    resumed: bool = False


def wanted_extension(key: str, extensions: frozenset[str]) -> bool:
    if key.endswith('/'):
        return False  # a "folder" placeholder object
    dot = key.rfind('.')
    return dot != -1 and key[dot:].lower() in extensions


class _Pass:
    def __init__(
        self,
        store: ObjectStore,
        queue: ObjectQueue,
        bucket: str,
        extensions: frozenset[str],
        watermark: float | None,
        state: dict[str, Any],
        should_stop: Callable[[], bool],
    ) -> None:
        self.store = store
        self.queue = queue
        self.bucket = bucket
        self.extensions = extensions
        self.watermark = watermark
        self.state = state
        self.should_stop = should_stop
        self.lock = threading.Lock()
        self.failed = threading.Event()

    def _save(self) -> None:
        self.queue.meta_set(SCAN_META, self.state)

    def take(self, objects: Iterable[ListedObject]) -> None:
        wanted = [
            QueueEntry(o.ref.key, o.ref.etag, o.ref.version_id, o.ref.size)
            for o in objects
            if wanted_extension(o.ref.key, self.extensions)
            and (self.watermark is None or o.last_modified.timestamp() >= self.watermark)
        ]
        queued, duplicates = self.queue.enqueue(wanted)
        with self.lock:
            self.state['queued'] += queued
            self.state['duplicates'] += duplicates

    def page_done(self, partition: str, token: str | None, listed: int) -> None:
        with self.lock:
            self.state['pages'] += 1
            self.state['listed'] += listed
            self.state['partitions'][partition] = token if token else _DONE
            self._save()

    def check(self) -> None:
        if self.should_stop():
            raise ScanStoppedError
        if self.failed.is_set():
            raise ScanStoppedError

    def list_partition(self, partition: str, prefix: str) -> None:
        token = self.state['partitions'].get(partition)
        if token == _DONE:
            return
        while True:
            self.check()
            page = self.store.list_page(self.bucket, prefix, token=token, delimiter=None)
            self.take(page.objects)
            token = page.next_token
            self.page_done(partition, token, len(page.objects))
            if token is None:
                return

    def list_root(self, prefix: str) -> list[str]:
        """Queue the objects directly under ``prefix``; return the child prefixes."""
        token = self.state.get('root_token')
        children: list[str] = list(self.state.get('children') or [])
        if self.state.get('root_done'):
            return children
        while True:
            self.check()
            page = self.store.list_page(self.bucket, prefix, token=token, delimiter='/')
            self.take(page.objects)
            children.extend(page.prefixes)
            token = page.next_token
            with self.lock:
                self.state['pages'] += 1
                self.state['listed'] += len(page.objects)
                self.state['children'] = children
                self.state['root_token'] = token
                self.state['root_done'] = token is None
                for child in page.prefixes:
                    self.state['partitions'].setdefault(child, None)
                self._save()
            if token is None:
                return children


def scan_s3(
    store: ObjectStore,
    queue: ObjectQueue,
    *,
    bucket: str,
    prefix: str,
    extensions: Iterable[str],
    workers: int,
    should_stop: Callable[[], bool] = lambda: False,
    now: float | None = None,
) -> ScanResult:
    """Run (or resume) one pass. Blocking; raises the store's classified error or
    :class:`ScanStoppedError`, leaving the checkpoint for the next call."""
    started = now if now is not None else time.time()
    state: dict[str, Any] | None = queue.meta_get(SCAN_META)
    resumed = state is not None
    if state is None:
        state = {
            'pass_started_at': started,
            'pages': 0,
            'listed': 0,
            'queued': 0,
            'duplicates': 0,
            'partitions': {},
        }
    run = _Pass(
        store,
        queue,
        bucket,
        frozenset(e.lower() for e in extensions),
        queue.meta_get(WATERMARK_META),
        state,
        should_stop,
    )
    children = run.list_root(prefix)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run.list_partition, child, child) for child in children]
        error: BaseException | None = None
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                run.failed.set()
                error = error or exc
        if error is not None:
            raise error
    queue.meta_set(WATERMARK_META, state['pass_started_at'] - WATERMARK_OVERLAP_S)
    queue.meta_set(
        LAST_SCAN_META,
        {
            'finished_at': time.time(),
            'pages': state['pages'],
            'listed': state['listed'],
            'queued': state['queued'],
        },
    )
    queue.meta_set(SCAN_META, None)
    return ScanResult(
        state['pages'], state['listed'], state['queued'], state['duplicates'], resumed
    )
