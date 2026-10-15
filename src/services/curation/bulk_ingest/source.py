"""Where a bulk run reads its image paths from: a folder or a manifest.

Both are paths inside the API container and must lie under a configured source
root (``OP_SOURCE_ROOT`` / ``OP_SOURCE_PATH_ALIASES``), the same rule every ingest
path obeys. Entries are streamed, never materialised, so a million-image source
costs no more memory than a small one.

A ``queue`` source names a registered ingest source (see ``sources.py``): the run drains
its durable queue, after one scan pass when ``scan`` is set, and yields the references the
queue holds (``s3://`` references for an S3 source, paths for a folder source).

A manifest is a text file: one absolute image path per line, or JSON Lines whose
objects carry a ``path``. Blank lines and ``#`` comments are skipped. An entry
outside the source roots is not dropped here: it reaches the runner and is
reported as a failure with its reason.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.services.curation.bulk_ingest.limits import DEFAULT_EXTENSIONS
from src.services.curation.bulk_ingest.object_store import ObjectRef
from src.services.curation.bulk_ingest.s3_listing import ScanStoppedError
from src.services.curation.bulk_ingest.source_scan import scan_source
from src.services.curation.bulk_ingest.sources import IngestSource, load_source, source_queue
from src.services.curation.bulk_ingest.walk import iter_image_paths
from src.services.curation.image_serving import UNSERVABLE_PATH_ERROR, is_servable_image_path


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from src.services.curation.bulk_ingest.object_queue import QueueEntry


class InvalidSourceError(ValueError):
    """The folder or manifest cannot be used; the message says why."""


@dataclass(frozen=True)
class BulkSource:
    kind: Literal['folder', 'manifest', 'queue']
    path: str
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    # A queue source only: scan the source into its queue before draining it.
    scan: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            'kind': self.kind,
            'path': self.path,
            'extensions': list(self.extensions),
            'scan': self.scan,
        }

    @property
    def key(self) -> str:
        """Identity of the source: the same folder (or manifest) and extensions, or the
        same registered source."""
        if self.kind == 'queue':
            return f'queue:{self.path}'
        return f'{self.kind}:{self.path}:{",".join(sorted(self.extensions))}'


def normalise_extensions(extensions: list[str] | None) -> tuple[str, ...]:
    if not extensions:
        return DEFAULT_EXTENSIONS
    return tuple(sorted({f'.{e.strip().lstrip(".").lower()}' for e in extensions if e.strip()}))


def validate_source(source: BulkSource) -> None:
    """Raise :class:`InvalidSourceError` unless the source exists, is the right
    kind of path and lies under a configured source root (a queue source: unless
    it is registered)."""
    if source.kind == 'queue':
        if load_source(source.path) is None:
            raise InvalidSourceError(f'no ingest source {source.path!r}')
        return
    if not is_servable_image_path(source.path):
        raise InvalidSourceError(f'{source.path}: {UNSERVABLE_PATH_ERROR}')
    path = Path(source.path)
    if source.kind == 'folder' and not path.is_dir():
        raise InvalidSourceError(f'{source.path}: not a directory')
    if source.kind == 'folder':
        # A walk lists an unreadable directory as empty, so a run would "complete" with
        # nothing discovered; refuse the unreadable root up front instead.
        try:
            os.scandir(path).close()
        except OSError as exc:
            raise InvalidSourceError(f'{source.path}: not readable ({exc.strerror})') from None
    if source.kind == 'manifest' and not path.is_file():
        raise InvalidSourceError(f'{source.path}: not a file')


def _manifest_entry(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith('#'):
        return None
    if text.startswith('{'):
        try:
            value = json.loads(text).get('path')
        except (ValueError, AttributeError):
            return text
        return value if isinstance(value, str) and value else None
    return text


def iter_manifest_paths(path: Path) -> Iterator[str]:
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            entry = _manifest_entry(line)
            if entry is not None:
                yield entry


def queue_ref(defn: IngestSource, entry: QueueEntry) -> str:
    """The string a run carries for a queue entry: an object reference or a path."""
    if defn.backend == 's3':
        return ObjectRef(defn.bucket, entry.key, entry.etag, entry.version, entry.size).to_uri()
    return entry.key


def _queue_refs(source: BulkSource, should_stop: Callable[[], bool]) -> Iterator[str]:
    defn = load_source(source.path)
    if defn is None:
        raise InvalidSourceError(f'no ingest source {source.path!r}')
    if source.scan:
        try:
            scan_source(defn, should_stop=should_stop)
        except ScanStoppedError:
            return
    for entry in source_queue(defn.source_id).iter_pending():
        yield queue_ref(defn, entry)


def iter_source_paths(
    source: BulkSource,
    *,
    walk_workers: int = 8,
    should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[str]:
    """Every image path (or object reference) of ``source``, in discovery order.
    Blocking: iterate it off the event loop."""
    if source.kind == 'queue':
        yield from _queue_refs(source, should_stop)
        return
    if source.kind == 'manifest':
        yield from iter_manifest_paths(Path(source.path))
        return
    for found in iter_image_paths(Path(source.path), frozenset(source.extensions), walk_workers):
        yield str(found)
