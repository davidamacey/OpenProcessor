#!/usr/bin/env python3
"""Byte-upload bulk ingest: read images locally, upload them to the API.

The path-based walker (``ingest_walker.py``) sends file *paths*, so the
API container must already have the same storage mounted at the same
location. This driver is for everything else — a laptop, a remote or
unmountable share, a high-latency NAS: it reads the files itself and
uploads the bytes to ``POST {api_base}/ingest/upload``.

Pipeline::

    directory walk (parallel os.scandir)
          |  paths, streamed — never materialized as one list
          v
    optional POST /ingest/path_lookup   (skip identifiers already indexed)
          |
          v
    reader thread pool  --bytes-->  bounded asyncio.Queue of batches
                                              |
                                              v
                              N concurrent multipart POST /ingest/upload

The queue is bounded, so when the API is the bottleneck the readers stall
instead of buffering the whole tree in memory; with a slow disk, the
reader pool keeps ``--reader-threads`` reads in flight to hide latency.

Resume: re-run the same command. Two mechanisms, cheapest first:

1. ``/ingest/path_lookup`` drops identifiers the images index already
   has before a single byte is read (skip with ``--force-rehash`` when
   files may have changed in place).
2. The server fingerprints every uploaded image (imohash) and returns
   ``duplicate`` without inference for content it has already ingested —
   so even with a different identifier, or after a crash mid-batch,
   nothing is ingested twice.

Identifiers: each image's ``image_path`` on the server is its local path,
optionally rewritten with ``--path-map LOCAL_PREFIX=ID_PREFIX`` so the
same corpus uploaded from different machines gets the same identifiers.

Usage::

    python3 scripts/curation/ingest_upload.py --root /media/archive \\
        --api-base http://api-host:4603/curation --source archive_2024

    # Stable identifiers independent of where the drive is mounted
    python3 scripts/curation/ingest_upload.py --root /media/usb0/archive \\
        --path-map /media/usb0/archive=archive:// --source archive_2024
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.curation._fast_walk import iter_image_paths
from src.config import get_curation_config


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator


logger = logging.getLogger('ingest_upload')

DEFAULT_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png'})
PATH_LOOKUP_CHUNK = 2000
MAX_BATCH = 128  # server-side MAX_UPLOAD_IMAGES


@dataclass
class UploadConfig:
    api_base: str
    source: str = 'upload'
    batch_size: int = 32
    reader_threads: int = 16
    submit_concurrency: int = 4
    queue_size: int = 16
    path_lookup: bool = True
    path_map: tuple[str, str] | None = None
    retries: int = 3
    retry_backoff_s: float = 2.0
    request_timeout_s: float = 300.0
    progress_file: Path | None = None
    failed_log: Path | None = None
    limit: int | None = None


@dataclass
class Counters:
    walked: int = 0
    skipped_known: int = 0
    read_failed: int = 0
    uploaded: int = 0
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    crops_indexed: int = 0
    started: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        elapsed = max(1e-6, time.monotonic() - self.started)
        out = {k: v for k, v in self.__dict__.items() if k != 'started'}
        out['elapsed_s'] = round(elapsed, 1)
        out['upload_rate_img_s'] = round(self.uploaded / elapsed, 2)
        return out


def map_identifier(path: Path, path_map: tuple[str, str] | None) -> str:
    """Local path -> the ``image_path`` identifier stored server-side."""
    text = str(path)
    if path_map is None:
        return text
    local, remote = path_map
    if text != local and not text.startswith(local.rstrip('/') + '/'):
        return text
    rel = text[len(local) :].lstrip('/')
    if not rel:
        return remote
    return f'{remote}{rel}' if remote.endswith('/') else f'{remote}/{rel}'


def parse_path_map(spec: str) -> tuple[str, str]:
    local, sep, remote = spec.partition('=')
    if not sep or not local or not remote:
        raise argparse.ArgumentTypeError(f'--path-map expects LOCAL=REMOTE, got {spec!r}')
    return local.rstrip('/') or '/', remote


def _take(paths: Iterator[Path], n: int) -> list[Path]:
    return list(itertools.islice(paths, n))


def _read(path: Path) -> tuple[Path, bytes | None, str | None]:
    try:
        return path, path.read_bytes(), None
    except OSError as exc:
        return path, None, str(exc)


def _append_jsonl(path: Path | None, rows: Iterable[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')


class UploadRunner:
    """Walk -> (path_lookup) -> threaded reads -> bounded queue -> concurrent uploads."""

    def __init__(self, cfg: UploadConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = client
        self.counts = Counters()
        self._last_progress = 0.0

    # -------------------------------------------------------------- server

    async def _known_identifiers(self, ids: list[str]) -> set[str]:
        if not self.cfg.path_lookup or not ids:
            return set()
        try:
            resp = await self.client.post(
                f'{self.cfg.api_base}/ingest/path_lookup',
                json={'image_paths': ids},
                timeout=60.0,
            )
            resp.raise_for_status()
            return set((resp.json().get('known_paths') or {}).keys())
        except httpx.HTTPError as exc:
            # Safe to degrade: server-side content dedup still catches
            # anything already ingested, just after the read + upload.
            logger.warning('path_lookup failed (%s); relying on server content dedup', exc)
            return set()

    async def _post_batch(self, batch: list[tuple[str, bytes]]) -> dict[str, Any] | None:
        files = [('images', (ident.rsplit('/', 1)[-1] or 'image', data)) for ident, data in batch]
        form = {'image_paths': json.dumps([ident for ident, _ in batch]), 'source': self.cfg.source}
        for attempt in range(1, self.cfg.retries + 1):
            try:
                resp = await self.client.post(
                    f'{self.cfg.api_base}/ingest/upload',
                    files=files,
                    data=form,
                    timeout=self.cfg.request_timeout_s,
                )
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                logger.warning('upload HTTP %d (attempt %d)', resp.status_code, attempt)
            except httpx.HTTPStatusError as exc:
                # 4xx is not transient — retrying the same payload won't help.
                logger.error('upload rejected: %s %s', exc, exc.response.text[:300])
                return None
            except httpx.HTTPError as exc:
                logger.warning('upload error (attempt %d): %s', attempt, exc)
            if attempt < self.cfg.retries:
                await asyncio.sleep(self.cfg.retry_backoff_s * attempt)
        return None

    # ------------------------------------------------------------ pipeline

    async def _submitter(self, queue: asyncio.Queue[list[tuple[str, bytes]] | None]) -> None:
        while True:
            batch = await queue.get()
            try:
                if batch is None:
                    return
                try:
                    result = await self._post_batch(batch)
                except Exception as exc:
                    # Anything unexpected (e.g. a non-JSON 2xx body) must not
                    # kill this task: the bounded queue would then fill and
                    # deadlock the reader. Counted + logged as a failed batch.
                    logger.error('upload batch crashed: %r', exc)
                    result = None
                self.counts.uploaded += len(batch)
                if result is None:
                    self.counts.failed += len(batch)
                    _append_jsonl(
                        self.cfg.failed_log,
                        ({'image_path': ident, 'error': 'upload failed'} for ident, _ in batch),
                    )
                else:
                    summary = result.get('summary') or {}
                    self.counts.successful += int(summary.get('successful', 0))
                    self.counts.duplicates += int(summary.get('duplicates', 0))
                    self.counts.failed += int(summary.get('failed', 0))
                    self.counts.crops_indexed += int(summary.get('crops_indexed', 0))
                    _append_jsonl(
                        self.cfg.failed_log,
                        (
                            {'image_path': r.get('image_path'), 'error': r.get('error')}
                            for r in result.get('results') or []
                            if r.get('status') == 'failed'
                        ),
                    )
                self._maybe_write_progress()
            finally:
                queue.task_done()

    async def _walk_chunks(self, paths: Iterator[Path]) -> AsyncIterator[list[Path]]:
        """Pull the (blocking) walker in chunks off the event loop."""
        loop = asyncio.get_running_loop()
        remaining = self.cfg.limit
        while remaining is None or remaining > 0:
            n = PATH_LOOKUP_CHUNK if remaining is None else min(PATH_LOOKUP_CHUNK, remaining)
            chunk: list[Path] = await loop.run_in_executor(None, _take, paths, n)
            if not chunk:
                return
            if remaining is not None:
                remaining -= len(chunk)
            yield chunk

    async def _reader(
        self,
        paths: Iterator[Path],
        pool: ThreadPoolExecutor,
        queue: asyncio.Queue[list[tuple[str, bytes]] | None],
    ) -> None:
        loop = asyncio.get_running_loop()
        inflight: set[asyncio.Future[tuple[Path, bytes | None, str | None]]] = set()
        batch: list[tuple[str, bytes]] = []
        max_inflight = max(1, self.cfg.reader_threads * 2)

        async def _collect(done: set[asyncio.Future[tuple[Path, bytes | None, str | None]]]):
            nonlocal batch
            for fut in done:
                path, data, err = fut.result()
                if data is None:
                    self.counts.read_failed += 1
                    logger.warning('read failed %s: %s', path, err)
                    _append_jsonl(self.cfg.failed_log, [{'image_path': str(path), 'error': err}])
                    continue
                batch.append((map_identifier(path, self.cfg.path_map), data))
                if len(batch) >= self.cfg.batch_size:
                    # Blocks when the queue is full: backpressure on reads.
                    await queue.put(batch)
                    batch = []

        async for chunk in self._walk_chunks(paths):
            self.counts.walked += len(chunk)
            known = await self._known_identifiers(
                [map_identifier(p, self.cfg.path_map) for p in chunk]
            )
            for path in chunk:
                if map_identifier(path, self.cfg.path_map) in known:
                    self.counts.skipped_known += 1
                    continue
                while len(inflight) >= max_inflight:
                    done, inflight = await asyncio.wait(
                        inflight, return_when=asyncio.FIRST_COMPLETED
                    )
                    await _collect(done)
                inflight.add(loop.run_in_executor(pool, _read, path))
        while inflight:
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            await _collect(done)
        if batch:
            await queue.put(batch)

    async def run(self, paths: Iterable[Path]) -> Counters:
        queue: asyncio.Queue[list[tuple[str, bytes]] | None] = asyncio.Queue(
            maxsize=max(1, self.cfg.queue_size)
        )
        submitters = [
            asyncio.create_task(self._submitter(queue))
            for _ in range(max(1, self.cfg.submit_concurrency))
        ]
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.cfg.reader_threads)) as pool:
                await self._reader(iter(paths), pool, queue)
        finally:
            for _ in submitters:
                await queue.put(None)
            await asyncio.gather(*submitters)
        self._write_progress(final=True)
        return self.counts

    # ------------------------------------------------------------ progress

    def _maybe_write_progress(self) -> None:
        now = time.monotonic()
        if now - self._last_progress >= 10.0:
            self._last_progress = now
            logger.info('progress %s', self.counts.as_dict())
            self._write_progress(final=False)

    def _write_progress(self, *, final: bool) -> None:
        if self.cfg.progress_file is None:
            return
        self.cfg.progress_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'updated_at': datetime.now(UTC).isoformat(),
            'final': final,
            'source': self.cfg.source,
            'counts': self.counts.as_dict(),
        }
        tmp = self.cfg.progress_file.with_suffix(self.cfg.progress_file.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        tmp.replace(self.cfg.progress_file)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--root', required=True, type=Path, help='Directory to walk')
    p.add_argument(
        '--api-base',
        default=f'http://localhost:4603{get_curation_config().api_prefix}',
        help='Curation API base URL including the api prefix (no trailing slash)',
    )
    p.add_argument('--source', default='upload', help='Provenance tag for every image')
    p.add_argument('--batch-size', type=int, default=32, help=f'Images per upload (<= {MAX_BATCH})')
    p.add_argument('--reader-threads', type=int, default=16)
    p.add_argument('--submit-concurrency', type=int, default=4)
    p.add_argument(
        '--queue-size', type=int, default=16, help='Max read-ahead batches held in memory'
    )
    p.add_argument('--walk-workers', type=int, default=8)
    p.add_argument(
        '--extensions',
        default=','.join(sorted(DEFAULT_EXTENSIONS)),
        help='Comma-separated, case-insensitive extensions',
    )
    p.add_argument(
        '--path-map',
        type=parse_path_map,
        default=None,
        metavar='LOCAL=ID_PREFIX',
        help='Rewrite a local path prefix into the stored identifier',
    )
    p.add_argument(
        '--force-rehash',
        action='store_true',
        help='Skip the /ingest/path_lookup pre-filter; read and upload everything',
    )
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--progress-file', type=Path, default=None, help='JSON progress snapshot')
    p.add_argument('--failed-log', type=Path, default=None, help='JSONL of failed images')
    p.add_argument('--limit', type=int, default=None, help='Stop after N walked files')
    return p


async def _async_main(args: argparse.Namespace) -> int:
    if not args.root.is_dir():
        logger.error('root is not a directory: %s', args.root)
        return 1
    batch_size = min(max(1, args.batch_size), MAX_BATCH)
    if batch_size != args.batch_size:
        logger.warning('batch size clamped to %d', batch_size)
    extensions = frozenset(f'.{e.strip().lstrip(".").lower()}' for e in args.extensions.split(','))
    cfg = UploadConfig(
        api_base=args.api_base.rstrip('/'),
        source=args.source,
        batch_size=batch_size,
        reader_threads=args.reader_threads,
        submit_concurrency=args.submit_concurrency,
        queue_size=args.queue_size,
        path_lookup=not args.force_rehash,
        path_map=args.path_map,
        retries=max(1, args.retries),
        progress_file=args.progress_file,
        failed_log=args.failed_log,
        limit=args.limit,
    )
    paths = iter_image_paths(args.root, extensions, workers=args.walk_workers)
    async with httpx.AsyncClient() as client:
        counts = await UploadRunner(cfg, client).run(paths)
    logger.info('done %s', counts.as_dict())
    return 0 if counts.failed == 0 and counts.read_failed == 0 else 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return asyncio.run(_async_main(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
