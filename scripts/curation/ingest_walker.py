#!/usr/bin/env python3
"""Bulk directory ingest walker — the front door for pointing a new
deployment at a directory of images.

Walks a directory tree (via :mod:`_fast_walk`'s parallel ``os.scandir``),
batches file paths, and POSTs them to ``/curation/ingest/batch``. The
ingest router reads each file from disk *inside the API container* (see
``src/routers/curation/ingest.py``'s docstring — "a path already
reachable inside the container"), so this walker sends paths, not
bytes; it never opens the image files itself. That keeps the walker
lightweight enough to run from a laptop against a remote deployment as
long as both sides can resolve the same mounted path.

Resumable: every successfully-submitted batch's paths are appended to
the progress file (JSON Lines, one path per line) as soon as the batch
call returns without a network error. On restart, any path already in
the progress file is skipped before rescanning — a crashed walker
never re-POSTs work it already handed to the server. A single failed
image ingest inside an otherwise-successful batch does *not* have its
error re-driven by this walker; it is reported and left in the
`op_items`/`op_images` write path's own results.

Usage:
    python3 scripts/curation/ingest_walker.py \\
        --root /data/incoming --api-base http://localhost:4603/curation \\
        --source my_dataset --batch-size 32 --concurrency 4

    # Resume an interrupted run (same --progress-file):
    python3 scripts/curation/ingest_walker.py --root /data/incoming \\
        --progress-file /tmp/ingest_walker_progress.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.curation._fast_walk import iter_image_paths
from src.config import get_curation_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ingest_walker')

DEFAULT_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png'})


def _load_progress(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()
    done: set[str] = set()
    with progress_file.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)['path'])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_progress(progress_file: Path, paths: list[str]) -> None:
    with progress_file.open('a', encoding='utf-8') as f:
        for p in paths:
            f.write(json.dumps({'path': p}) + '\n')


def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _submit_batch(
    client: httpx.AsyncClient,
    api_base: str,
    paths: list[str],
    source: str,
) -> dict:
    body = {'items': [{'path': p, 'source': source} for p in paths]}
    resp = await client.post(f'{api_base}/ingest/batch', json=body, timeout=300.0)
    resp.raise_for_status()
    return resp.json()


async def run(
    root: Path,
    api_base: str,
    source: str,
    batch_size: int,
    concurrency: int,
    progress_file: Path,
    extensions: frozenset[str],
    walk_workers: int,
    dry_run: bool,
) -> None:
    already_done = _load_progress(progress_file)
    if already_done:
        logger.info('resuming: %d paths already recorded as submitted', len(already_done))

    all_paths = sorted(str(p) for p in iter_image_paths(root, extensions, workers=walk_workers))
    pending = [p for p in all_paths if p not in already_done]
    logger.info(
        'discovered %d files under %s (%d already submitted, %d pending)',
        len(all_paths),
        root,
        len(all_paths) - len(pending),
        len(pending),
    )

    if dry_run:
        logger.info(
            'dry-run: would submit %d files in %d batches',
            len(pending),
            -(-len(pending) // batch_size),
        )
        return

    batches = _batches(pending, batch_size)
    sem = asyncio.Semaphore(max(1, concurrency))
    totals = {'successful': 0, 'duplicates': 0, 'failed': 0}

    async with httpx.AsyncClient() as client:

        async def _one(batch: list[str]) -> None:
            async with sem:
                try:
                    result = await _submit_batch(client, api_base, batch, source)
                except httpx.HTTPError as exc:
                    logger.error('batch of %d failed to submit: %s', len(batch), exc)
                    return
                summary = result.get('summary', {})
                totals['successful'] += summary.get('successful', 0)
                totals['duplicates'] += summary.get('duplicates', 0)
                totals['failed'] += summary.get('failed', 0)
                # Record every path in this batch as submitted regardless of
                # per-image outcome (duplicate/failed are terminal server-side
                # states too) -- resuming should never re-hand the server work
                # it already evaluated.
                _append_progress(progress_file, batch)
                logger.info(
                    'batch done: %d paths -> %s',
                    len(batch),
                    summary,
                )

        await asyncio.gather(*[_one(b) for b in batches])

    logger.info('ingest walk complete: %s', totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Directory to walk')
    parser.add_argument(
        '--api-base',
        default=f'http://localhost:4603{get_curation_config().api_prefix}',
        help='Curation API base URL (no trailing slash)',
    )
    parser.add_argument(
        '--source', default='ingest_walker', help='hdd_source tag for ingested images'
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--concurrency', type=int, default=4, help='Concurrent in-flight batch POSTs'
    )
    parser.add_argument(
        '--walk-workers', type=int, default=8, help='Directory-scan thread pool size'
    )
    parser.add_argument(
        '--progress-file',
        type=Path,
        default=Path('/tmp/ingest_walker_progress.jsonl'),
        help='Resume-progress file (JSON Lines of submitted paths)',
    )
    parser.add_argument(
        '--extensions',
        default=','.join(sorted(DEFAULT_EXTENSIONS)),
        help='Comma-separated, case-insensitive file extensions to include',
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Only report what would be submitted'
    )
    args = parser.parse_args()

    extensions = frozenset(f'.{e.strip().lstrip(".").lower()}' for e in args.extensions.split(','))

    asyncio.run(
        run(
            root=args.root,
            api_base=args.api_base.rstrip('/'),
            source=args.source,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            progress_file=args.progress_file,
            extensions=extensions,
            walk_workers=args.walk_workers,
            dry_run=args.dry_run,
        )
    )


if __name__ == '__main__':
    main()
