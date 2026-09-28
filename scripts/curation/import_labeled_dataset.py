#!/usr/bin/env python3
"""Bulk-ingest a YOLO dataset's images (``--images-only`` only).

Walks a YOLO dataset (``data.yaml`` splits, or ``images/<split>`` /
``<split>/images`` directories) and, per split, sends every image to
``POST {api_base}/ingest/batch``. The server ingests the image (detector +
embeddings). Use it when the dataset's labels are not item classes to
import — e.g. whole frames labeled with the *region* class
(``names: {0: defect}``) that should be checked against the region cascade
afterwards with ``eval_regions_vs_gt.py``, not imported into the item
registry.

**Labeled-import mode (posting ground-truth boxes as validated item labels,
and ``--relabel-duplicates``) is currently disabled.** It depended on
``/ingest/batch`` label fields and ``/import_labels/batch``, both removed
from this repo's API surface; the planned replacement,
``POST /datasets/imports`` fronting ``dataset_import.import_dataset()``, is
not built yet (W10 Opus review 2026-09-28, finding M1). Passing anything
other than ``--images-only`` fails immediately with a clear error — pass
``--images-only``, or call ``src.services.curation.dataset_import.job.import_dataset()``
directly for a labeled import today.

``--images-only`` behavior: no registry class check, no label import, no
disagreement report. Resume, checkpoints, ``--limit`` (stratified by
positive = non-empty label file), ``--seed``, ``--splits`` and
``--path-map`` all still apply, and local label-file stats (positives,
backgrounds, label row counts) are still collected and reported for context.

Resume, at two levels, under ``--state-dir``:

* ``checkpoints/<split>.json`` — a finished split is skipped on re-run
  (``--force`` redoes it); its counts still roll into the summary.
* ``progress/<split>.jsonl`` — one line per completed batch (paths +
  counts), so an interrupted split resumes where it stopped.

Server-side content dedup additionally makes a re-sent image a cheap
``duplicate``.

Every run writes ``ingested/<split>.jsonl`` under ``--state-dir``: one line
per image that landed (``image``, ``server_path``, ``image_id``,
``status``, ``positive``). ``eval_regions_vs_gt.py --state-dir`` reads it
as its cohort.

Usage::

    # Preview: discovered splits, positives/backgrounds
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/ds/data.yaml \\
        --api-base http://localhost:4603/curation --path-map /data/ds=/datasets/ds \\
        --images-only --dry-run

    # Region ground truth: ingest images only, then evaluate the region cascade
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/regions/data.yaml \\
        --images-only --splits test --limit 1000 --state-dir ./state/regions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.curation.ingest_upload import map_identifier, parse_path_map
from scripts.curation.yolo_dataset import (
    DatasetError,
    Sample,
    discover,
    label_path_for,  # noqa: F401 - public re-export (discovery moved to yolo_dataset)
    load_samples,
    stratified_sample,
)
from src.config import get_curation_config
from src.services.projects.script_binding import add_project_argument, bind_script_project


logger = logging.getLogger('import_labeled_dataset')

COUNT_KEYS = (
    'images',
    'positives',
    'backgrounds',
    'label_rows',
    'label_files_missing',
    'successful',
    'duplicates',
    'failed',
)


# =============================================================================
# Import
# =============================================================================


@dataclass
class ImportConfig:
    api_base: str
    state_dir: Path
    source_prefix: str
    path_map: tuple[str, str] | None = None
    batch_size: int = 32
    concurrency: int = 4
    images_only: bool = False
    force: bool = False
    retries: int = 3
    retry_backoff_s: float = 2.0
    timeout_s: float = 600.0


def _zero() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def _add(into: dict[str, int], other: dict[str, Any]) -> None:
    for k in COUNT_KEYS:
        into[k] += int(other.get(k, 0))


class DatasetImporter:
    """Images-only dataset ingest driver. Labeled-import mode is disabled --
    see the module docstring."""

    def __init__(self, cfg: ImportConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = client
        for sub in ('checkpoints', 'progress', 'ingested'):
            (cfg.state_dir / sub).mkdir(parents=True, exist_ok=True)

    def server_path(self, local: Path) -> str:
        return map_identifier(local, self.cfg.path_map)

    # ------------------------------------------------------------- server

    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any] | None:
        for attempt in range(1, self.cfg.retries + 1):
            try:
                resp = await self.client.post(url, json=body, timeout=self.cfg.timeout_s)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                logger.warning('%s -> HTTP %d (attempt %d)', url, resp.status_code, attempt)
            except httpx.HTTPStatusError as exc:
                logger.error('%s rejected: %s', url, exc.response.text[:300])
                return None
            except httpx.HTTPError as exc:
                logger.warning('%s failed (attempt %d): %s', url, attempt, exc)
            if attempt < self.cfg.retries:
                await asyncio.sleep(self.cfg.retry_backoff_s * attempt)
        return None

    async def import_batch(self, split: str, batch: list[Sample]) -> dict[str, Any] | None:
        """One ``/ingest/batch`` call (images only). Returns this batch's
        counts, or None on failure."""
        by_server = {self.server_path(s.image): s for s in batch}
        items: list[dict[str, Any]] = [
            {'path': self.server_path(s.image), 'source': f'{self.cfg.source_prefix}:{split}'}
            for s in batch
        ]
        result = await self._post(f'{self.cfg.api_base}/ingest/batch', {'items': items})
        if result is None:
            return None

        counts = _zero()
        summary = result.get('summary') or {}
        for key in ('successful', 'duplicates', 'failed'):
            counts[key] = int(summary.get(key, 0))
        rows = {r.get('image_path'): r for r in result.get('results') or []}
        status = {p: r.get('status') for p, r in rows.items()}
        landed = [(p, s) for p, s in by_server.items() if status.get(p) in ('success', 'duplicate')]
        return {
            'paths': [str(s.image) for _p, s in landed],
            # The evaluator's cohort: a duplicate's items live under the
            # *first* copy's image_id, so the id is what joins back to them.
            'ingested': [
                {
                    'image': str(s.image),
                    'server_path': p,
                    'image_id': rows[p].get('image_id') or None,
                    'status': status[p],
                    'positive': s.positive,
                }
                for p, s in landed
            ],
            'counts': counts,
        }

    # -------------------------------------------------------------- split

    def _load_progress(self, split: str) -> tuple[set[str], dict[str, int]]:
        done: set[str] = set()
        counts = _zero()
        path = self.cfg.state_dir / 'progress' / f'{split}.jsonl'
        if path.exists() and not self.cfg.force:
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                done.update(row['paths'])
                _add(counts, row['counts'])
        elif path.exists():
            path.unlink()
        return done, counts

    def write_ingested_list(self, split: str) -> Path:
        """Rebuild ``ingested/<split>.jsonl`` from the progress file.

        Derived from progress (not appended per batch) so a resumed or
        checkpoint-skipped split still yields its complete cohort.
        """
        out = self.cfg.state_dir / 'ingested' / f'{split}.jsonl'
        progress = self.cfg.state_dir / 'progress' / f'{split}.jsonl'
        seen: set[str] = set()
        lines: list[str] = []
        if progress.exists():
            for line in progress.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                entries = row.get('ingested') or [{'image': p} for p in row['paths']]
                for entry in entries:
                    if entry['image'] not in seen:
                        seen.add(entry['image'])
                        lines.append(json.dumps(entry))
        out.write_text(''.join(f'{ln}\n' for ln in lines), encoding='utf-8')
        return out

    async def run_split(self, split: str, samples: list[Sample]) -> dict[str, int]:
        counts = await self._run_split(split, samples)
        self.write_ingested_list(split)
        return counts

    async def _run_split(self, split: str, samples: list[Sample]) -> dict[str, int]:
        ckpt = self.cfg.state_dir / 'checkpoints' / f'{split}.json'
        if ckpt.exists() and not self.cfg.force:
            logger.info('[%s] checkpoint exists; skipping (use --force to redo)', split)
            return json.loads(ckpt.read_text(encoding='utf-8'))['counts']

        done, counts = self._load_progress(split)
        pending = [s for s in samples if str(s.image) not in done]
        counts['images'] = len(samples)
        counts['positives'] = sum(1 for s in samples if s.positive)
        counts['backgrounds'] = counts['images'] - counts['positives']
        counts['label_rows'] = sum(s.n_labels for s in samples)
        counts['label_files_missing'] = sum(1 for s in samples if not s.label_exists)
        logger.info(
            '[%s] %d images (%d positive, %d background), %d already done',
            split,
            len(samples),
            counts['positives'],
            counts['backgrounds'],
            len(samples) - len(pending),
        )

        progress_path = self.cfg.state_dir / 'progress' / f'{split}.jsonl'
        batches = [
            pending[i : i + self.cfg.batch_size]
            for i in range(0, len(pending), self.cfg.batch_size)
        ]
        started = time.monotonic()

        async def _process(batch: list[Sample]) -> None:
            out = await self.import_batch(split, batch)
            if out is None:
                counts['failed'] += len(batch)
                return
            _add(counts, out['counts'])
            with progress_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(out) + '\n')
            handled = counts['successful'] + counts['duplicates'] + counts['failed']
            rate = handled / max(1e-6, time.monotonic() - started)
            logger.info('[%s] %d handled (%.1f img/s)', split, handled, rate)

        async def _worker() -> None:
            while batches:
                await _process(batches.pop(0))

        await asyncio.gather(*[_worker() for _ in range(max(1, self.cfg.concurrency))])

        # Failed images are not in the progress file, so a re-run retries
        # them; only a split with no failures is checkpointed as complete.
        if counts['failed'] == 0:
            ckpt.write_text(
                json.dumps(
                    {
                        'split': split,
                        'completed_at': datetime.now(UTC).isoformat(),
                        'counts': counts,
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )
        return counts


def summarize(per_split: dict[str, dict[str, int]]) -> dict[str, Any]:
    total = _zero()
    for c in per_split.values():
        _add(total, c)
    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'splits': dict(per_split),
        'total': total,
    }


async def run(
    cfg: ImportConfig,
    client: httpx.AsyncClient,
    splits: dict[str, list[Sample]],
) -> dict[str, Any]:
    importer = DatasetImporter(cfg, client)
    per_split: dict[str, dict[str, int]] = {}
    for split, samples in splits.items():
        per_split[split] = await importer.run_split(split, samples)
    summary = summarize(per_split)
    (cfg.state_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--dataset', required=True, type=Path, help='data.yaml or dataset root')
    p.add_argument(
        '--api-base',
        default=f'http://localhost:4603{get_curation_config().api_prefix}',
        help='Curation API mount; requests go to <api-base>/projects/<--project>/...',
    )
    p.add_argument('--splits', default=None, help='Comma-separated subset (default: all found)')
    p.add_argument(
        '--path-map',
        type=parse_path_map,
        default=None,
        metavar='LOCAL=SERVER',
        help='Rewrite the local dataset prefix to the path the API container sees',
    )
    p.add_argument(
        '--source-prefix', default=None, help='Image source tag prefix (<prefix>:<split>)'
    )
    p.add_argument('--state-dir', type=Path, default=None, help='Checkpoints, progress, report')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=4, help='Concurrent in-flight batches')
    p.add_argument('--limit', type=int, default=None, help='Stratified sample of N per split')
    p.add_argument('--seed', type=int, default=0, help='Seed for --limit sampling')
    p.add_argument(
        '--images-only',
        action='store_true',
        help='Ingest the dataset images without importing their labels. REQUIRED today -- '
        'labeled-import mode is disabled (see module docstring); any run without this flag '
        'fails immediately.',
    )
    p.add_argument('--force', action='store_true', help='Redo splits that have checkpoints')
    p.add_argument('--dry-run', action='store_true', help='Discover only')
    add_project_argument(p)
    return p


async def _async_main(args: argparse.Namespace) -> int:
    if not args.images_only:
        # W10 (Opus review 2026-09-28, finding M1): labeled mode posted
        # forbidden fields (label_txt_path/detect_mismatches) to
        # /ingest/batch (IngestBatchRequest is extra='forbid' -- every
        # batch 422s) and relabel-duplicates posted to the deleted
        # /import_labels/batch (404). Neither surface exists anymore;
        # dataset_import's Python API (import_dataset()) has no HTTP
        # route yet to front it (planned: POST /datasets/imports), so the
        # labeled-mode code was deleted rather than kept unreachable. Fail
        # loudly and immediately here -- before any dataset discovery,
        # project binding, or HTTP call -- instead of erroring deep in a
        # request with no clear signal to the operator.
        logger.error(
            'Labeled import mode is not available: it posted to routes this repo removed '
            "(/ingest/batch's label fields, /import_labels/batch), and the replacement "
            '(POST /datasets/imports, fronting dataset_import.import_dataset()) is not built '
            'yet. Pass --images-only to ingest images without labels, or call '
            'src.services.curation.dataset_import.job.import_dataset() directly for a labeled '
            'import today.'
        )
        return 1
    try:
        found, _names = discover(args.dataset)
    except DatasetError as exc:
        logger.error('%s', exc)
        return 1
    wanted = [s.strip() for s in args.splits.split(',')] if args.splits else list(found)
    missing = [s for s in wanted if s not in found]
    if missing:
        logger.error('splits not in dataset: %s (found: %s)', missing, sorted(found))
        return 1
    splits = {}
    for name in wanted:
        samples = load_samples(found[name])
        if args.limit is not None:
            samples = stratified_sample(samples, args.limit, args.seed)
        splits[name] = samples
    dataset_root = args.dataset.parent if args.dataset.is_file() else args.dataset
    cfg = ImportConfig(
        api_base=f'{args.api_base.rstrip("/")}/projects/{args.project}',
        state_dir=args.state_dir or Path('dataset_import_state') / dataset_root.name,
        source_prefix=args.source_prefix or dataset_root.name,
        path_map=args.path_map,
        batch_size=max(1, args.batch_size),
        concurrency=max(1, args.concurrency),
        images_only=args.images_only,
        force=args.force,
    )
    for name, samples in splits.items():
        pos = sum(1 for s in samples if s.positive)
        logger.info(
            '%s: %d images, %d positive, %d background', name, len(samples), pos, len(samples) - pos
        )
    if args.dry_run:
        return 0
    async with httpx.AsyncClient() as client:
        summary = await run(cfg, client, splits)
    logger.info('summary: %s', json.dumps(summary['total']))
    logger.info('ingested image lists: %s', cfg.state_dir / 'ingested')
    return 0 if summary['total']['failed'] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    # The disabled-labeled-mode guard inside _async_main needs no project
    # binding (no OpenSearch connection) to fire -- skip bind_script_project
    # (which does contact OpenSearch to resolve the project) when it is
    # about to fail loudly anyway, so the failure is immediate/cheap.
    if args.images_only:
        bind_script_project(args.project)
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    sys.exit(main())
