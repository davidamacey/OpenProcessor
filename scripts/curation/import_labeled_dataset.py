#!/usr/bin/env python3
"""Bulk-import an existing YOLO-labeled dataset and report where the model disagrees.

Walks a YOLO dataset (``data.yaml`` splits, or ``images/<split>`` /
``<split>/images`` directories) and, per split, sends every image together
with its paired ``.txt`` to ``POST {api_base}/ingest/batch``
(``label_txt_path`` + ``label_source`` + ``detect_mismatches``). The server
ingests the image (detector + embeddings), imports the ground-truth boxes as
validated labels, and IoU-matches them against the detector's own boxes.
Every disagreement comes back and is written to a JSONL report:

* ``class_mismatch`` — same box, different class.
* ``missed_label`` — a labeled object no detector box overlaps.
* ``unmatched_detection`` — a detector box no label overlaps. On a background
  image (empty or absent ``.txt``) every detection is one — that is the
  false-positive signal for single-class datasets with hard negatives.

The server reads the images and labels itself, so it must see the dataset:
``--path-map LOCAL=SERVER`` rewrites the local dataset prefix to where the API
container mounts it. Before anything is written the driver checks (a) that
the dataset's ``names`` agree id-for-id with the server's class registry — a
label ``0`` imported into a registry whose id 0 is another class would
silently corrupt training data — and (b) on the first batch with labels,
that the server actually imported some (else the label files are not
visible server-side and the run aborts instead of recording every image as
a background).

Resume, at two levels, under ``--state-dir``:

* ``checkpoints/<split>.json`` — a finished split is skipped on re-run
  (``--force`` redoes it); its counts still roll into the summary.
* ``progress/<split>.jsonl`` — one line per completed batch (paths +
  counts), so an interrupted split resumes where it stopped.

Server-side content dedup additionally makes a re-sent image a cheap
``duplicate``. Note a duplicate's labels are *not* re-imported (they were
imported with it the first time); ``--relabel-duplicates`` sends them through
``POST /import_labels/batch`` for images first ingested without labels.

``--images-only`` ingests the images without their labels: no
``label_txt_path``, no registry class check, no label checks. Use it when
the dataset's labels are not item classes — e.g. whole frames labeled with
the *region* class (``names: {0: license_plate}``) that should be checked
against the region cascade, not imported into the item registry.
Resume, checkpoints, ``--limit`` (still stratified by positive = non-empty
label file), ``--seed``, ``--splits`` and ``--path-map`` behave as usual.

Every run writes ``ingested/<split>.jsonl`` under ``--state-dir``: one line
per image that landed (``image``, ``server_path``, ``image_id``,
``status``, ``positive``). ``eval_regions_vs_gt.py --state-dir`` reads it
as its cohort.

Usage::

    # Preview: discovered splits, positives/backgrounds, class check
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/ds/data.yaml \\
        --api-base http://localhost:4603/curation --path-map /data/ds=/datasets/ds --dry-run

    # Smoke cohort: 500 images from the test split, stratified positives/backgrounds
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/ds/data.yaml \\
        --path-map /data/ds=/datasets/ds --splits test --limit 500 --state-dir ./state/smoke

    # Region ground truth: ingest images only, then evaluate the region cascade
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/regions/data.yaml \\
        --images-only --splits test --limit 1000 --state-dir ./state/regions

    # Full run (re-run the same command to resume)
    python3 scripts/curation/import_labeled_dataset.py --dataset /data/ds/data.yaml \\
        --path-map /data/ds=/datasets/ds --label-source dataset_v1 --state-dir ./state/full
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
    'label_rows_on_ingested',
    'labels_imported',
    'mismatches',
    'missed_labels',
    'unmatched_detections',
    'backgrounds_with_detections',
    'relabeled',
)


class PreflightError(RuntimeError):
    """A check that must pass before (or early in) an import failed."""


# =============================================================================
# Import
# =============================================================================


@dataclass
class ImportConfig:
    api_base: str
    state_dir: Path
    source_prefix: str
    path_map: tuple[str, str] | None = None
    label_source: str | None = None
    detect_mismatches: bool = True
    batch_size: int = 32
    concurrency: int = 4
    relabel_duplicates: bool = False
    verify_labels: bool = True
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
    def __init__(self, cfg: ImportConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = client
        self.labels_verified = not cfg.verify_labels or cfg.images_only
        self.report_path = cfg.state_dir / 'disagreements.jsonl'
        for sub in ('checkpoints', 'progress', 'ingested'):
            (cfg.state_dir / sub).mkdir(parents=True, exist_ok=True)

    def server_path(self, local: Path) -> str:
        return map_identifier(local, self.cfg.path_map)

    # ---------------------------------------------------------- preflight

    async def check_classes(self, names: list[str]) -> None:
        resp = await self.client.get(f'{self.cfg.api_base}/classes', timeout=60.0)
        resp.raise_for_status()
        registry = {int(c['class_id']): c for c in resp.json().get('classes', [])}
        problems = []
        for i, name in enumerate(names):
            entry = registry.get(i)
            if entry is None:
                problems.append(f'id {i} ({name!r}) is not in the server registry')
            elif entry.get('class_name') != name:
                problems.append(f'id {i}: dataset {name!r} vs registry {entry.get("class_name")!r}')
            elif entry.get('deprecated'):
                problems.append(f'id {i} ({name!r}) is deprecated in the server registry')
        if problems:
            raise PreflightError(
                'dataset class ids do not match the server class registry:\n  '
                + '\n  '.join(problems)
            )

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

    async def _relabel(self, samples: list[Sample]) -> tuple[int, list[dict[str, Any]], int]:
        items = [
            {
                'image_path': self.server_path(s.image),
                'label_txt_path': self.server_path(s.label),
                'detect_mismatches': self.cfg.detect_mismatches,
                **({'label_source': self.cfg.label_source} if self.cfg.label_source else {}),
            }
            for s in samples
        ]
        result = await self._post(f'{self.cfg.api_base}/import_labels/batch', {'items': items})
        if result is None:
            return 0, [], 0
        return int(result.get('labels_imported', 0)), result.get('disagreements') or [], len(items)

    async def import_batch(self, split: str, batch: list[Sample]) -> dict[str, Any] | None:
        """One ``/ingest/batch`` call. Returns this batch's counts, or None on failure."""
        by_server = {self.server_path(s.image): s for s in batch}
        items: list[dict[str, Any]] = [
            {'path': self.server_path(s.image), 'source': f'{self.cfg.source_prefix}:{split}'}
            for s in batch
        ]
        body: dict[str, Any] = {'items': items}
        if not self.cfg.images_only:
            for item, s in zip(items, batch, strict=True):
                item['label_txt_path'] = self.server_path(s.label)
            body['detect_mismatches'] = self.cfg.detect_mismatches
            if self.cfg.label_source:
                body['label_source'] = self.cfg.label_source
        result = await self._post(f'{self.cfg.api_base}/ingest/batch', body)
        if result is None:
            return None

        counts = _zero()
        summary = result.get('summary') or {}
        for key in (
            'successful',
            'duplicates',
            'failed',
            'labels_imported',
            'mismatches',
            'missed_labels',
            'unmatched_detections',
        ):
            counts[key] = int(summary.get(key, 0))
        rows = {r.get('image_path'): r for r in result.get('results') or []}
        status = {p: r.get('status') for p, r in rows.items()}
        ingested = [s for p, s in by_server.items() if status.get(p) == 'success']
        if not self.cfg.images_only:
            counts['label_rows_on_ingested'] = sum(s.n_labels for s in ingested)

        if not self.labels_verified and counts['label_rows_on_ingested'] > 0:
            if counts['labels_imported'] == 0:
                raise PreflightError(
                    f'server imported 0 of {counts["label_rows_on_ingested"]} label rows in the '
                    f'first labeled batch — it cannot read the label files (check --path-map; '
                    f'e.g. {self.server_path(ingested[0].label)}) or the class ids are out of '
                    'range for its registry'
                )
            self.labels_verified = True

        records = list(result.get('disagreements') or [])
        duplicates = [s for p, s in by_server.items() if status.get(p) == 'duplicate']
        if self.cfg.relabel_duplicates and duplicates:
            imported, extra, n = await self._relabel(duplicates)
            counts['labels_imported'] += imported
            counts['relabeled'] += n
            records.extend(extra)
            for r in extra:
                kind_key = {
                    'class_mismatch': 'mismatches',
                    'missed_label': 'missed_labels',
                    'unmatched_detection': 'unmatched_detections',
                }.get(str(r.get('kind')))
                if kind_key:
                    counts[kind_key] += 1

        backgrounds = {p for p, s in by_server.items() if not s.positive}
        counts['backgrounds_with_detections'] = len(
            {
                r.get('image_path')
                for r in records
                if r.get('kind') == 'unmatched_detection' and r.get('image_path') in backgrounds
            }
        )
        if records:
            with self.report_path.open('a', encoding='utf-8') as fh:
                for r in records:
                    sample = by_server.get(str(r.get('image_path')))
                    row = {'split': split, **r}
                    if sample is not None:
                        row['local_image_path'] = str(sample.image)
                        row['background'] = not sample.positive
                    fh.write(json.dumps(row) + '\n')
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
        if counts['labels_imported'] > 0:
            self.labels_verified = True
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

        # Until one labeled batch has proven the server can read the label
        # files, go one batch at a time: a bad --path-map then aborts after
        # a single batch instead of --concurrency of them.
        while batches and not self.labels_verified:
            await _process(batches.pop(0))

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

    def _derived(c: dict[str, int]) -> dict[str, Any]:
        rows = c['label_rows_on_ingested']
        return {
            **c,
            # Share of ground-truth boxes some detector box overlapped at the
            # label importer's IoU threshold — a plumbing-level recall proxy.
            'label_match_rate': round(1 - c['missed_labels'] / rows, 4) if rows else None,
        }

    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'splits': {k: _derived(v) for k, v in per_split.items()},
        'total': _derived(total),
    }


async def run(
    cfg: ImportConfig,
    client: httpx.AsyncClient,
    splits: dict[str, list[Sample]],
    names: list[str] | None,
    *,
    check_classes: bool = True,
) -> dict[str, Any]:
    importer = DatasetImporter(cfg, client)
    if check_classes and not cfg.images_only:
        if names is None:
            logger.warning('dataset declares no class names; skipping the registry check')
        else:
            await importer.check_classes(names)
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
    p.add_argument('--api-base', default='http://localhost:4603/curation')
    p.add_argument('--splits', default=None, help='Comma-separated subset (default: all found)')
    p.add_argument(
        '--path-map',
        type=parse_path_map,
        default=None,
        metavar='LOCAL=SERVER',
        help='Rewrite the local dataset prefix to the path the API container sees',
    )
    p.add_argument('--label-source', default=None, help='label_source stamped on imported labels')
    p.add_argument(
        '--source-prefix', default=None, help='Image source tag prefix (<prefix>:<split>)'
    )
    p.add_argument('--state-dir', type=Path, default=None, help='Checkpoints, progress, report')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=4, help='Concurrent in-flight batches')
    p.add_argument('--limit', type=int, default=None, help='Stratified sample of N per split')
    p.add_argument('--seed', type=int, default=0, help='Seed for --limit sampling')
    p.add_argument('--no-detect-mismatches', action='store_true')
    p.add_argument('--relabel-duplicates', action='store_true')
    p.add_argument(
        '--images-only',
        action='store_true',
        help='Ingest the images without importing their labels (no registry check). For '
        'datasets whose labels are a different taxonomy, e.g. region-level ground truth '
        'checked afterwards with eval_regions_vs_gt.py',
    )
    p.add_argument('--skip-class-check', action='store_true')
    p.add_argument(
        '--no-verify-labels',
        action='store_true',
        help='Do not abort when the first labeled batch imports zero labels',
    )
    p.add_argument('--force', action='store_true', help='Redo splits that have checkpoints')
    p.add_argument('--dry-run', action='store_true', help='Discover and check only')
    return p


async def _async_main(args: argparse.Namespace) -> int:
    try:
        found, names = discover(args.dataset)
    except DatasetError as exc:
        logger.error('%s', exc)
        return 1
    if args.images_only and args.relabel_duplicates:
        logger.error(
            '--relabel-duplicates imports labels; it cannot be combined with --images-only'
        )
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
        api_base=args.api_base.rstrip('/'),
        state_dir=args.state_dir or Path('dataset_import_state') / dataset_root.name,
        source_prefix=args.source_prefix or dataset_root.name,
        path_map=args.path_map,
        label_source=args.label_source,
        detect_mismatches=not args.no_detect_mismatches,
        batch_size=max(1, args.batch_size),
        concurrency=max(1, args.concurrency),
        relabel_duplicates=args.relabel_duplicates,
        verify_labels=not args.no_verify_labels,
        images_only=args.images_only,
        force=args.force,
    )
    for name, samples in splits.items():
        pos = sum(1 for s in samples if s.positive)
        logger.info(
            '%s: %d images, %d positive, %d background', name, len(samples), pos, len(samples) - pos
        )
    async with httpx.AsyncClient() as client:
        try:
            if args.dry_run:
                if names is not None and not (args.skip_class_check or args.images_only):
                    await DatasetImporter(cfg, client).check_classes(names)
                    logger.info('class registry check passed for %d classes', len(names))
                return 0
            summary = await run(cfg, client, splits, names, check_classes=not args.skip_class_check)
        except PreflightError as exc:
            logger.error('%s', exc)
            return 3
    logger.info('summary: %s', json.dumps(summary['total']))
    if not cfg.images_only:
        logger.info('report: %s', cfg.state_dir / 'disagreements.jsonl')
    logger.info('ingested image lists: %s', cfg.state_dir / 'ingested')
    return 0 if summary['total']['failed'] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return asyncio.run(_async_main(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
