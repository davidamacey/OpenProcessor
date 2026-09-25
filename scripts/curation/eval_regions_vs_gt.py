#!/usr/bin/env python3
"""Check the region-detection cascade against a ground-truth YOLO dataset.

Point it at a YOLO dataset of whole frames labeled with the *region* class
(e.g. ``nc: 1, names: {0: defect}``, background frames included) whose
images were ingested — typically with ``import_labeled_dataset.py
--images-only`` so the region labels never enter the item class registry.
It reads every item on those frames from the items index, takes the region
boxes in an accepted status (``--accept-status``, default ``detected``) and
scores them against the ground truth:

* recall at IoU 0.5 and at ``--iou``, precision, F1, mean IoU of matches
  (greedy one-to-one matching, highest IoU first);
* the false-positive gate: background frames with >= 1 region, and the
  number of such regions;
* frames not ingested, frames with no items (the ingest detector found no
  parent object), and frames still pending in the cascade — pending frames
  are reported and left out of the metrics, never counted as misses
  (``--wait-pending`` polls until the cohort drains);
* breakdowns by region detector and by region status, plus, for missed
  boxes, whether a region in a *non*-accepted status (e.g. ``verify_rejected``)
  did cover them.

Outputs under ``--out-dir``: ``summary.json``, ``misses.jsonl`` (every missed
GT box, worst first) and ``false_positives.jsonl``; a table goes to stdout.

Cohort, in order of precedence:

* ``--state-dir`` — the import state dir; reads ``ingested/<split>.jsonl``
  (carries each image's ``image_id``, so content duplicates ingested under
  another path still resolve);
* ``--image-list`` — a file of local image paths, one per line;
* otherwise every image of ``--splits`` in the dataset.

Paths are joined to the index by ``image_id`` when known, else by the
server-side path (``--path-map LOCAL=SERVER`` if the dataset is mounted
elsewhere on the server; unnecessary when both see the same path). Field and
index names follow ``RegionFields`` (``OP_REGION_FIELD_*``) and
``CurationConfig`` (``OP_*``).

    # After: import_labeled_dataset.py --dataset DS --images-only --splits test \\
    #            --state-dir ./state/regions
    python3 scripts/curation/eval_regions_vs_gt.py --dataset DS/data.yaml \\
        --state-dir ./state/regions --splits test --wait-pending 1800 \\
        --out-dir ./state/regions/eval

Exit codes: 0 done; 1 bad input; 2 ``--wait-pending`` timed out with frames
still pending; 3 a region box is stored in an unmappable coordinate frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from scripts.curation.ingest_upload import map_identifier, parse_path_map
from scripts.curation.yolo_dataset import DatasetError, discover, label_path_for
from src.config import RegionStatus, get_curation_config, get_region_fields
from src.services.curation.region_eval import (
    DEFAULT_ACCEPTED_STATUSES,
    CohortImage,
    EvalResult,
    RegionFrameError,
    parse_yolo_labels,
    run_eval,
)


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')

logger = logging.getLogger('eval_regions_vs_gt')


# =============================================================================
# Cohort
# =============================================================================


def read_gt(image: Path, class_ids: list[int] | None) -> list[tuple[float, float, float, float]]:
    try:
        text = label_path_for(image).read_text(encoding='utf-8')
    except FileNotFoundError:
        return []
    return parse_yolo_labels(text, class_ids)


def _entry(
    image: Path,
    split: str,
    *,
    path_map: tuple[str, str] | None,
    class_ids: list[int] | None,
    server_path: str | None = None,
    image_id: str | None = None,
) -> CohortImage:
    return CohortImage(
        key=str(image),
        server_path=server_path or map_identifier(image, path_map),
        gt=read_gt(image, class_ids),
        split=split,
        image_id=image_id or None,
    )


def build_cohort(args: argparse.Namespace) -> list[CohortImage]:
    """Resolve the evaluated frames from state dir / image list / dataset."""
    kw: dict[str, Any] = {'path_map': args.path_map, 'class_ids': args.gt_class or None}
    wanted = [s.strip() for s in args.splits.split(',')] if args.splits else None

    if args.state_dir is not None:
        ingested = args.state_dir / 'ingested'
        files = sorted(ingested.glob('*.jsonl')) if ingested.is_dir() else []
        if wanted:
            files = [f for f in files if f.stem in wanted]
            missing = sorted(set(wanted) - {f.stem for f in files})
            if missing:
                raise DatasetError(f'no {ingested}/<split>.jsonl for splits {missing}')
        if not files:
            raise DatasetError(
                f'{ingested}: no ingested/<split>.jsonl lists (run the import first)'
            )
        cohort = []
        for f in files:
            for line in f.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    row = json.loads(line)
                    cohort.append(
                        _entry(
                            Path(row['image']),
                            f.stem,
                            server_path=row.get('server_path'),
                            image_id=row.get('image_id'),
                            **kw,
                        )
                    )
        return cohort

    if args.image_list is not None:
        split = wanted[0] if wanted and len(wanted) == 1 else 'list'
        lines = args.image_list.read_text(encoding='utf-8').splitlines()
        return [_entry(Path(ln.strip()), split, **kw) for ln in lines if ln.strip()]

    found, _names = discover(args.dataset)
    splits = wanted or list(found)
    missing = [s for s in splits if s not in found]
    if missing:
        raise DatasetError(f'splits not in dataset: {missing} (found: {sorted(found)})')
    return [_entry(img, split, **kw) for split in splits for img in found[split]]


# =============================================================================
# Output
# =============================================================================


def _fmt(v: Any) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.4f}'
    return f'{v:,}' if isinstance(v, int) else str(v)


def render(result: EvalResult, *, worst: int) -> str:
    s = result.summary
    rows = [('total', s['total'])] + [(f'split {k}', v) for k, v in s['splits'].items()]
    cols = [
        ('images', 'images'),
        ('not_ingested', 'not ingested'),
        ('pending_images', 'pending'),
        ('no_item_images', 'no items'),
        ('evaluated_images', 'evaluated'),
        ('evaluated_gt_boxes', 'GT boxes'),
        ('predictions', 'regions'),
        ('tp', 'TP'),
        ('false_positives', 'FP'),
        ('recall_at_0.5', 'R@0.5'),
        ('recall', f'R@{s["iou_threshold"]}'),
        ('precision', 'P'),
        ('f1', 'F1'),
        ('mean_iou_matched', 'mIoU'),
        ('evaluated_backgrounds', 'bg frames'),
        ('backgrounds_with_detection', 'bg w/ region'),
        ('background_fp_regions', 'bg FP regions'),
    ]
    out = [
        f'accepted statuses: {", ".join(s["accepted_statuses"])}   '
        f'IoU threshold: {s["iou_threshold"]}   dedup IoU: {s["dedup_iou"]}',
        '',
    ]
    for label, m in rows:
        out.append(f'[{label}]')
        out.extend(f'  {title:<16}{_fmt(m.get(key)):>12}' for key, title in cols)
    out += ['', 'by detector (accepted regions):']
    for det, c in s['total']['by_detector'].items():
        out.append(
            f'  {det:<32} regions={c["predictions"]:>8,} tp={c["tp"]:>8,} fp={c["fp"]:>8,}'
            f' precision={_fmt(c["precision"])}'
        )
    out += ['', 'by region status (all items on cohort frames):']
    for status, c in s['by_status'].items():
        out.append(f'  {status:<24} items={c["items"]:>8,} with_box={c["with_box"]:>8,}')
    out += ['', f'miss reasons: {json.dumps(s["miss_reasons"])}']
    if s['missed_but_boxed_by_status']:
        out.append(
            'missed GT boxes covered by a non-accepted region: '
            + json.dumps(s['missed_but_boxed_by_status'])
        )
    if worst and result.misses:
        out += ['', f'worst {min(worst, len(result.misses))} misses:']
        for m in result.misses[:worst]:
            box = ', '.join(f'{v:.3f}' for v in m['gt_box'])
            out.append(f'  iou={m["best_iou"]:.3f} {m["reason"]:<28} [{box}] {m["image"]}')
    if s.get('wait_timed_out'):
        out += ['', 'WARNING: --wait-pending timed out; pending frames are excluded above.']
    return '\n'.join(out)


def write_outputs(result: EvalResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(result.summary, indent=2), encoding='utf-8')
    for name, rows in (('misses', result.misses), ('false_positives', result.false_positives)):
        (out_dir / f'{name}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8'
        )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--dataset', required=True, type=Path, help='data.yaml or dataset root')
    p.add_argument('--splits', default=None, help='Comma-separated subset (default: all)')
    p.add_argument('--state-dir', type=Path, default=None, help='Import state dir (cohort)')
    p.add_argument('--image-list', type=Path, default=None, help='File of local image paths')
    p.add_argument(
        '--path-map',
        type=parse_path_map,
        default=None,
        metavar='LOCAL=SERVER',
        help='Rewrite the local dataset prefix to the path the server indexed',
    )
    p.add_argument(
        '--gt-class',
        type=int,
        action='append',
        default=[],
        help='Ground-truth class id(s) to score (repeatable; default: every row)',
    )
    p.add_argument(
        '--accept-status',
        action='append',
        default=[],
        choices=[s.value for s in RegionStatus],
        help='Region status counted as a detection (repeatable; default: '
        f'{", ".join(DEFAULT_ACCEPTED_STATUSES)})',
    )
    p.add_argument('--iou', type=float, default=0.5, help='Match IoU threshold (default 0.5)')
    p.add_argument(
        '--dedup-iou',
        type=float,
        default=0.7,
        help='Merge accepted regions on one frame overlapping at >= this IoU (0 = off)',
    )
    p.add_argument(
        '--wait-pending',
        type=float,
        default=0.0,
        metavar='SECONDS',
        help='Poll until no cohort frame is pending (or this many seconds pass)',
    )
    p.add_argument('--poll-interval', type=float, default=30.0, metavar='SECONDS')
    p.add_argument('--out-dir', type=Path, default=Path('region_eval'))
    p.add_argument('--worst', type=int, default=20, help='Misses to print (all go to JSONL)')
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    return p


async def _async_main(args: argparse.Namespace, cohort: list[CohortImage]) -> int:
    def _progress(result: EvalResult, remaining: float) -> None:
        t = result.summary['total']
        logger.info(
            'pending frames=%d evaluated=%d recall=%s (%.0fs left)',
            t['pending_images'],
            t['evaluated_images'],
            t['recall'],
            max(0.0, remaining),
        )

    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        result = await run_eval(
            client,
            cohort,
            config=get_curation_config(),
            fields=get_region_fields(),
            accepted=args.accept_status or DEFAULT_ACCEPTED_STATUSES,
            iou_threshold=args.iou,
            dedup_iou=args.dedup_iou,
            wait_pending_s=args.wait_pending,
            poll_interval_s=args.poll_interval,
            on_poll=_progress if args.wait_pending > 0 else None,
        )
    except RegionFrameError as exc:
        logger.error('%s', exc)
        return 3
    finally:
        await client.close()
    write_outputs(result, args.out_dir)
    print(render(result, worst=args.worst))
    print(f'\nwritten: {args.out_dir}/summary.json, misses.jsonl, false_positives.jsonl')
    return 2 if result.summary.get('wait_timed_out') else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    if not 0.0 < args.iou <= 1.0 or not 0.0 <= args.dedup_iou <= 1.0:
        logger.error('--iou must be in (0, 1] and --dedup-iou in [0, 1]')
        return 1
    try:
        cohort = build_cohort(args)
    except (DatasetError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.error('%s', exc)
        return 1
    if not cohort:
        logger.error('empty cohort')
        return 1
    pos = sum(1 for c in cohort if c.positive)
    logger.info('cohort: %d frames (%d with GT regions, %d background)', len(cohort), pos,
                len(cohort) - pos)  # fmt: skip
    return asyncio.run(_async_main(args, cohort))


if __name__ == '__main__':
    sys.exit(main())
