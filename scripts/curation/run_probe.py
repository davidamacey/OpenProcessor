#!/usr/bin/env python3
"""Probe-inference backfill — populate the active-learning probe fields.

Thin operator driver over
:func:`src.services.curation.probe_predictions.run_probe_inference`. It
runs a small/fast "probe" detector over every non-holdout item in the
configured items index and writes ``probe_pred_class`` /
``probe_pred_confidence`` / ``probe_pred_entropy`` / ``probe_pred_margin``
/ ``probe_disagreement`` / ``probe_model_version`` / ``probe_scored_at``.
Those fields feed the ``/review`` uncertainty and model-disagreement tabs
and the ``mistakenness`` item score; without a run of this script they
stay empty.

Operator-triggered on purpose: nothing in the stack tracks "which trained
checkpoint is the current probe", so run it once after a probe-profile
training job exports its ONNX (or when you want to reuse an already
deployed detector as the probe via ``--architecture yolov5_objectness``).

    # Count what would be scored — no model load, no writes.
    python3 scripts/curation/run_probe.py --model /runs/probe-1/weights/best.onnx

    # Score everything.
    python3 scripts/curation/run_probe.py --model /runs/probe-1/weights/best.onnx --apply

    # Resume an interrupted pass (skips items already stamped with this
    # --model-version), with smaller scroll pages for a slow CPU probe.
    python3 scripts/curation/run_probe.py --model /models/detector.onnx \\
        --architecture yolov5_objectness --model-version detector-v2 --resume --page-size 200 --apply

Without ``--resume`` every run re-scores every item: probe fields are
meant to reflect the most recently promoted probe, not a mix of versions.
Index names and image roots come from ``CurationConfig`` (``OP_*`` env).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.config import get_curation_config
from src.services.curation.probe_predictions import count_probe_candidates, run_probe_inference


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
ARCHITECTURES = ('yolo11', 'yolov5_objectness')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('curation_run_probe')


def default_model_version(model_path: Path) -> str:
    """``<run>/weights/best.onnx`` -> ``<run>``; anything else -> the file stem.

    A bare ``best.onnx`` is the same filename for every training run, so it
    is useless as a provenance tag (and would make ``--resume`` skip items
    scored by a *different* run).
    """
    if model_path.parent.name == 'weights' and model_path.parent.parent.name:
        return model_path.parent.parent.name
    return model_path.stem


async def _async_main(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        logger.error('model not found: %s', model_path)
        return 1

    cfg = get_curation_config()
    version = args.model_version or default_model_version(model_path)
    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        candidates = await count_probe_candidates(
            client, config=cfg, skip_version=version if args.resume else None
        )
        planned = min(candidates, args.limit) if args.limit is not None else candidates
        print(
            f'\nprobe {model_path} (architecture={args.architecture}, version={version})'
            f'\n  index={cfg.items_index} candidates={candidates:,} planned={planned:,}'
            f' resume={args.resume}'
        )
        if args.dry_run:
            print('\nDry-run only (model not loaded). Pass --apply to score and write.')
            return 0

        processed = await run_probe_inference(
            model_path,
            client,
            config=cfg,
            max_crops=args.limit,
            model_version=version,
            architecture=args.architecture,
            page_size=args.page_size,
            resume=args.resume,
        )
        await client.indices.refresh(index=cfg.items_index)
        print(f'\nprobe-scored {processed:,} items with {model_path}')
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--model', required=True, help='Path to the probe ONNX export.')
    p.add_argument(
        '--architecture',
        choices=ARCHITECTURES,
        default='yolo11',
        help=(
            "'yolo11' (default): ultralytics-loadable ONNX from a probe training run. "
            "'yolov5_objectness': a non-ultralytics YOLOv5-family ONNX with an objectness channel "
            '(reuse an already deployed detector instead of training a probe).'
        ),
    )
    p.add_argument(
        '--model-version',
        default=None,
        help='Provenance tag for probe_model_version (default: run dir name or file stem).',
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--limit', type=int, default=None, help='Cap items scored (smoke runs).')
    p.add_argument(
        '--page-size',
        type=int,
        default=1000,
        help='Items per scroll page; lower it for a slow probe so a page finishes '
        'inside the scroll keep-alive.',
    )
    p.add_argument(
        '--resume',
        action='store_true',
        help='Skip items already stamped with this --model-version.',
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    args = p.parse_args()
    if args.page_size <= 0:
        p.error('--page-size must be positive')
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    raise SystemExit(main())
