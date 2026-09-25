#!/usr/bin/env python3
"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import asyncio
import sys

from src.core.logging import get_logger


logger = get_logger('curation_worker')


from scripts.curation.worker import run
from scripts.curation.worker.state import (
    DEFAULT_OPENSEARCH,
    DEFAULT_PAUSE_SENTINEL,
    DEFAULT_SEGMENTER_URL,
    DEFAULT_SEGMENTER_URLS,
    DEFAULT_TRITON,
    DEFAULT_VLM_URL,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser. Defaults match production environment vars."""
    p = argparse.ArgumentParser(
        description='Async secondary-segmenter detection worker (Phase 0c).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--opensearch', default=DEFAULT_OPENSEARCH)
    p.add_argument('--triton', default=DEFAULT_TRITON)
    p.add_argument(
        '--segmenter-url',
        default=DEFAULT_SEGMENTER_URLS or DEFAULT_SEGMENTER_URL,
        help=(
            'Secondary-segmenter base URL. May be a comma-separated list '
            'to round-robin across multiple segmenter services on '
            'different GPUs (env OP_SEGMENTER_URLS=http://segmenter-gpu0:8000,'
            'http://segmenter-gpu1:8000).'
        ),
    )
    p.add_argument(
        '--vlm-url',
        default=DEFAULT_VLM_URL,
        help='VLM base URL. Empty → use VlmLabeler defaults.',
    )
    p.add_argument(
        '--pause-sentinel',
        default=str(DEFAULT_PAUSE_SENTINEL),
        help='File whose presence pauses the worker (trainer arbiter writes it).',
    )
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument(
        '--concurrency',
        type=int,
        default=16,
        help='Concurrent per-crop pipelines. The shared VLM container is sized '
        'for 32 in-flight decode slots (--max-num-seqs 32); 16 here keeps that '
        'pipeline reasonably full while leaving headroom for the class-labeling '
        'worker. The secondary segmenter has its own dual-instance lock pool so '
        'high concurrency queues there, not at this worker.',
    )
    p.add_argument('--pool-size', type=int, default=8)
    p.add_argument(
        '--poll-interval',
        type=float,
        default=2.0,
        help='Sleep between polls when the queue is empty (continuous mode only).',
    )
    p.add_argument(
        '--sentinel-sleep',
        type=float,
        default=10.0,
        help='Seconds to sleep between sentinel checks while paused.',
    )
    p.add_argument(
        '--max-iterations',
        type=int,
        default=0,
        help='Exit after this many bulk iterations (0 = unlimited). Useful for '
        'one-shot backfill mode.',
    )
    p.add_argument(
        '--continuous',
        action='store_true',
        help='Run forever (Docker daemon mode). Polls every poll-interval '
        'until SIGINT/SIGTERM. Without this flag, exits on the first empty poll.',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.config.retired_env import reject_retired_env

    reject_retired_env()
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
