#!/usr/bin/env python3
"""Keep the photos of a proxy dataset that hold exactly one face (offline step, needs the stack).

    filter_one_face.py --dataset <dir> [--api http://127.0.0.1:4603] [--confidence 0.5] [--dry-run]

Each image goes to ``POST /faces/detect`` (SCRFD). Rows with exactly one detection are written to
``labels_one_face.jsonl`` next to ``labels.jsonl`` and a count report to ``one_face_report.json``;
the images, ``manifest.json`` and ``labels.jsonl`` are left untouched. ``--dry-run`` checks the
first ``--limit`` images and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.datasets._common import write_json
from scripts.datasets.proxy_manifest import LABELS_NAME, read_labels


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('filter_one_face')

OUT_LABELS = 'labels_one_face.jsonl'
OUT_REPORT = 'one_face_report.json'


class DetectError(RuntimeError):
    """The face API did not return a usable answer for one image."""


def filter_rows(
    rows: list[dict[str, Any]], count_faces: Callable[[str], int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    report = {'checked': 0, 'kept': 0, 'no_face': 0, 'multi_face': 0, 'error': 0}
    kept: list[dict[str, Any]] = []
    for row in rows:
        report['checked'] += 1
        try:
            n = count_faces(row['path'])
        except DetectError as exc:
            logger.warning('%s: %s', row['path'], exc)
            report['error'] += 1
            continue
        if n == 1:
            kept.append(row)
            report['kept'] += 1
        elif n == 0:
            report['no_face'] += 1
        else:
            report['multi_face'] += 1
    return kept, report


def detect_count(api: str, image: Path, confidence: float) -> int:
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{image.name}"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode()
    body = head + image.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(
        f'{api.rstrip("/")}/faces/detect?confidence={confidence}',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
            return int(json.load(resp)['num_faces'])
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise DetectError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--api', default='http://127.0.0.1:4603')
    p.add_argument('--confidence', type=float, default=0.5)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--limit', type=int, default=20)
    args = p.parse_args(argv)
    rows = read_labels(args.dataset / LABELS_NAME)
    if args.dry_run:
        rows = rows[: args.limit]
    kept, report = filter_rows(
        rows, lambda rel: detect_count(args.api, args.dataset / rel, args.confidence)
    )
    print(report)
    if args.dry_run:
        return 0
    (args.dataset / OUT_LABELS).write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in kept), encoding='utf-8'
    )
    write_json(args.dataset / OUT_REPORT, report)
    return 0 if report['error'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
