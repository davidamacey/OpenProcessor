#!/usr/bin/env python3
"""The wheel example against a LIVE stack (Triton, segmenter, VLM, GPU).

The offline twin of this walk is ``tests/integration/test_wheel_example_e2e.py``:
same steps, same routes, fakes at the boundary. This script runs it for real,
so it needs everything the offline test fakes:

* the stack up (API, OpenSearch, Triton with the primary detector loaded,
  the segmenter service, the ``curation`` compose profile's detection worker);
* an activated VLM endpoint for the project (``POST .../vlm/endpoints/{name}/activate``),
  unless you pass ``--no-vlm`` to accept segmenter boxes unverified;
* ``make sample-coco-cars`` already run (public COCO car images, CC BY),
  with ``data/samples/coco_car`` mounted into the API container (see "Mounting
  your image source" in docs/CURATION.md) and passed as ``--container-dir``.

The primary detector (a COCO-class YOLO) proposes vehicles; a proposal's own
label ("car") is kept as ``proposal_name`` and the wheel profile's
``parent_classes`` selects items by that NAME, never by the detector's class
index. COCO has no wheel labels: what this produces are machine-proposed wheel
boxes for a human to review, which is the point of the example.

Usage::

    .venv/bin/python scripts/examples/wheel_example_live.py \\
        --api http://localhost:4603 --project wheels \\
        --container-dir /data/source/coco_car/images

Nothing here is run by CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'
TIMEOUT = 60


def call(method: str, url: str, **kwargs: Any) -> Any:
    response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    if response.status_code >= 400:
        sys.exit(f'{method} {url} -> {response.status_code}: {response.text[:500]}')
    return response.json() if response.content else None


def example_body(kind: str) -> dict[str, Any]:
    raw = json.loads((EXAMPLES / kind / 'vehicle_wheel.json').read_text())
    return {k: v for k, v in raw.items() if k not in ('_comment', 'name')}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--api', default='http://localhost:4603')
    ap.add_argument('--project', default='wheels')
    ap.add_argument(
        '--container-dir', required=True, help='car images as the API container sees them'
    )
    ap.add_argument('--host-dir', type=Path, default=Path('data/samples/coco_car/images'))
    ap.add_argument('--drain-timeout', type=float, default=1800.0)
    args = ap.parse_args()
    base = f'{args.api}/curation'
    scoped = f'{base}/projects/{args.project}'

    print(f'1. create project {args.project!r}')
    requests.post(
        f'{base}/projects', json={'slug': args.project, 'display_name': 'Wheels'}, timeout=TIMEOUT
    )  # 409 when it already exists: fine, the rest is idempotent
    for name in ('car', 'wheel'):
        requests.post(f'{scoped}/classes', json={'name': name}, timeout=TIMEOUT)  # 409 = exists

    print('2. activate the example region profile and prompt pack')
    for kind in ('region_profiles', 'prompt_packs'):
        call('POST', f'{scoped}/{kind}', json={'name': 'wheel_example', 'body': example_body(kind)})
        call('POST', f'{scoped}/{kind}/wheel_example/activate', json={'expected_active': None})

    print('3. ingest the car images (the primary detector proposes the cars)')
    files = sorted(p.name for p in args.host_dir.glob('*.jpg'))
    if not files:
        sys.exit(f'no images under {args.host_dir}; run `make sample-coco-cars` first')
    for start in range(0, len(files), 16):
        items = [
            {'path': f'{args.container_dir}/{name}', 'source': 'coco_car'}
            for name in files[start : start + 16]
        ]
        call('POST', f'{scoped}/ingest/batch', json={'items': items})

    print('4. wait for the detection worker to drain the region queue')
    deadline = time.time() + args.drain_timeout
    while time.time() < deadline:
        drain = call('GET', f'{scoped}/ingest/region_drain')
        if drain['drained']:
            break
        time.sleep(5)
    else:
        sys.exit(f'region queue did not drain: {drain}')

    print('5. export the wheel boxes cropped to their car')
    body: dict[str, Any] = {
        'profile_name': 'wheels',
        'box_source': 'region',
        'region_class_name': 'wheel',
        'image_mode': 'item_crop',
        # The primary detector's proposals are unlabeled (no registry car yet),
        # so no parent-class id filter: the profile's parent_classes already
        # kept the region stage to items proposed as "car".
        'class_ids': [],
    }
    result = call('POST', f'{scoped}/export/single_class', json=body)
    print(json.dumps({k: result[k] for k in ('export_dir', 'positive_images', 'split_counts')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
