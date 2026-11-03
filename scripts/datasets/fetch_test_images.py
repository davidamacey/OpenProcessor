#!/usr/bin/env python3
"""Fetch the two public COCO val2017 sample images the demo targets, the smoke test and
``scripts/validation/validate_models.py`` use.

* ``STREET_IMAGE``: a street scene with a bus and readable text (detection, OCR, embeddings).
* ``FACES_IMAGE``: two frontal faces with a tie (faces, detection).

Both land in ``test_images/coco/``. Exits non-zero, naming the image, when a download fails.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import FetchError, download
from scripts.datasets.bench_set import IMAGE_URL_TMPL


STREET_IMAGE = '000000001584.jpg'
FACES_IMAGE = '000000011051.jpg'
IMAGE_DIR = Path('test_images/coco')


def fetch(directory: Path = IMAGE_DIR) -> list[Path]:
    """Download both images into ``directory`` (idempotent) and return their paths."""
    return [
        download(IMAGE_URL_TMPL['val2017'].format(file_name=name), directory / name)
        for name in (STREET_IMAGE, FACES_IMAGE)
    ]


def main() -> int:
    try:
        paths = fetch()
    except (FetchError, OSError) as exc:
        print(
            f'ERROR: could not fetch the COCO test images into {IMAGE_DIR}: {exc}', file=sys.stderr
        )
        return 1
    print(f'Test images ready: {", ".join(str(p) for p in paths)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
