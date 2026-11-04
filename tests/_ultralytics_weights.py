"""Official model files for the ``families`` tests.

The files live in ``~/.cache/op-test-weights`` (``$XDG_CACHE_HOME`` when set) and are never
committed. The fixture only reads them: when one is absent the test skips locally and fails
under ``CI``. The CI job ``families`` fills the cache with ``python tests/_ultralytics_weights.py
fetch`` and builds the original-YOLOv5 ONNX files (see ``.github/workflows/ci.yml``).

The weights and the YOLOv5 repository are AGPL-3.0; they are used for tests only and are not
redistributed.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


ASSETS = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'
YOLOV5 = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/'

# file name -> (download url, sha256 prefix). The prefixes are the ones recorded on 2026-11-03; a
# prefix of at least 8 hex digits is compared, replace with the full digest on the first fetch.
WEIGHTS: dict[str, tuple[str, str]] = {
    'yolo26n.pt': (ASSETS + 'yolo26n.pt', '9b09cc8b'),
    'yolo11n.pt': (ASSETS + 'yolo11n.pt', '0ebbc80d'),
    'yolov10n.pt': (ASSETS + 'yolov10n.pt', '11287ed0'),
    'yolov5nu.pt': (ASSETS + 'yolov5nu.pt', '9e9c1be4'),
    'yolov5n.pt': (YOLOV5 + 'yolov5n.pt', '4f180cf2'),
}

# Produced by the official ultralytics/yolov5 export.py (the CI job, or by hand once).
ONNX_FILES = ('yolov5n_640.onnx', 'yolov5n_dynamic.onnx')


def cache_dir() -> Path:
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'op-test-weights'


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(name: str) -> Path | None:
    """The cached file, or ``None`` when absent. A digest mismatch raises."""
    path = cache_dir() / name
    if not path.is_file():
        return None
    if name in WEIGHTS:
        want = WEIGHTS[name][1]
        got = sha256_of(path)
        if not got.startswith(want):
            raise RuntimeError(f'{path}: sha256 {got} does not start with the pinned {want}')
    return path


def fetch_all() -> None:
    cache_dir().mkdir(parents=True, exist_ok=True)
    for name, (url, _) in WEIGHTS.items():
        target = cache_dir() / name
        if not target.is_file():
            urllib.request.urlretrieve(url, target)  # fixed https release URLs above
        print(f'{name} {sha256_of(target)}')
        verified_path(name)


if __name__ == '__main__':
    if sys.argv[1:] == ['fetch']:
        fetch_all()
    else:
        raise SystemExit('usage: python tests/_ultralytics_weights.py fetch')
