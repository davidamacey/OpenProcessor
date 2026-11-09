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

# file name -> (download url, sha256 or sha256 prefix). The digests are the ones GitHub publishes
# for the release assets; the original yolov5n.pt asset publishes none, so its 8-digit prefix
# (recorded 2026-11-03) is pinned until its full digest is taken from a first fetch.
WEIGHTS: dict[str, tuple[str, str]] = {
    'yolo26n.pt': (
        ASSETS + 'yolo26n.pt',
        '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef',
    ),
    'yolo11n.pt': (
        ASSETS + 'yolo11n.pt',
        '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
    ),
    'yolov10n.pt': (
        ASSETS + 'yolov10n.pt',
        '11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f',
    ),
    'yolov8n.pt': (
        ASSETS + 'yolov8n.pt',
        'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
    ),
    'yolov9t.pt': (
        ASSETS + 'yolov9t.pt',
        '410e3caca789aa8fec19971c05577616cbb963a416bcd0228171b2dff7f718f7',
    ),
    'yolo12n.pt': (
        ASSETS + 'yolo12n.pt',
        '419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6',
    ),
    'yolo26s.pt': (
        ASSETS + 'yolo26s.pt',
        '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b',
    ),
    'yolov5nu.pt': (
        ASSETS + 'yolov5nu.pt',
        '9e9c1be448b0e1b8598975a9abcab9a0fd0e21182eb422cda7f044a0442d0937',
    ),
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
