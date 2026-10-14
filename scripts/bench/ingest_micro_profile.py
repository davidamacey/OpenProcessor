"""Single-thread cost of the per-image primitives the ingest path runs, per image set.

Times, for each image, the operations `CurationIngestService` performs on the event-loop thread
and that no stage timer covers or only partly covers: file read, SHA-256, PIL open plus EXIF
transpose (the decode), the detector letterbox, the PE whole-frame preprocess, and for one
central crop the crop-to-array, PE resize and normalize, and JPEG encode. Run it where the
code runs (inside the API container, so the library versions match the stack):

    docker exec -i -w /app <api container> python3 - DIR_A DIR_B < scripts/bench/ingest_micro_profile.py

Each argument is a directory of JPEGs; the output is one JSON line with the median
milliseconds per operation for the first ``--limit`` images of each directory. Unlike the
stack-wide timers this isolates a single call from the other workers, so it is a floor: under 32
busy workers on a hyper-threaded host the same calls cost more CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


_HERE = Path(globals().get('__file__', ''))
if _HERE.is_file():  # piped through stdin inside the API container there is no file
    sys.path.insert(0, str(_HERE.resolve().parents[2]))
sys.path.insert(0, '/app')

# ruff: noqa: E402

from src.services.detection.geometry import letterbox_to_square
from src.utils.pe_preprocess import resize_crop_rgb, to_chw, whole_frame_chw_from_bytes


OPERATIONS = (
    'read',
    'sha256',
    'decode',
    'letterbox',
    'whole_frame',
    'crop_to_array',
    'resize_chw',
    'jpeg_encode_crop',
)
LETTERBOX_SIZE = 640


def profile_image(path: Path) -> dict[str, float]:
    """Milliseconds per operation for one image file, plus its megapixels."""
    marks = [time.perf_counter()]

    def lap() -> None:
        marks.append(time.perf_counter())

    data = path.read_bytes()
    lap()
    hashlib.sha256(data).hexdigest()
    lap()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    img.load()
    lap()
    letterbox_to_square(img, LETTERBOX_SIZE)
    lap()
    whole_frame_chw_from_bytes(data)
    lap()
    w, h = img.size
    crop = img.crop((w // 4, h // 4, w // 4 + w // 3, h // 4 + h // 3))
    arr = np.asarray(crop)
    lap()
    to_chw(resize_crop_rgb(arr))
    lap()
    crop.save(io.BytesIO(), format='JPEG', quality=85)
    lap()
    out = {op: (b - a) * 1000.0 for op, a, b in zip(OPERATIONS, marks[:-1], marks[1:], strict=True)}
    out['megapixels'] = w * h / 1e6
    return out


def profile_directory(directory: Path, limit: int) -> dict[str, float]:
    """Median of :func:`profile_image` over the first ``limit`` JPEGs of ``directory``."""
    files = sorted(directory.glob('*.jpg'))[:limit]
    if not files:
        msg = f'no .jpg files in {directory}'
        raise FileNotFoundError(msg)
    rows = [profile_image(p) for p in files]
    return {k: round(statistics.median(r[k] for r in rows), 4) for k in rows[0]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--limit', type=int, default=12)
    args = p.parse_args(argv)
    print(json.dumps({str(d): profile_directory(d, args.limit) for d in args.directories}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
