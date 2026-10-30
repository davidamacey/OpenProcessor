"""Labelled crop pools of set C at make and model level, for the judge agreement run (FG-7).

    FG_SETC_ROOT=<set C directory> python -m scripts.research.fine_grained.fg7_pool --boxes boxes_setc.json --out DIR

Writes ``DIR/make/pool.jsonl`` and ``DIR/model/pool.jsonl`` (``visual_judge.py make-e0`` input) over
one shared set of crop PNGs: the detector's largest car or truck box of each image, at least
48 px on its short side.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from scripts.research.fine_grained import data, setc


MIN_SIDE_PX = 48
MIN_PER_LABEL = 12


def crop_png(path: Path, box: list[float]) -> bytes | None:
    with Image.open(path) as im:
        frame = ImageOps.exif_transpose(im).convert('RGB')
    x1, y1, x2, y2 = (
        max(0, round(box[0])),
        max(0, round(box[1])),
        min(frame.width, round(box[2])),
        min(frame.height, round(box[3])),
    )
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    buf = io.BytesIO()
    frame.crop((x1, y1, x2, y2)).save(buf, 'PNG')
    return buf.getvalue()


def build(boxes: dict[str, list[float] | None], out: Path) -> dict[str, int]:
    root = data.env_path('FG_SETC_ROOT')
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for it in setc.load_items(root):
        box = boxes.get(it.path)
        png = crop_png(root / it.path, box[:4]) if box else None
        if png is None:
            continue
        name = it.path.replace('/', '__').removesuffix('.jpg')
        (out / 'crops' / f'{name}.png').write_bytes(png)
        rows.append(
            {
                'item_id': it.path,
                'make': it.make,
                'model': it.model,
                'group': it.path,
                'path': str(out / 'crops' / f'{name}.png'),
                'source': 'commons_set_c',
            }
        )
    counts = {}
    for level in ('make', 'model'):
        (out / level).mkdir(exist_ok=True)
        per: dict[str, int] = {}
        for r in rows:
            per[r[level]] = per.get(r[level], 0) + 1
        keep = [
            {**{k: v for k, v in r.items() if k not in ('make', 'model')}, 'label': r[level]}
            for r in rows
            if per[r[level]] >= MIN_PER_LABEL
        ]
        (out / level / 'pool.jsonl').write_text(
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in keep)
        )
        counts[level] = len(keep)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--boxes', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    print(build(json.loads(Path(args.boxes).read_text()), Path(args.out)))


if __name__ == '__main__':
    main()
