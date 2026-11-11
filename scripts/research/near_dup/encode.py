"""Whole-frame vectors for the near-duplicate prototypes.

Two spaces, both through the product's own preprocessing so the research vectors are the product's space:

* ``pe_frame``: PE-Core-L14-336, ``src.utils.pe_preprocess.whole_frame_rgb`` (shorter edge to 336, centre square);
* ``mobileclip_frame``: MobileCLIP2-S2, ``src.utils.cpu_preprocess.center_crop_cpu`` (256 px).

The encoders are the research loaders of ``scripts/research/fine_grained/models.py`` (set ``FG_MOBILECLIP_CKPT``
to the MobileCLIP2-S2 checkpoint; the PE weights come from the Hugging Face cache). Output: ``<out>.npy``
(float32, unit norm, one row per input line) and ``<out>.json`` (ids and the settings).

Run from the repository root with ``PYTHONPATH=.``; ``--help`` lists the arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SPACES = {'pe_frame': 's_pe_core', 'mobileclip_frame': 's_mobileclip2'}


def _pe_input(path: str) -> np.ndarray:
    from src.utils.pe_preprocess import PE_SIZE, whole_frame_rgb

    rgb = whole_frame_rgb(path, PE_SIZE)
    if rgb is None:
        raise ValueError(f'undecodable image {path}')
    return rgb


def _mobileclip_input(path: str) -> np.ndarray:
    from src.utils.cpu_preprocess import center_crop_cpu

    with Image.open(path) as img:
        rgb = np.asarray(ImageOps.exif_transpose(img).convert('RGB'))
    chw = center_crop_cpu(rgb, 256)
    return np.asarray(np.rint(chw.transpose(1, 2, 0) * 255.0), dtype=np.uint8)


INPUTS: dict[str, Callable[[str], np.ndarray]] = {
    'pe_frame': _pe_input,
    'mobileclip_frame': _mobileclip_input,
}


def batches(paths: list[str], space: str, batch: int, threads: int) -> Iterator[np.ndarray]:
    """HWC uint8 batches of the model inputs, decoded and preprocessed on ``threads`` CPU threads."""
    fn = INPUTS[space]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for start in range(0, len(paths), batch):
            yield np.stack(list(pool.map(fn, paths[start : start + batch])))


def encode_paths(
    paths: list[str], space: str, *, batch: int = 64, threads: int = 8, device: str = 'cuda'
) -> np.ndarray:
    from scripts.research.fine_grained import models

    enc = models.load(SPACES[space], device)
    out = [enc.encode(x) for x in batches(paths, space, batch, threads)]
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paths', type=Path, required=True, help='text file, one image path per line')
    ap.add_argument('--space', choices=sorted(SPACES), required=True)
    ap.add_argument('--out', type=Path, required=True, help='output prefix (.npy and .json)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args(argv)
    paths = [ln for ln in args.paths.read_text().splitlines() if ln]
    vec = encode_paths(
        paths, args.space, batch=args.batch, threads=args.threads, device=args.device
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix('.npy'), vec)
    args.out.with_suffix('.json').write_text(
        json.dumps({'space': args.space, 'n': len(paths), 'dim': int(vec.shape[1]), 'paths': paths})
    )
    sys.stdout.write(f'{len(paths)} vectors {vec.shape} -> {args.out}.npy\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
