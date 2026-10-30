"""Embedding extraction for the standard-benchmark re-run. Research venv, locked GPU.

    flock $SCRATCH/locks/gpu2.lock nice -n 10 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
        $VENV/bin/python -m scripts.research.standard_benchmarks.embed_run \
        --dataset cars196 --models s_pe_core s_siglip2 --variants centre letterbox

``WB2_DATA_ROOT`` (the ``clustering/`` directory) and ``WB2_CACHE`` (output directory) come from
the environment; ``FG_MOBILECLIP_CKPT`` is needed for MobileCLIP2 (see
``scripts/research/fine_grained/models.py``). Each image is decoded once, upright RGB, and every
requested (model, variant) square is cut from that frame. Work is done in shards that are kept on
disk, so an interrupted run resumes at the first unfinished shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

from scripts.research.fine_grained import models as enc
from scripts.research.fine_grained.data import env_path, run_info
from scripts.research.fine_grained.preprocess import centre, letterbox
from scripts.research.standard_benchmarks import protocol


if TYPE_CHECKING:
    from pathlib import Path


SHARD = 8192
ENCODE_BATCH = 16
VARIANTS = {'centre': centre, 'letterbox': letterbox}


def combo_name(model_key: str, variant: str) -> str:
    return f'{enc.SPECS[model_key].name}__{variant}'


class Squares:
    """Map-style dataset: decode one file and cut every requested square from the frame."""

    def __init__(self, paths: list[Path], sizes: list[int], variants: list[str]) -> None:
        self.paths, self.sizes, self.variants = paths, sizes, variants

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        with Image.open(self.paths[i]) as im:
            frame = ImageOps.exif_transpose(im).convert('RGB')
        return {f'{s}_{v}': VARIANTS[v](frame, s) for s in self.sizes for v in self.variants}


def _collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}


def encode_chunks(encoder: Any, squares: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            encoder.encode(squares[i : i + ENCODE_BATCH])
            for i in range(0, len(squares), ENCODE_BATCH)
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    root = env_path('WB2_DATA_ROOT')
    out = env_path('WB2_CACHE') / args.dataset
    work = out / 'shards'
    work.mkdir(parents=True, exist_ok=True)
    rows = protocol.load_rows(root, args.dataset)
    (out / 'rows.jsonl').write_text(
        '\n'.join(json.dumps({'path': r.path, 'label': r.label, 'part': r.part}) for r in rows)
        + '\n'
    )
    combos = [(m, v) for m in args.models for v in args.variants]
    todo = [c for c in combos if not (out / f'{combo_name(*c)}.npy').exists()]
    if not todo:
        return {'skipped': 'all requested embeddings exist'}
    encoders = {m: enc.load(m) for m in dict.fromkeys(m for m, _ in todo)}
    sizes = sorted({enc.SPECS[m].size for m, _ in todo})
    variants = sorted({v for _, v in todo})
    encode_s = {combo_name(*c): 0.0 for c in todo}
    t0 = time.time()
    for s in range(0, len(rows), SHARD):
        shard = rows[s : s + SHARD]
        need = [c for c in todo if not (work / f'{combo_name(*c)}__{s:07d}.npy').exists()]
        if not need:
            continue
        loader: DataLoader[Any] = DataLoader(
            Squares(
                [root / protocol.SETS[args.dataset].directory / r.path for r in shard],
                sizes,
                variants,
            ),  # type: ignore[arg-type]  # map-style dataset, not a Dataset subclass
            batch_size=ENCODE_BATCH * 2,
            num_workers=args.workers,
            collate_fn=_collate,
        )
        parts: dict[tuple[str, str], list[np.ndarray]] = {c: [] for c in need}
        for batch in loader:
            for m, v in need:
                torch.cuda.synchronize()
                ts = time.time()
                parts[(m, v)].append(encode_chunks(encoders[m], batch[f'{enc.SPECS[m].size}_{v}']))
                torch.cuda.synchronize()
                encode_s[combo_name(m, v)] += time.time() - ts
        for c, arrs in parts.items():
            np.save(work / f'{combo_name(*c)}__{s:07d}.npy', np.concatenate(arrs))
        print(f'{args.dataset}: shard {s // SHARD + 1}/{-(-len(rows) // SHARD)} done', flush=True)
    for c in todo:
        shards = sorted(work.glob(f'{combo_name(*c)}__*.npy'))
        arr = np.concatenate([np.load(p) for p in shards])
        if len(arr) != len(rows):
            raise SystemExit(f'{combo_name(*c)}: {len(arr)} vectors for {len(rows)} rows')
        np.save(out / f'{combo_name(*c)}.npy', arr)
        for p in shards:
            p.unlink()
    shutil.rmtree(work, ignore_errors=True)
    meta = {
        **run_info(),
        'dataset': args.dataset,
        'rows': len(rows),
        'combos': [combo_name(*c) for c in todo],
        'wall_s': time.time() - t0,
        'encode_s': encode_s,
        'torch': torch.__version__,
        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(0),
    }
    (out / f'extract_{args.dataset}_{int(t0)}.json').write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=sorted(protocol.SETS))
    ap.add_argument('--models', nargs='+', required=True, choices=sorted(enc.SPECS))
    ap.add_argument(
        '--variants', nargs='+', default=['centre', 'letterbox'], choices=sorted(VARIANTS)
    )
    ap.add_argument('--workers', type=int, default=6)
    print(json.dumps(run(ap.parse_args()), indent=1))


if __name__ == '__main__':
    main()
