"""Encode throughput and peak GPU memory per model, 3 repetitions on synthetic batches.

Run under the GPU lock in the research venv. Preprocessing is excluded: this times the encoder
alone, at batch 32, fp16 autocast. Re-embed time at N crops is N / throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.fine_grained.data import run_info


BATCH, BATCHES, REPS = 32, 16, 3


def bench(keys: list[str]) -> dict[str, Any]:
    import torch

    from scripts.research.fine_grained import models as enc

    out: dict[str, Any] = {
        **run_info(),
        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'models': {},
    }
    for key in keys:
        torch.cuda.reset_peak_memory_stats()
        e = enc.load(key)
        size = enc.SPECS[key].size
        batch = np.random.default_rng(0).integers(0, 255, (BATCH, size, size, 3), dtype=np.uint8)
        e.encode(batch)  # warm-up
        rates = []
        for _ in range(REPS):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(BATCHES):
                vec = e.encode(batch)
            torch.cuda.synchronize()
            rates.append(BATCH * BATCHES / (time.time() - t0))
        out['models'][enc.SPECS[key].name] = {
            'imgs_per_s': rates,
            'mean_imgs_per_s': float(np.mean(rates)),
            'peak_mem_mib': torch.cuda.max_memory_allocated() / 2**20,
            'dim': int(vec.shape[1]),
            'bytes_per_vector_fp32': int(vec.shape[1] * 4),
            'reembed_100k_s': 100_000 / float(np.mean(rates)),
            'reembed_1m_s': 1_000_000 / float(np.mean(rates)),
        }
        del e
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()
    Path(args.out).write_text(json.dumps(bench(args.models), indent=1))


if __name__ == '__main__':
    main()
