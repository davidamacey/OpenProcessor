"""nd:P2: exact and perceptual tiers. Registration: ``docs/research/near_duplicate_detection.md`` section 1.10.

Subcommands:

* ``exact``: ``imohash`` collisions against SHA-256 over every file of the given roots, and SHA-256 cost;
* ``cost``: PDQ timings (full decode, 512 px tile, JPEG draft decode) on large images;
* ``perceptual``: PDQ, pHash and dHash alone and as an extra edge over the embedding edges at the P1 cut.

The SSCD copy-model arm is cut from v0.7.0 and is not implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.near_dup import group, hashes, sweep


SIZE_BUCKETS = (
    (0, 128 << 10, '<128KiB'),
    (128 << 10, 1 << 20, '128KiB-1MiB'),
    (1 << 20, 4 << 20, '1-4MiB'),
    (4 << 20, 1 << 40, '>=4MiB'),
)
FILE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
INGEST_MS = (
    1000.0 / 35.8
)  # mid of the measured 35.0-36.6 img/s on the private photo set (v070_measurement_log E2)
HASH_COST_BAR = 0.05


def _bucket(size: int) -> str:
    return next(name for lo, hi, name in SIZE_BUCKETS if lo <= size < hi)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in FILE_SUFFIXES)


def _both(path: str) -> tuple[str, str, int]:
    return hashes.sha256_file(path), hashes.imohash_file(path), Path(path).stat().st_size


def exact(roots: dict[str, Path], repeat: int, workers: int) -> dict:
    files = [(name, p) for name, root in roots.items() for p in _files(root)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_both, [str(p) for _, p in files], chunksize=64))
    by_imo: dict[str, set[str]] = defaultdict(set)
    sizes_of: dict[str, set[int]] = defaultdict(set)
    by_sha: dict[str, int] = defaultdict(int)
    for sha, imo, size in rows:
        by_imo[imo].add(sha)
        sizes_of[imo].add(size)
        by_sha[sha] += 1
    collisions = [(imo, shas) for imo, shas in by_imo.items() if len(shas) > 1]
    sets = {name: sum(1 for n, _ in files if n == name) for name in roots}
    sample = sorted(files, key=lambda f: f[1].as_posix())[:: max(1, len(files) // 3000)]
    timing = _sha_cost([p for _, p in sample], repeat)
    return {
        'files': len(files),
        'files_per_set': sets,
        'distinct_sha256': len(by_sha),
        'byte_identical_extra_copies': sum(c - 1 for c in by_sha.values()),
        'imohash_collisions': len(collisions),
        'collision_sizes': sorted({s for imo, _ in collisions for s in sizes_of[imo]}),
        'sha256_cost': timing,
    }


def _sha_cost(paths: list[Path], repeat: int) -> dict:
    """SHA-256 cost from a warm page cache: a throwaway pass, then ``repeat`` timed passes."""
    import time

    for p in paths:
        hashes.sha256_file(p)
    per_bucket: dict[str, list[tuple[int, float]]] = defaultdict(list)
    mbps = []
    for _ in range(repeat):
        total_bytes, total_s = 0, 0.0
        for p in paths:
            size = p.stat().st_size
            t0 = time.perf_counter()
            hashes.sha256_file(p)
            dt = time.perf_counter() - t0
            total_bytes += size
            total_s += dt
            per_bucket[_bucket(size)].append((size, dt))
        mbps.append(total_bytes / total_s / 1e6)
    buckets = {}
    for name, vals in per_bucket.items():
        ms = [dt * 1000.0 for _, dt in vals]
        buckets[name] = {
            'files_timed': len(vals) // repeat,
            'median_ms_per_file': statistics.median(ms),
            'mean_bytes': statistics.fmean(s for s, _ in vals),
        }
    return {
        'files_timed': len(paths),
        'repeats': repeat,
        'mb_per_s_median': statistics.median(mbps),
        'mb_per_s_range': [min(mbps), max(mbps)],
        'by_size': buckets,
        'load_1min': os.getloadavg()[0],
    }


def cost(paths: list[Path], repeat: int) -> dict:
    """PDQ cost per image: full decode + PDQ, tile resize + PDQ, JPEG draft decode + PDQ."""
    rows = []
    for p in paths:
        with Image.open(p) as probe:
            w, h = probe.size

        def full(path: Path = p) -> None:
            hashes.pdq_bits(np.asarray(hashes.decode_rgb(path)))

        def tile(path: Path = p) -> None:
            img = hashes.decode_rgb(path)
            img.thumbnail((512, 512), Image.Resampling.BILINEAR)
            hashes.pdq_bits(np.asarray(img))

        def draft(path: Path = p) -> None:
            with Image.open(path) as im:
                im.draft('RGB', (512, 512))
                rgb = im.convert('RGB')
            rgb.thumbnail((512, 512), Image.Resampling.BILINEAR)
            hashes.pdq_bits(np.asarray(rgb))

        decoded = hashes.decode_rgb(p)

        def phash_only(img: Image.Image = decoded) -> None:
            hashes.phash_bits(img)

        def dhash_only(img: Image.Image = decoded) -> None:
            hashes.dhash_bits(img)

        def decode_only(path: Path = p) -> None:
            hashes.decode_rgb(path)

        rows.append(
            {
                'megapixels': w * h / 1e6,
                'bytes': p.stat().st_size,
                'decode_ms': hashes.timed(decode_only, repeat=repeat)[0],
                'phash_ms_after_decode': hashes.timed(phash_only, repeat=repeat)[0],
                'dhash_ms_after_decode': hashes.timed(dhash_only, repeat=repeat)[0],
                'full_pdq_ms': hashes.timed(full, repeat=repeat)[0],
                'tile_pdq_ms': hashes.timed(tile, repeat=repeat)[0],
                'draft_pdq_ms': hashes.timed(draft, repeat=repeat)[0]
                if p.suffix.lower() in {'.jpg', '.jpeg'}
                else None,
            }
        )
    keys = (
        'decode_ms',
        'phash_ms_after_decode',
        'dhash_ms_after_decode',
        'full_pdq_ms',
        'tile_pdq_ms',
        'draft_pdq_ms',
    )
    summary = {k: statistics.median([v for r in rows if (v := r[k]) is not None]) for k in keys}
    summary['median_megapixels'] = statistics.median(float(r['megapixels'] or 0) for r in rows)
    summary['ingest_ms_per_image'] = INGEST_MS
    summary['bar_ms_5pct'] = INGEST_MS * HASH_COST_BAR
    summary['load_1min'] = os.getloadavg()[0]
    # the hash step beyond the decode the pipeline already pays for: tile resize + PDQ, not the decode
    return {'images': len(rows), 'summary': summary, 'rows': rows}


def _hash_row(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    return hashes.all_hashes(path)


def compute_hashes(paths: list[str], workers: int) -> dict[str, np.ndarray]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_hash_row, paths, chunksize=32))
    return {
        'pdq': np.stack([o[0] for o in out]),
        'phash': np.stack([o[1] for o in out]),
        'dhash': np.stack([o[2] for o in out]),
        'quality': np.array([o[3] for o in out]),
    }


def _union_eval(
    pool: sweep.Pool, base_cut: float, hi: np.ndarray, hj: np.ndarray, t_mask: np.ndarray
) -> dict:
    ei, ej = pool.edges.at(base_cut)
    ii = np.concatenate([ei, hi[t_mask]])
    jj = np.concatenate([ej, hj[t_mask]])
    labels = group.components(pool.edges.n, ii, jj)
    return sweep.trim(group.evaluate(labels, pool.family, pool.transform, pool.original_index))


def perceptual(
    x: np.ndarray,
    rows: list[dict],
    hs: dict[str, np.ndarray],
    p1: dict,
    seeds: list[int],
    bits: dict[str, int],
) -> dict:
    """Hash alone and hash + embedding edge, on the test half of each P1 split."""
    n_fam = 1 + max(r['family'] for r in rows)
    out = []
    for split in p1['splits']:
        seed = split['seed']
        if seed not in seeds:
            continue
        _, test_f = sweep.split_families(n_fam, seed)
        idx = [i for i, r in enumerate(rows) if r['family'] in test_f]
        pool = sweep.Pool(x[idx], [rows[i] for i in idx])
        chosen = split['chosen']['single']
        base_cut = chosen if chosen is not None else sweep.CONTROL_CUT
        base = sweep.trim(pool.evaluate('single', base_cut, 50))
        entry: dict = {'seed': seed, 'base_cut': base_cut, 'embedding_only': base, 'hashes': {}}
        for name, words in hs.items():
            if name == 'quality':
                continue
            hi, hj, hd = hashes.hamming_pairs(words[idx], bits[name])
            grid = list(range(0, bits[name] + 1, 1 if bits[name] <= 32 else 2))
            alone, union = {}, {}
            for t in grid:
                mask = hd <= t
                empty = np.zeros(0, dtype=np.int64)
                labels = (
                    group.components(pool.edges.n, hi[mask], hj[mask])
                    if mask.any()
                    else group.components(pool.edges.n, empty, empty)
                )
                alone[str(t)] = sweep.trim(
                    group.evaluate(labels, pool.family, pool.transform, pool.original_index)
                )
                union[str(t)] = _union_eval(pool, base_cut, hi, hj, mask)
            entry['hashes'][name] = {'alone': alone, 'plus_embedding': union}
        out.append(entry)
    return {'splits': out}


def _load_pool(manifest: Path, vectors: Path) -> tuple[list[dict], np.ndarray]:
    rows = sweep.pool_from_manifest(json.loads(manifest.read_text()))
    x = np.load(vectors.with_suffix('.npy'))
    return rows, x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('exact')
    e.add_argument('--root', action='append', required=True, help='name=directory (repeatable)')
    e.add_argument('--repeat', type=int, default=3)
    e.add_argument('--workers', type=int, default=8)
    e.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('sha-cost', help='SHA-256 cost only, on a sample of the given roots')
    s.add_argument('--root', action='append', required=True, help='name=directory (repeatable)')
    s.add_argument('--repeat', type=int, default=3)
    s.add_argument('--sample', type=int, default=3000)
    s.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('cost')
    c.add_argument('--images', type=Path, required=True, help='directory of large images')
    c.add_argument('--limit', type=int, default=30)
    c.add_argument('--repeat', type=int, default=3)
    c.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('perceptual')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument(
        '--vectors', type=Path, required=True, help='encode.py prefix of the pe_frame vectors'
    )
    p.add_argument('--p1', type=Path, required=True, help='nd:P1 results JSON of the same space')
    p.add_argument(
        '--hash-cache', type=Path, required=True, help='npz of the pool hashes (made if missing)'
    )
    p.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit-originals', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == 'exact':
        roots = {s.split('=', 1)[0]: Path(s.split('=', 1)[1]) for s in args.root}
        result = exact(roots, args.repeat, args.workers)
    elif args.cmd == 'sha-cost':
        files = sorted(p for r in args.root for p in _files(Path(r.split('=', 1)[1])))
        result = _sha_cost(files[:: max(1, len(files) // args.sample)], args.repeat)
    elif args.cmd == 'cost':
        imgs = sorted(args.images.rglob('*.jp*g'), key=lambda q: -q.stat().st_size)[: args.limit]
        result = cost(imgs, args.repeat)
    else:
        rows, x = _load_pool(args.manifest, args.vectors)
        if args.limit_originals:
            keep = [i for i, r in enumerate(rows) if r['family'] < args.limit_originals]
            rows, x = [rows[i] for i in keep], x[keep]
        if args.hash_cache.exists():
            hs = dict(np.load(args.hash_cache))
        else:
            hs = compute_hashes([r['path'] for r in rows], args.workers)
            np.savez(args.hash_cache, **hs)  # type: ignore[arg-type]
        p1 = json.loads(args.p1.read_text())
        result = perceptual(x, rows, hs, p1, args.seeds, {'pdq': 64, 'phash': 32, 'dhash': 32})
        result['pdq_quality_median'] = float(np.median(hs['quality']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    sys.stdout.write(f'wrote {args.out}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
