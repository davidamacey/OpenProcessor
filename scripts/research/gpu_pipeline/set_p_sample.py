"""Copy a seeded sample of a private photo set to local NVMe under content-hash names.

The source is read only. Files are chosen by a seeded shuffle of the sorted relative paths, copied
byte for byte, and verified by SHA-256. The destination holds ``<sha256[:16]>.jpg`` only; the
manifest (relative source name, size, SHA-256 per copy) goes to ``--manifest-out``, a path outside
the repository. The printed result carries counts only.

    set_p_sample.py --src DIR --dst DIR --manifest-out FILE --n 1000 --seed 20261104
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any


CHUNK = 8 * 1024 * 1024
JPEG_SUFFIXES = ('.jpg', '.jpeg')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def list_jpegs(src: Path) -> list[str]:
    """Sorted relative paths of every JPEG under ``src``."""
    found: list[str] = []
    for root, _dirs, files in os.walk(src):
        found.extend(
            str((Path(root) / f).relative_to(src))
            for f in files
            if f.lower().endswith(JPEG_SUFFIXES)
        )
    return sorted(found)


def choose(relative: list[str], n: int, seed: int) -> list[str]:
    """The seeded draw: shuffle the sorted list with ``Random(seed)`` and keep the first ``n``."""
    shuffled = sorted(relative)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


def copy_sample(src: Path, dst: Path, picks: list[str]) -> list[dict[str, Any]]:
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rel in picks:
        source = src / rel
        digest = sha256_file(source)
        target = dst / f'{digest[:16]}.jpg'
        shutil.copyfile(source, target)
        if sha256_file(target) != digest:
            raise OSError(f'copy of one sample file differs from its source: {target.name}')
        rows.append(
            {
                'name': target.name,
                'source_rel': rel,
                'sha256': digest,
                'bytes': target.stat().st_size,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--dst', type=Path, required=True)
    p.add_argument('--manifest-out', type=Path, required=True)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261104)
    args = p.parse_args(argv)
    picks = choose(list_jpegs(args.src), args.n, args.seed)
    rows = copy_sample(args.src, args.dst, picks)
    args.manifest_out.write_text(json.dumps({'seed': args.seed, 'files': rows}, indent=1))
    print(json.dumps({'copied': len(rows), 'bytes': sum(r['bytes'] for r in rows)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
