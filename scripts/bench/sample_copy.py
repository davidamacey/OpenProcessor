"""Copy a seeded sample to local NVMe under neutral names and write its manifest.

The input is a manifest from ``select_baseline_set.py local`` (the seeded draw). Each file is
copied byte for byte to ``DEST/000000.jpg``, ``000001.jpg``, ... (never moved, never linked),
the copy is checked by SHA-256 against its source, and a new manifest lists the copies. Neither
the manifest nor the printed result carries a source path or name, so a private set is reported
by numbers only.

    sample_copy.py --manifest src_manifest.txt --dest /nvme/bench/setp3000 \\
        --out artifacts_local/bench/v070/sp0/setp3000.txt --seed 20261029
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.select_baseline_set import Candidate, read_manifest, write_manifest


CHUNK = 8 * 1024 * 1024


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def copy_sample(manifest: Path, dest: Path, out: Path, *, seed: int) -> dict[str, int]:
    """Copy every file of ``manifest`` into an empty ``dest``; ``out`` lists the copies."""
    sources = read_manifest(manifest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f'{dest} is not empty; copies go to a fresh directory')
    copied: list[Candidate] = []
    for i, src in enumerate(sources):
        target = dest / f'{i:06d}.jpg'
        shutil.copyfile(src, target)
        if _sha256(str(target)) != _sha256(src):
            raise OSError(f'copy {target.name} differs from its source')
        copied.append(Candidate(target, target.stat().st_size, 'copy'))
    write_manifest(out, copied, seed=seed)
    return {'count': len(copied), 'bytes': sum(c.size for c in copied)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True, help='empty or new directory on NVMe')
    p.add_argument('--out', type=Path, required=True, help='manifest of the copies')
    p.add_argument('--seed', type=int, required=True, help='seed recorded in the new manifest')
    args = p.parse_args(argv)
    print(json.dumps(copy_sample(args.manifest, args.dest, args.out, seed=args.seed)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
