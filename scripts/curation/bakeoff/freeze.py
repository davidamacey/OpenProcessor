"""Freeze + verify the bake-off test split so the benchmark never drifts.

The benchmark is the training export's ``test/`` split (NOT a separate
export). Freezing records a content hash of the test labels and makes the
test images/labels read-only; verification recomputes the hash before any
(re-)evaluation and refuses to score if it changed. Relabeling continues
on train/val only.

CLI:
    python -m scripts.curation.bakeoff.freeze --dataset <root> --freeze
    python -m scripts.curation.bakeoff.freeze --dataset <root> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path


LOCK_NAME = 'TEST_FROZEN.json'


def test_sha(root: Path, *, split: str = 'test') -> tuple[str, int]:
    """SHA-256 over sorted ``(rel_label_path, sha(content))`` for the split.

    Returns ``(hex16, n_label_files)``. Hashing labels (not images) is
    enough: the labels define the ground truth the metric uses, and image
    pixels are produced deterministically by the export's resize step.
    """
    labels_dir = root / 'labels' / split
    h = hashlib.sha256()
    n = 0
    if labels_dir.is_dir():
        for f in sorted(labels_dir.rglob('*.txt')):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(b'\0')
            h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode())
            h.update(b'\n')
            n += 1
    return h.hexdigest()[:16], n


def _set_readonly(root: Path, *, split: str, writable: bool = False) -> None:
    """Toggle read-only on the split's image + label files."""
    for sub in ('images', 'labels'):
        d = root / sub / split
        if not d.is_dir():
            continue
        for f in d.rglob('*'):
            if f.is_file():
                mode = f.stat().st_mode
                if writable:
                    f.chmod(mode | stat.S_IWUSR)
                else:
                    f.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def freeze(root: Path, *, split: str = 'test') -> dict[str, object]:
    """Record the test-split hash and make the split read-only."""
    sha, n = test_sha(root, split=split)
    lock = {
        'frozen_test_sha': sha,
        'n_label_files': n,
        'split': split,
        'frozen_at': datetime.now(UTC).isoformat(),
    }
    (root / LOCK_NAME).write_text(json.dumps(lock, indent=2), encoding='utf-8')
    _set_readonly(root, split=split, writable=False)
    return lock


def verify(root: Path, *, split: str = 'test') -> tuple[bool, str]:
    """Recompute the hash and compare to the lock file.

    Returns ``(ok, message)``. ``ok`` is False if no lock exists or the
    hash changed --- callers MUST refuse to score in that case so model
    comparisons stay on identical data.
    """
    lock_path = root / LOCK_NAME
    if not lock_path.is_file():
        return False, f'no {LOCK_NAME}; test set was never frozen'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    current, n = test_sha(root, split=split)
    if current != lock.get('frozen_test_sha'):
        return False, (
            f'test set CHANGED: lock={lock.get("frozen_test_sha")} now={current} '
            f'({n} label files) --- benchmark would drift, refusing'
        )
    return True, f'verified: {current} ({n} label files)'


def main() -> int:
    p = argparse.ArgumentParser(description='Freeze/verify the bake-off test split.')
    p.add_argument('--dataset', type=Path, required=True, help='Export root')
    p.add_argument('--split', default='test')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', action='store_true')
    g.add_argument('--verify', action='store_true')
    g.add_argument('--unlock', action='store_true', help='Make split writable again')
    args = p.parse_args()

    if args.freeze:
        lock = freeze(args.dataset, split=args.split)
        print(f'frozen: {lock["frozen_test_sha"]} ({lock["n_label_files"]} labels), read-only')
        return 0
    if args.unlock:
        _set_readonly(args.dataset, split=args.split, writable=True)
        print('test split is writable again (unlocked)')
        return 0
    ok, msg = verify(args.dataset, split=args.split)
    print(msg)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
