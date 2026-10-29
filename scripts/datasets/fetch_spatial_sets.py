#!/usr/bin/env python3
"""Fetch or verify the public multi-view / relocalisation datasets used for spatial-memory research.

The archives are large (about 150 GB for everything), so this tool is resumable and size-checked
against the server rather than hashed: every download goes to ``<name>.part`` with HTTP Range
resume, is renamed only when its size equals the server's, and a rerun skips files that already
match. ``verify`` compares local sizes to the server; ``--test-archives`` additionally runs the
format's own integrity test (``unzip -t``, ``7z t``, ``tar -t``).

    fetch_spatial_sets.py list
    fetch_spatial_sets.py fetch  --root data/spatial --only 7-scenes --only eth3d
    fetch_spatial_sets.py verify --root data/spatial --test-archives

Tanks and Temples is not listed: its archives sit behind a manual form on tanksandtemples.org.
Run on the host, not inside a container. The data is not committed. Licences differ per dataset
(several are non-commercial); read docs/research/spatial_datasets.md before redistributing.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess  # nosec B404  runs the archive tools unzip, 7z and tar on local files only
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import USER_AGENT, FetchError


logger = logging.getLogger('datasets.spatial')

_CAM = 'https://www.repository.cam.ac.uk/bitstream/handle/1810'
_MSR = 'https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8'
_CVUT = 'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization'
_IMC = 'https://www.cs.ubc.ca/research/kmyi_data/imw2020'
_GLD = 'https://s3.amazonaws.com/google-landmark'

# dataset -> [(path relative to <root>/<dataset>, url)]
REGISTRY: dict[str, list[tuple[str, str]]] = {
    '7-scenes': [
        (f'{s}.zip', f'{_MSR}/{s}.zip')
        for s in ('chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs')
    ],
    'cambridge-landmarks': [
        (f'{n}.zip', f'{_CAM}/{h}/{n}.zip')
        for n, h in (
            ('KingsCollege', 251342),
            ('OldHospital', 251340),
            ('ShopFacade', 251336),
            ('StMarysChurch', 251294),
            ('GreatCourt', 251291),
        )
    ],
    'aachen-day-night': [
        ('aachen_v1_1.zip', f'{_CVUT}/Aachen-Day-Night/aachen_v1_1.zip'),
        (
            'images/database_and_query_images.zip',
            f'{_CVUT}/Aachen-Day-Night/images/database_and_query_images.zip',
        ),
    ],
    'robotcar-seasons': [
        (f'images/{c}.zip', f'{_CVUT}/RobotCar-Seasons/images/{c}.zip')
        for c in (
            'dawn',
            'dusk',
            'night',
            'night-rain',
            'overcast-reference',
            'overcast-summer',
            'overcast-winter',
            'rain',
            'snow',
            'sun',
        )
    ],
    'mip-nerf-360': [
        ('360_v2.zip', 'https://storage.googleapis.com/gresearch/refraw360/360_v2.zip'),
        (
            '360_extra_scenes.zip',
            'https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip',
        ),
    ],
    'eth3d': [
        (f'multi_view_training_{k}.7z', f'https://www.eth3d.net/data/multi_view_training_{k}.7z')
        for k in ('dslr_undistorted', 'dslr_scan_eval', 'rig_undistorted', 'rig_scan_eval')
    ],
    'imc-phototourism': [
        (f'train/{s}.tar.gz', f'{_IMC}/TrainingData/{s}.tar.gz')
        for s in ('brandenburg_gate', 'reichstag', 'sacre_coeur', 'temple_nara_japan')
    ]
    + [
        (f'test/{s}.tar.gz', f'{_IMC}/TestData/{s}.tar.gz')
        for s in (
            'british_museum',
            'florence_cathedral_side',
            'milan_cathedral',
            'mount_rushmore',
            'piazza_san_marco',
            'sagrada_familia',
            'st_pauls_cathedral',
        )
    ],
    'google-landmarks-v2-subset': [
        (f'train/images_{i:03d}.tar', f'{_GLD}/train/images_{i:03d}.tar') for i in range(6)
    ]
    + [
        (f'md5sum/md5.images_{i:03d}.txt', f'{_GLD}/md5sum/train/md5.images_{i:03d}.txt')
        for i in range(6)
    ]
    + [('metadata/train_clean.csv', f'{_GLD}/metadata/train_clean.csv')],
    'coco-val2017': [
        ('val2017.zip', 'http://images.cocodataset.org/zips/val2017.zip'),
        (
            'annotations_trainval2017.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        ),
    ],
}


def _open(url: str, start: int = 0, timeout: int = 60):  # type: ignore[no-untyped-def]
    headers = {'User-Agent': USER_AGENT, 'Range': f'bytes={start}-'}
    return urllib.request.urlopen(  # nosec B310  https dataset URLs from REGISTRY
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


def remote_size(url: str) -> int:
    """Total size in bytes, from a one-byte Range request (HEAD is unreliable on some hosts)."""
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310  https dataset URLs from REGISTRY
        m = re.search(r'/(\d+)$', resp.headers.get('Content-Range', ''))
        if m:
            return int(m.group(1))
        length = resp.headers.get('Content-Length')
        if length and resp.status == 200:
            return int(length)
    raise FetchError(f'{url}: server reported no total size')


def download_resumable(url: str, dest: Path, total: int, rate_mb_s: float) -> None:
    """Resume ``<dest>.part`` with Range, then rename once the size equals ``total``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        if have == total:
            break
        try:
            with _open(url, have) as resp, part.open('ab') as out:
                if resp.status == 200 and have:  # server ignored Range: restart cleanly
                    out.truncate(0)
                budget = rate_mb_s * (1 << 20)
                t0, sent = time.monotonic(), 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    sent += len(chunk)
                    ahead = sent / budget - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning('%s: attempt %d failed: %s', dest.name, attempt, exc)
            time.sleep(min(2**attempt, 30))
    if not part.exists() or part.stat().st_size != total:
        raise FetchError(f'{dest.name}: incomplete after retries (kept {part.name} for resume)')
    part.replace(dest)


def test_archive(path: Path) -> bool:
    """Run the format's own integrity test; False when the archive is damaged or truncated."""
    name = path.name
    if name.endswith('.zip'):
        cmd = ['unzip', '-tqq', str(path)]
    elif name.endswith('.7z'):
        cmd = ['7z', 't', '-bd', str(path)]
    elif name.endswith(('.tar.gz', '.tar')):
        cmd = ['tar', '-tf', str(path)]
    else:
        return True
    if shutil.which(cmd[0]) is None:
        raise FetchError(f'{cmd[0]} is not installed')
    proc = subprocess.run(['nice', '-n', '10', *cmd], capture_output=True, check=False)  # nosec B603 B607  fixed argv, local file
    return proc.returncode == 0


def _select(only: list[str]) -> dict[str, list[tuple[str, str]]]:
    unknown = set(only) - set(REGISTRY)
    if unknown:
        raise FetchError(f'unknown dataset(s) {sorted(unknown)}; try the list command')
    return {k: v for k, v in REGISTRY.items() if not only or k in only}


def run(args: argparse.Namespace) -> int:
    chosen = _select(args.only)
    if args.command == 'list':
        for name, files in chosen.items():
            print(f'{name}: {len(files)} files')
        return 0
    root = Path(args.root)
    bad = 0
    for name, files in chosen.items():
        for rel, url in files:
            dest = root / name / rel
            total = remote_size(url)
            size = dest.stat().st_size if dest.is_file() else -1
            if args.command == 'fetch' and size != total:
                logger.info('fetching %s/%s (%.1f MB)', name, rel, total / 1e6)
                download_resumable(url, dest, total, args.rate_mb_s)
                size = total
            ok = size == total and (not args.test_archives or test_archive(dest))
            bad += not ok
            print(f'{"OK  " if ok else "BAD "} {name}/{rel} local={size} remote={total}')
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('command', choices=('list', 'fetch', 'verify'))
    ap.add_argument(
        '--root', default='data/spatial', help='dataset root (one directory per dataset)'
    )
    ap.add_argument('--only', action='append', default=[], help='dataset name; repeatable')
    ap.add_argument(
        '--rate-mb-s', type=float, default=25.0, help='download cap, MiB/s (default 25)'
    )
    ap.add_argument(
        '--test-archives', action='store_true', help='also run unzip -t / 7z t / tar -t'
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        return run(args)
    except FetchError as exc:
        logger.error('%s', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
