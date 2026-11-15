#!/usr/bin/env python3
"""The photo lists of the native engine's N0 comparison (plan 17 section 5, N0 "Inputs"): built
from header reads only, by one rule, so every arm of every hold sends the same photos.

The rule: keep baseline or extended-sequential, 8-bit, 3-component JPEGs (SOF0 or SOF1) whose
embedded ICC profile, if any, is sRGB-class by ``src/utils/icc_srgb.py`` (the colour step is not
in N0, so a photo that needs it would answer ``decoder`` and the row would not measure the
pipeline). Then the pixel window, in the pin's row order: shuffle with ``random.Random(seed)``
and take the first ``n``. Presets:

    coco   up to 2 MP, seed 1, 1,000 photos      (``coco_bench_2000.json``)
    hires  18 to 24.5 MP, seed 3, 120 photos     (``commons_hires_500.json``)

and ``--preset custom`` takes every number from the command line (set P: the coordinator's list
through ``--from-list``, same header rule, only counts and rates ever leave the host).

For each list the script writes ``<name>.txt`` (one absolute path per line), and ``<name>.json``
with the list's SHA-256 (of the file's bytes, the figure ``op-fe-loadgen`` records in every row),
the rule's parameters, the pin's SHA-256 and the count excluded under each reason. Only the first
bytes of a file are read, as many as the header needs.

    python scripts/bench/frame_engine_sets.py --preset coco \\
        --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images-dir data/samples/coco_bench_2000/images --out-dir artifacts_local/native_engine/n0/lists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

# ``python scripts/bench/frame_engine_sets.py`` puts scripts/bench, not the repository, on the path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.domain.jpeg_markers import (
    ICC_MARKER,
    ICC_TAG,
    SOF_MARKERS,
    header_segments,
    join_icc_chunks,
)


#: SOF0 (baseline) and SOF1 (extended sequential, Huffman): what nvJPEG's GPU backends take.
KEPT_MARKERS = frozenset({0xC0, 0xC1})
#: The prefix read first, doubled until the frame header (and a whole ICC profile) is inside it.
FIRST_READ = 64 * 1024
LAST_READ = 64 * 1024 * 1024
MEGAPIXEL = 1_000_000

NOT_JPEG = 'not_jpeg'
NO_FRAME_HEADER = 'no_frame_header'
SOF_NOT_SEQUENTIAL = 'sof_not_baseline_or_extended_sequential'
PRECISION = 'precision_not_8'
COMPONENTS = 'components_not_3'
PROFILE = 'profile_not_srgb'
BELOW_WINDOW = 'below_pixel_window'
ABOVE_WINDOW = 'above_pixel_window'
MISSING = 'file_missing'


class Header(NamedTuple):
    """What the rule reads from a file's first bytes."""

    jpeg: bool
    sof: int | None
    precision: int
    height: int
    width: int
    components: int
    icc: bytes | None
    #: The chunks of an ICC profile are all inside the bytes given (else read more).
    icc_complete: bool


def scan(data: bytes) -> Header:
    """The frame header and the ICC profile of the JPEG whose first bytes are ``data``."""
    if data[:2] != b'\xff\xd8':
        return Header(False, None, 0, 0, 0, 0, None, True)
    chunks: list[bytes] = []
    sof, precision, height, width, components = None, 0, 0, 0, 0
    for marker, payload in header_segments(data):
        if marker == ICC_MARKER and payload[:12] == ICC_TAG:
            chunks.append(payload)
        elif marker in SOF_MARKERS and len(payload) >= 6:
            precision, height, width, components = struct.unpack('>BHHB', payload[:6])
            sof = marker
            break
    icc = join_icc_chunks(chunks) if chunks else None
    return Header(
        True, sof, precision, height, width, components, icc, not chunks or icc is not None
    )


def read_header(path: Path) -> Header:
    """:func:`scan` of a file, reading a longer prefix until the header is whole: a frame header
    found, and any ICC profile complete (a camera's profile can be hundreds of kilobytes)."""
    with path.open('rb') as f:
        size = FIRST_READ
        while True:
            f.seek(0)
            data = f.read(size)
            header = scan(data)
            whole = len(data) < size or size >= LAST_READ
            if whole or (header.sof is not None and header.icc_complete) or not header.jpeg:
                return header
            size *= 4


def _format_reason(header: Header) -> str | None:
    """The reason the file's coding is not one the GPU route takes, or ``None``."""
    if not header.jpeg:
        return NOT_JPEG
    if header.sof is None:
        return NO_FRAME_HEADER
    if header.sof not in KEPT_MARKERS:
        return SOF_NOT_SEQUENTIAL
    if header.precision != 8:
        return PRECISION
    return COMPONENTS if header.components != 3 else None


def _window_reason(header: Header, window: tuple[float, float] | None) -> str | None:
    if window is None:
        return None
    pixels = header.width * header.height
    if pixels < window[0] * MEGAPIXEL:
        return BELOW_WINDOW
    return ABOVE_WINDOW if pixels > window[1] * MEGAPIXEL else None


def exclusion(
    header: Header, window: tuple[float, float] | None, profile_kind: Callable[[bytes | None], str]
) -> str | None:
    """Why the file is not kept, or ``None``. Checked in this order (coding, profile, pixel
    window), so each file is counted once, under the first reason."""
    reason = _format_reason(header)
    if reason is None and profile_kind(header.icc) != 'srgb':
        reason = PROFILE
    return reason or _window_reason(header, window)


@dataclass(frozen=True)
class Rule:
    """The numbers of one list."""

    name: str
    seed: int
    n: int
    min_mp: float | None
    max_mp: float | None

    @property
    def window(self) -> tuple[float, float] | None:
        if self.min_mp is None and self.max_mp is None:
            return None
        return (self.min_mp or 0.0, self.max_mp if self.max_mp is not None else float('inf'))


PRESETS: dict[str, Rule] = {
    'coco': Rule('coco', seed=1, n=1000, min_mp=None, max_mp=2.0),
    'hires': Rule('hires', seed=3, n=120, min_mp=18.0, max_mp=24.5),
}


class Selection(NamedTuple):
    paths: list[Path]
    considered: int
    kept: int
    excluded: dict[str, int]


def select(
    paths: Iterable[Path],
    rule: Rule,
    profile_kind: Callable[[bytes | None], str],
    header_of: Callable[[Path], Header] = read_header,
) -> Selection:
    """Apply the rule to ``paths`` in their given (the pin's row) order, then shuffle the kept
    ones with ``random.Random(rule.seed)`` and keep the first ``rule.n``."""
    kept: list[Path] = []
    excluded: Counter[str] = Counter()
    considered = 0
    for path in paths:
        considered += 1
        try:
            header = header_of(path)
        except FileNotFoundError:
            excluded[MISSING] += 1
            continue
        reason = exclusion(header, rule.window, profile_kind)
        if reason is None:
            kept.append(path)
        else:
            excluded[reason] += 1
    shuffled = list(kept)
    random.Random(rule.seed).shuffle(shuffled)
    return Selection(shuffled[: rule.n], considered, len(kept), dict(sorted(excluded.items())))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_paths(manifest: Path, images_dir: Path) -> list[Path]:
    """The pin's files in its row order, under ``images_dir``."""
    rows = json.loads(manifest.read_text())['images']
    return [(images_dir / row['file_name']).resolve() for row in rows]


def list_paths(listing: Path) -> list[Path]:
    """The files a list names (one path per line, ``#`` comments), in order."""
    lines = (ln.strip() for ln in listing.read_text().splitlines())
    return [Path(ln).resolve() for ln in lines if ln and not ln.startswith('#')]


def write_list(out_dir: Path, rule: Rule, sel: Selection, source: dict[str, Any]) -> dict[str, Any]:
    """Write ``<name>.txt`` and ``<name>.json``; return the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f'{rule.name}.txt'
    listing.write_text(''.join(f'{p}\n' for p in sel.paths))
    report = {
        'name': rule.name,
        'list': str(listing),
        'list_sha256': sha256_file(listing),
        'photos': len(sel.paths),
        'rule': {
            'seed': rule.seed,
            'n': rule.n,
            'min_megapixels': rule.min_mp,
            'max_megapixels': rule.max_mp,
            'jpeg': 'SOF0 or SOF1, 8-bit, 3 components, ICC profile absent or sRGB-class (src/utils/icc_srgb.py)',
            'order': 'rows in source order, kept files shuffled by random.Random(seed), first n',
        },
        'source': source,
        'considered': sel.considered,
        'kept_before_cut': sel.kept,
        'excluded_by_reason': sel.excluded,
        'excluded_total': sum(sel.excluded.values()),
    }
    (out_dir / f'{rule.name}.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def rule_of(args: argparse.Namespace) -> Rule:
    if args.preset != 'custom':
        base = PRESETS[args.preset]
        return Rule(
            args.name or base.name,
            args.seed if args.seed is not None else base.seed,
            args.n if args.n is not None else base.n,
            args.min_mp if args.min_mp is not None else base.min_mp,
            args.max_mp if args.max_mp is not None else base.max_mp,
        )
    if args.name is None or args.seed is None or args.n is None:
        raise SystemExit('--preset custom needs --name, --seed and --n')
    return Rule(args.name, args.seed, args.n, args.min_mp, args.max_mp)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--preset', choices=[*PRESETS, 'custom'], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', type=Path, help='the pin (JSON with an "images" list)')
    src.add_argument('--from-list', type=Path, help='a list of paths instead of a pin')
    ap.add_argument('--images-dir', type=Path, help='with --manifest: where the files are')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--name', help='list name (default: the preset)')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--n', type=int)
    ap.add_argument('--min-mp', type=float)
    ap.add_argument('--max-mp', type=float)
    args = ap.parse_args(argv)
    if args.manifest and not args.images_dir:
        ap.error('--manifest needs --images-dir')
    from src.utils.icc_srgb import profile_kind

    rule = rule_of(args)
    if args.manifest:
        paths = manifest_paths(args.manifest, args.images_dir)
        source = {'manifest': str(args.manifest), 'manifest_sha256': sha256_file(args.manifest)}
    else:
        paths = list_paths(args.from_list)
        source = {'from_list': str(args.from_list), 'from_list_sha256': sha256_file(args.from_list)}
    report = write_list(args.out_dir, rule, select(paths, rule, profile_kind), source)
    print(
        json.dumps(
            {k: report[k] for k in ('name', 'list', 'list_sha256', 'photos', 'excluded_by_reason')}
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
