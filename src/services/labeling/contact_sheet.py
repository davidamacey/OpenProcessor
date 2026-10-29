"""Contact sheets: seeded member selection, letterboxed grid rendering and a rebuildable manifest.

A sheet is one PNG of numbered thumbnails that a judge (a hosted model, a local VLM or a person)
looks at in a single call. This module is pure: the caller supplies crop bytes, nothing here reads
OpenSearch or the network. Three layouts share one renderer: a cluster sheet (members of one
group), a pair sheet (one pair per row, labelled A1/B1 ...) and a search sheet (the query in the
first cell, then the results).

Everything is deterministic. Selection and cell order come from ``random.Random`` seeded with a
string built from the seed and the sheet id, over members sorted by id, so input order never
matters. Digits are drawn from the 5x7 bitmap font below, so no font file (and no Pillow default
font that varies by build) is involved, and the PNG carries no timestamp. The manifest records
what was drawn where; ``rebuild_sheet`` re-renders from it and refuses when a source crop changed
or the PNG hash differs.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import PIL
from PIL import Image, ImageDraw


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


BUILDER_VERSION = 'cs1'
GUTTER_PX = 4
DEFAULT_GRID = (6, 4)
DEFAULT_THUMB_PX = 160
MAX_PROTOTYPES = 8
MAX_PAIRS = 12
GREY = (128, 128, 128)
_LABEL_BG = (24, 24, 24)
_LABEL_FG = (255, 255, 255)
_GLYPH_SCALE = 2
_GLYPH_PAD = 2

GLYPHS: dict[str, tuple[str, ...]] = {
    '0': ('01110', '10001', '10011', '10101', '11001', '10001', '01110'),
    '1': ('00100', '01100', '00100', '00100', '00100', '00100', '01110'),
    '2': ('01110', '10001', '00001', '00010', '00100', '01000', '11111'),
    '3': ('11110', '00001', '00001', '01110', '00001', '00001', '11110'),
    '4': ('00010', '00110', '01010', '10010', '11111', '00010', '00010'),
    '5': ('11111', '10000', '11110', '00001', '00001', '10001', '01110'),
    '6': ('00110', '01000', '10000', '11110', '10001', '10001', '01110'),
    '7': ('11111', '00001', '00010', '00100', '01000', '01000', '01000'),
    '8': ('01110', '10001', '10001', '01110', '10001', '10001', '01110'),
    '9': ('01110', '10001', '10001', '01111', '00001', '00010', '01100'),
    'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'),
    'B': ('11110', '10001', '10001', '11110', '10001', '10001', '11110'),
    'Q': ('01110', '10001', '10001', '10001', '10101', '10010', '01101'),
}


class SheetRebuildError(ValueError):
    """The manifest cannot be reproduced from the supplied crops."""


@dataclass(frozen=True)
class SheetMember:
    """One crop a sheet may show; ``distance`` is to its best prototype when known."""

    item_id: str
    image: bytes
    box_id: str = ''
    bbox_norm: tuple[float, float, float, float] | None = None
    is_prototype: bool = False
    distance: float | None = None


@dataclass(frozen=True)
class Sheet:
    sheet_id: str
    png: bytes
    manifest: dict[str, Any]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def member_key(item_id: str, box_id: str) -> str:
    return f'{item_id}#{box_id}'


def cell_to_item(manifest: Mapping[str, Any], cell: int) -> tuple[str, str]:
    """Map a 1-based cell number from a verdict back to ``(item_id, box_id)``."""
    for c in manifest['cells']:
        if c['cell'] == cell:
            return c['item_id'], c['box_id']
    raise KeyError(f'cell {cell} is not on sheet {manifest["sheet_id"]}')


def letterbox_geometry(width: int, height: int, thumb_px: int) -> tuple[int, int, int, int]:
    """Fit ``width x height`` inside a square without stretching: ``(w, h, x, y)`` in the cell."""
    scale = thumb_px / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return new_w, new_h, (thumb_px - new_w) // 2, (thumb_px - new_h) // 2


def _rng(*parts: object) -> random.Random:
    return random.Random(':'.join(str(p) for p in parts))


def _sorted(members: Sequence[SheetMember]) -> list[SheetMember]:
    return sorted(members, key=lambda m: (m.item_id, m.box_id))


def select_members(
    members: Sequence[SheetMember], *, seed: int | str, selection_id: str, n_cells: int
) -> list[tuple[SheetMember, str]]:
    """Prototypes, then members nearest to them, outliers, and random fill; each with its role."""
    pool = _sorted(members)
    n = min(n_cells, len(pool))
    rng = _rng(seed, selection_id)
    n_out = 3 if n >= 12 else 2 if n >= 6 else 0
    protos = [m for m in pool if m.is_prototype]
    if len(protos) > MAX_PROTOTYPES:
        protos = _sorted(rng.sample(protos, MAX_PROTOTYPES))
    protos = protos[: max(0, n - n_out)]
    chosen: dict[str, tuple[SheetMember, str]] = {
        member_key(m.item_id, m.box_id): (m, 'prototype') for m in protos
    }

    def free(with_distance: bool) -> list[SheetMember]:
        return [
            m
            for m in pool
            if member_key(m.item_id, m.box_id) not in chosen
            and (m.distance is not None or not with_distance)
        ]

    far = sorted(free(True), key=lambda m: (-(m.distance or 0.0), m.item_id, m.box_id))
    for m in far[: min(n_out, n - len(chosen))]:
        chosen[member_key(m.item_id, m.box_id)] = (m, 'outlier')
    near_slots = (n - len(chosen)) // 2
    near = sorted(free(True), key=lambda m: (m.distance or 0.0, m.item_id, m.box_id))
    for m in near[:near_slots]:
        chosen[member_key(m.item_id, m.box_id)] = (m, 'near')
    rest = free(False)
    for m in rng.sample(rest, min(n - len(chosen), len(rest))):
        chosen[member_key(m.item_id, m.box_id)] = (m, 'random')
    return sorted(chosen.values(), key=lambda t: (t[0].item_id, t[0].box_id))


def _draw_label(img: Image.Image, text: str, x: int, y: int) -> None:
    glyph_w = 5 * _GLYPH_SCALE
    width = len(text) * (glyph_w + _GLYPH_SCALE) - _GLYPH_SCALE + 2 * _GLYPH_PAD
    height = 7 * _GLYPH_SCALE + 2 * _GLYPH_PAD
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x + width - 1, y + height - 1), fill=_LABEL_BG)
    for i, ch in enumerate(text):
        ox = x + _GLYPH_PAD + i * (glyph_w + _GLYPH_SCALE)
        for row, bits in enumerate(GLYPHS[ch]):
            for col, bit in enumerate(bits):
                if bit == '1':
                    px = ox + col * _GLYPH_SCALE
                    py = y + _GLYPH_PAD + row * _GLYPH_SCALE
                    draw.rectangle(
                        (px, py, px + _GLYPH_SCALE - 1, py + _GLYPH_SCALE - 1), fill=_LABEL_FG
                    )


def _thumb(data: bytes, thumb_px: int) -> Image.Image:
    with Image.open(io.BytesIO(data)) as src:
        rgb = src.convert('RGB')
    w, h, x, y = letterbox_geometry(rgb.width, rgb.height, thumb_px)
    cell = Image.new('RGB', (thumb_px, thumb_px), GREY)
    cell.paste(rgb.resize((w, h), Image.Resampling.LANCZOS), (x, y))
    return cell


def _render(
    images: Sequence[bytes], labels: Sequence[str], cols: int, rows: int, thumb_px: int
) -> bytes:
    width = cols * thumb_px + (cols + 1) * GUTTER_PX
    height = rows * thumb_px + (rows + 1) * GUTTER_PX
    sheet = Image.new('RGB', (width, height), GREY)
    for i, (data, label) in enumerate(zip(images, labels, strict=True)):
        x = GUTTER_PX + (i % cols) * (thumb_px + GUTTER_PX)
        y = GUTTER_PX + (i // cols) * (thumb_px + GUTTER_PX)
        sheet.paste(_thumb(data, thumb_px), (x, y))
        _draw_label(sheet, label, x, y)
    buf = io.BytesIO()
    sheet.save(buf, 'PNG', optimize=False, compress_level=6)
    return buf.getvalue()


def _cell(n: int, label: str, m: SheetMember, role: str, **extra: Any) -> dict[str, Any]:
    return {
        'cell': n,
        'label': label,
        'item_id': m.item_id,
        'box_id': m.box_id,
        'bbox_norm': list(m.bbox_norm) if m.bbox_norm is not None else None,
        'content_hash': sha256_hex(m.image),
        'role': role,
        **extra,
    }


def _finish(
    *,
    sheet_id: str,
    layout: str,
    seed: int | str,
    kind: str,
    grid: tuple[int, int],
    thumb_px: int,
    cells: list[dict[str, Any]],
    images: Sequence[bytes],
    order_variant: int,
    selection_id: str,
) -> Sheet:
    png = _render(images, [c['label'] for c in cells], grid[0], grid[1], thumb_px)
    manifest = {
        'sheet_id': sheet_id,
        'layout': layout,
        'builder_version': BUILDER_VERSION,
        'renderer': f'pillow-{PIL.__version__}',
        'seed': seed,
        'kind': kind,
        'selection_id': selection_id,
        'order_variant': order_variant,
        'grid': list(grid),
        'thumb_px': thumb_px,
        'cells': cells,
        'png_sha256': sha256_hex(png),
    }
    return Sheet(sheet_id=sheet_id, png=png, manifest=manifest)


def build_cluster_sheet(
    members: Sequence[SheetMember],
    *,
    sheet_id: str,
    seed: int | str,
    kind: str,
    grid: tuple[int, int] = DEFAULT_GRID,
    thumb_px: int = DEFAULT_THUMB_PX,
    selection_id: str | None = None,
    order_variant: int = 0,
) -> Sheet:
    """One cluster sheet; ``selection_id``/``order_variant`` re-render the same members reordered."""
    sel_id = selection_id or sheet_id
    picked = select_members(members, seed=seed, selection_id=sel_id, n_cells=grid[0] * grid[1])
    if not picked:
        raise ValueError('a sheet needs at least one member')
    _rng(seed, sheet_id, 'order', order_variant).shuffle(picked)
    cols = min(grid[0], len(picked))
    rows = math.ceil(len(picked) / cols)
    cells = [_cell(i + 1, str(i + 1), m, role) for i, (m, role) in enumerate(picked)]
    return _finish(
        sheet_id=sheet_id,
        layout='cluster',
        seed=seed,
        kind=kind,
        grid=(cols, rows),
        thumb_px=thumb_px,
        cells=cells,
        images=[m.image for m, _ in picked],
        order_variant=order_variant,
        selection_id=sel_id,
    )


def build_pair_sheet(
    pairs: Sequence[tuple[SheetMember, SheetMember]],
    *,
    sheet_id: str,
    seed: int | str,
    kind: str,
    thumb_px: int = DEFAULT_THUMB_PX,
) -> Sheet:
    """Up to 12 pairs, one per row, A1/B1 ...; the sides of each pair are swapped at random."""
    if not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError(f'a pair sheet holds 1 to {MAX_PAIRS} pairs, got {len(pairs)}')
    rng = _rng(seed, sheet_id, 'swap')
    cells: list[dict[str, Any]] = []
    images: list[bytes] = []
    for row, (first, second) in enumerate(pairs, start=1):
        swapped = rng.random() < 0.5
        left, right = (second, first) if swapped else (first, second)
        for side, m in (('A', left), ('B', right)):
            cells.append(_cell(len(cells) + 1, f'{side}{row}', m, 'pair', row=row, swapped=swapped))
            images.append(m.image)
    return _finish(
        sheet_id=sheet_id,
        layout='pair',
        seed=seed,
        kind=kind,
        grid=(2, len(pairs)),
        thumb_px=thumb_px,
        cells=cells,
        images=images,
        order_variant=0,
        selection_id=sheet_id,
    )


def build_search_sheet(
    query: SheetMember,
    results: Sequence[SheetMember],
    *,
    sheet_id: str,
    seed: int | str,
    kind: str,
    grid: tuple[int, int] = DEFAULT_GRID,
    thumb_px: int = DEFAULT_THUMB_PX,
) -> Sheet:
    """The query in the first cell, then up to ``cols x rows - 1`` results in shuffled order."""
    ranked = list(enumerate(results[: grid[0] * grid[1] - 1], start=1))
    _rng(seed, sheet_id, 'order', 0).shuffle(ranked)
    cells = [_cell(1, 'Q', query, 'query')]
    images = [query.image]
    for rank, m in ranked:
        cells.append(_cell(len(cells) + 1, str(len(cells)), m, 'result', rank=rank))
        images.append(m.image)
    cols = min(grid[0], len(cells))
    return _finish(
        sheet_id=sheet_id,
        layout='search',
        seed=seed,
        kind=kind,
        grid=(cols, math.ceil(len(cells) / cols)),
        thumb_px=thumb_px,
        cells=cells,
        images=images,
        order_variant=0,
        selection_id=sheet_id,
    )


def rebuild_sheet(manifest: Mapping[str, Any], crops: Mapping[str, bytes]) -> Sheet:
    """Re-render a sheet from its manifest; refuses a changed crop or a differing PNG hash."""
    images: list[bytes] = []
    for c in manifest['cells']:
        key = member_key(c['item_id'], c['box_id'])
        data = crops.get(key)
        if data is None:
            raise SheetRebuildError(f'crop {key} is missing')
        if sha256_hex(data) != c['content_hash']:
            raise SheetRebuildError(f'content_hash of {key} changed since the sheet was built')
        images.append(data)
    cols, rows = manifest['grid']
    png = _render(images, [c['label'] for c in manifest['cells']], cols, rows, manifest['thumb_px'])
    if sha256_hex(png) != manifest['png_sha256']:
        raise SheetRebuildError(
            f'png_sha256 differs for sheet {manifest["sheet_id"]} (renderer {manifest["renderer"]}'
            f' vs pillow-{PIL.__version__})'
        )
    return Sheet(sheet_id=manifest['sheet_id'], png=png, manifest=dict(manifest))
