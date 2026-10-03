"""Ingest storage-cost arithmetic for detector vocabulary and embedding choices.

Pure arithmetic: no OpenSearch or Triton access. Feed it per-image figures
measured on a stack (items per image, bytes per metadata-only item, bytes per
vector) and it prints the per-1,000-image storage and GPU-call table used in
docs/PERFORMANCE.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


KB = 1000.0
MB = 1000.0 * KB


@dataclass(frozen=True)
class Scenario:
    name: str
    items_per_image: float
    embedded_fraction: float


@dataclass(frozen=True)
class CostRow:
    name: str
    items: float
    embedded_items: float
    item_store_mb: float
    frame_vectors_mb: float
    total_mb: float
    crop_forwards: float


def cost_row(
    scenario: Scenario,
    *,
    n_images: int,
    metadata_bytes: float,
    vector_bytes: float,
    frame_vector_bytes: float,
) -> CostRow:
    """Storage and encoder-call cost of ingesting ``n_images`` under ``scenario``."""
    if not 0.0 <= scenario.embedded_fraction <= 1.0:
        raise ValueError('embedded_fraction must be within 0..1')
    items = scenario.items_per_image * n_images
    embedded = items * scenario.embedded_fraction
    item_store = (items * metadata_bytes + embedded * vector_bytes) / MB
    frames = n_images * frame_vector_bytes / MB
    return CostRow(
        name=scenario.name,
        items=items,
        embedded_items=embedded,
        item_store_mb=item_store,
        frame_vectors_mb=frames,
        total_mb=item_store + frames,
        crop_forwards=embedded,
    )


def ratios(rows: list[CostRow]) -> list[float]:
    """Total-size ratio of every row against the first (baseline) row."""
    base = rows[0].total_mb
    return [row.total_mb / base for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=int, default=1000)
    parser.add_argument('--metadata-bytes', type=float, default=1800.0)
    parser.add_argument('--vector-bytes', type=float, default=8400.0)
    parser.add_argument('--frame-vector-bytes', type=float, default=8400.0)
    parser.add_argument(
        '--narrow-items', type=float, default=1.37, help='items/image, narrow detector'
    )
    parser.add_argument(
        '--full-items', type=float, default=7.0, help='items/image, full vocabulary'
    )
    args = parser.parse_args()
    scenarios = [
        Scenario('narrow detector, embed all', args.narrow_items, 1.0),
        Scenario('full vocabulary, embed all', args.full_items, 1.0),
        Scenario('full vocabulary, embed none', args.full_items, 0.0),
        Scenario(
            'full vocabulary, embed narrow classes only',
            args.full_items,
            args.narrow_items / args.full_items,
        ),
    ]
    rows = [
        cost_row(
            s,
            n_images=args.images,
            metadata_bytes=args.metadata_bytes,
            vector_bytes=args.vector_bytes,
            frame_vector_bytes=args.frame_vector_bytes,
        )
        for s in scenarios
    ]
    print(f'{"scenario":45} {"items":>8} {"total MB":>9} {"vs first":>9} {"crop fwd":>9}')
    for row, ratio in zip(rows, ratios(rows), strict=True):
        print(
            f'{row.name:45} {row.items:8.0f} {row.total_mb:9.1f} '
            f'{ratio:8.2f}x {row.crop_forwards:9.0f}'
        )


if __name__ == '__main__':
    main()
