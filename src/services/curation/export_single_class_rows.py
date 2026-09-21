"""Index-to-rows half of the single-class / class-subset dataset export.

Split out of :mod:`src.services.curation.export_single_class` along a
real seam, not a line count: this module answers *"which frames, with
which boxes, in which stratum"* by querying the items index, and knows
nothing about splits, label files, manifests or checksums. The service
module consumes :class:`_FrameRow` objects and never issues a query.

That separation is what lets the exporter support two very different box
sources — an item's own class-labeled bbox, or the region-of-interest
sub-annotation hanging off an item — behind one materialization path. It
is also the only half that touches deployment-specific document field
names, all of which arrive via
:class:`~src.config.region_fields.RegionFields` rather than literals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.config.region_state import RegionStatus
from src.services.curation.export_support import scroll_hits


if TYPE_CHECKING:
    from src.config import CurationConfig
    from src.config.region_fields import RegionFields
    from src.services.curation.export_single_class import SingleClassExportProfile


ImageMode = Literal['whole_frame', 'item_crop']

# Region states this exporter reads, by role. Values come from the
# canonical RegionStatus enum -- never literal strings.
POSITIVE_REGION_STATUSES: frozenset[str] = frozenset({RegionStatus.DETECTED.value})
# A detector drew a box here and a human rejected it. The most valuable
# background frames there are, so they are ALWAYS kept in full and never
# subject to the empty-background ratio.
HARD_NEGATIVE_REGION_STATUSES: frozenset[str] = frozenset({RegionStatus.FALSE_POSITIVE.value})
# Genuinely region-free frames. Plentiful and cheap, so sampled rather
# than exported wholesale. VERIFY_REJECTED is deliberately excluded: a
# model-rejected region is an unverified non-detection, and mixing those
# into the negatives teaches the next detector the current one's mistakes.
EMPTY_REGION_STATUSES: frozenset[str] = frozenset({RegionStatus.NO_REGION_VISIBLE.value})


@dataclass
class _FrameRow:
    """One output image's worth of state: its boxes, stratum and split key.

    Structurally satisfies both
    :class:`~src.services.curation.export_support.SplittableRow` (via
    ``item_id`` / ``class_id`` / ``has_test_crop``) and
    ``frame_dedup._DedupRow`` (via ``image_id`` / ``has_test_crop``), so
    the same objects flow through the shared splitter and the shared
    near-duplicate collapse without an adapter.
    """

    # Globally unique output stem. Source basenames collide heavily across
    # capture folders, so this is never derived from the file name.
    item_id: str
    image_id: str
    image_path: str
    # Dense class id + YOLO cx, cy, w, h, all relative to the OUTPUT image.
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    stratum: str = ''
    # Dense ordinal of `stratum`, the stratification key stratified_split reads.
    class_id: int = 0
    has_test_crop: bool = False
    # item_crop mode: the parent item region to crop out before resizing.
    crop_norm: tuple[float, float, float, float] | None = None
    # A detector fired here and a human rejected it. Kept in full.
    is_hard_negative: bool = False

    @property
    def is_positive(self) -> bool:
        return bool(self.boxes)


class RowCollector:
    """Turns items-index documents into per-output-image export rows."""

    def __init__(
        self,
        opensearch: Any,
        *,
        profile: SingleClassExportProfile,
        config: CurationConfig,
        region_fields: RegionFields,
    ) -> None:
        self.opensearch = opensearch
        self.profile = profile
        self.config = config
        self.fields = region_fields

    async def collect(self, *, image_mode: ImageMode, empty_bg_ratio: float) -> list[_FrameRow]:
        if self.profile.box_source == 'region':
            return await self._collect_region_rows(
                image_mode=image_mode, empty_bg_ratio=empty_bg_ratio
            )
        return await self._collect_item_rows()

    # ------------------------------------------------------------- item mode

    async def _collect_item_rows(self) -> list[_FrameRow]:
        """Group validated items into per-frame rows for a class subset.

        Frames carrying at least one target-class item are positives; every
        other validated frame becomes a background. That background pool is
        exactly what a narrowed detector needs: real scenes full of the
        classes it must learn to ignore.
        """
        dense_by_registry_id = {cid: i for i, cid in enumerate(self.profile.class_ids)}
        hits = await scroll_hits(
            self.opensearch,
            index=self.config.items_index,
            query={
                'bool': {
                    'must': [{'term': {'class_validated': True}}],
                    'must_not': [{'exists': {'field': 'review_dismissed_at'}}],
                }
            },
            source=[
                'crop_id',
                'image_id',
                'image_path',
                'bbox_norm',
                'class_id',
                'test_holdout',
                'cluster_id',
            ],
        )

        per_frame: dict[str, _FrameRow] = {}
        cluster_by_frame: dict[str, list[str]] = defaultdict(list)
        for hit in hits:
            src = hit.get('_source') or {}
            image_path = str(src.get('image_path') or '')
            if not image_path:
                continue
            frame_key = str(src.get('image_id') or image_path)
            row = per_frame.setdefault(
                frame_key,
                _FrameRow(item_id=frame_key, image_id=frame_key, image_path=image_path),
            )
            row.has_test_crop = row.has_test_crop or bool(src.get('test_holdout'))
            if src.get('cluster_id') is not None:
                cluster_by_frame[frame_key].append(str(src['cluster_id']))

            class_id = src.get('class_id')
            dense = dense_by_registry_id.get(int(class_id)) if class_id is not None else None
            if dense is None:
                continue
            box = xyxy_to_yolo(src.get('bbox_norm'))
            if box is not None:
                row.boxes.append((dense, *box))

        for key, row in per_frame.items():
            row.stratum = self._fallback_stratum(row, cluster_by_frame.get(key, []))
        return list(per_frame.values())

    # ----------------------------------------------------------- region mode

    async def _collect_region_rows(
        self, *, image_mode: ImageMode, empty_bg_ratio: float
    ) -> list[_FrameRow]:
        """Collect region-of-interest rows: positives, hard negatives, empties.

        Positives and hard negatives are scrolled in full. The region-free
        pool is usually orders of magnitude larger than the sample this
        export wants, so it is bounded-scrolled instead — reading a few times
        the needed frames and stopping, rather than paying for a full scroll
        and discarding almost all of it.
        """
        f = self.fields
        source = [
            'crop_id',
            'image_id',
            'image_path',
            'bbox_norm',
            'class_id',
            'test_holdout',
            'cluster_id',
            f.bbox_norm,
            f.status,
            f.cluster_id,
            f.cluster_subid,
        ]
        wanted = sorted(POSITIVE_REGION_STATUSES | HARD_NEGATIVE_REGION_STATUSES)
        hits = await scroll_hits(
            self.opensearch,
            index=self.config.items_index,
            query=self._region_query({'terms': {f'{f.status}.keyword': wanted}}),
            source=source,
        )
        rows = self._build_region_rows(hits, image_mode=image_mode)

        n_pos = sum(1 for r in rows if r.is_positive)
        n_empty = round(n_pos * max(0.0, empty_bg_ratio))
        if n_empty > 0:
            seen = {r.image_id for r in rows}
            empty_hits = await scroll_hits(
                self.opensearch,
                index=self.config.items_index,
                query=self._region_query(
                    {'terms': {f'{f.status}.keyword': sorted(EMPTY_REGION_STATUSES)}}
                ),
                source=source,
                # Over-read: empty frames carry several items each, and this
                # export needs distinct FRAMES, not documents.
                cap=max(n_empty * 6, 3000),
            )
            rows += [
                r
                for r in self._build_region_rows(empty_hits, image_mode=image_mode)
                if r.image_id not in seen
            ]
        return rows

    def _region_query(self, status_clause: dict[str, Any]) -> dict[str, Any]:
        """Region-status clause, narrowed to the profile's parent classes."""
        must: list[dict[str, Any]] = [status_clause]
        if self.profile.class_ids:
            must.append({'terms': {'class_id': list(self.profile.class_ids)}})
        return {'bool': {'must': must}}

    def _build_region_rows(
        self, hits: list[dict[str, Any]], *, image_mode: ImageMode
    ) -> list[_FrameRow]:
        if image_mode == 'item_crop':
            return self._build_region_crop_rows(hits)
        return self._build_region_frame_rows(hits)

    def _build_region_frame_rows(self, hits: list[dict[str, Any]]) -> list[_FrameRow]:
        """whole_frame: one row per source frame, boxes in frame coordinates."""
        f = self.fields
        per_frame: dict[str, _FrameRow] = {}
        cluster_keys: dict[str, list[str]] = defaultdict(list)
        for hit in hits:
            src = hit.get('_source') or {}
            image_path = str(src.get('image_path') or '')
            if not image_path:
                continue
            frame_key = str(src.get('image_id') or image_path)
            row = per_frame.setdefault(
                frame_key,
                _FrameRow(item_id=frame_key, image_id=frame_key, image_path=image_path),
            )
            row.has_test_crop = row.has_test_crop or bool(src.get('test_holdout'))
            status = str(src.get(f.status) or '')
            if status in POSITIVE_REGION_STATUSES:
                box = xyxy_to_yolo(src.get(f.bbox_norm))
                if box is not None:
                    row.boxes.append((0, *box))
                    cluster_keys[frame_key].append(f'pos:{self._region_cluster_key(src)}')
            elif status in HARD_NEGATIVE_REGION_STATUSES:
                row.is_hard_negative = True
                cluster_keys[frame_key].append(f'neg:{self._region_cluster_key(src)}')
            else:
                cluster_keys[frame_key].append(f'bg:{src.get("cluster_id")}')

        for key, row in per_frame.items():
            row.stratum = dominant(cluster_keys.get(key, [])) or self._fallback_stratum(row, [])
        return list(per_frame.values())

    def _build_region_crop_rows(self, hits: list[dict[str, Any]]) -> list[_FrameRow]:
        """item_crop: one row per item, the region re-projected into crop coords.

        Rows still carry their source ``image_id``, so the shared splitter's
        group key keeps every crop of a frame in the same split — no frame's
        background can straddle train and test, and the two image modes split
        the same underlying frame partition, which is what makes a
        whole-frame-vs-crop A/B honest.
        """
        f = self.fields
        rows: list[_FrameRow] = []
        for hit in hits:
            src = hit.get('_source') or {}
            image_path = str(src.get('image_path') or '')
            item_id = str(src.get('crop_id') or hit.get('_id') or '')
            parent = as_xyxy(src.get('bbox_norm'))
            if not image_path or not item_id or parent is None:
                continue
            row = _FrameRow(
                item_id=item_id,
                image_id=str(src.get('image_id') or image_path),
                image_path=image_path,
                has_test_crop=bool(src.get('test_holdout')),
                crop_norm=parent,
            )
            status = str(src.get(f.status) or '')
            if status in POSITIVE_REGION_STATUSES:
                box = reproject_into_crop(src.get(f.bbox_norm), parent)
                if box is None:
                    continue  # region didn't land inside its own parent crop
                row.boxes.append((0, *box))
                row.stratum = f'pos:{self._region_cluster_key(src)}'
            elif status in HARD_NEGATIVE_REGION_STATUSES:
                row.is_hard_negative = True
                row.stratum = f'neg:{self._region_cluster_key(src)}'
            else:
                row.stratum = f'bg:{src.get("cluster_id")}'
            rows.append(row)
        return rows

    def _region_cluster_key(self, src: dict[str, Any]) -> str:
        """Finest-grained region cluster id available, for stratification."""
        subid = src.get(self.fields.cluster_subid)
        if subid:
            return str(subid)
        cid = src.get(self.fields.cluster_id)
        return str(cid) if cid is not None else 'none'

    @staticmethod
    def _fallback_stratum(row: _FrameRow, cluster_ids: list[str]) -> str:
        """Dominant class for positives, dominant cluster for the rest.

        Stratifying backgrounds by cluster keeps each *kind* of background
        scene represented in every split, rather than letting one shoot's
        worth of scenery land entirely in train.
        """
        if row.is_positive:
            return f'pos:{dominant([str(b[0]) for b in row.boxes])}'
        prefix = 'neg' if row.is_hard_negative else 'bg'
        return f'{prefix}:{dominant(cluster_ids) or "none"}'


# =============================================================================
# Geometry helpers
# =============================================================================


def as_xyxy(bbox: Any) -> tuple[float, float, float, float] | None:
    """Validate a normalized ``[x1,y1,x2,y2]``; ``None`` if absent or degenerate."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def xyxy_to_yolo(bbox: Any) -> tuple[float, float, float, float] | None:
    """Normalized ``[x1,y1,x2,y2]`` -> YOLO ``cx, cy, w, h``, clamped to 0..1."""
    box = as_xyxy(bbox)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (
        min(max((x1 + x2) / 2.0, 0.0), 1.0),
        min(max((y1 + y2) / 2.0, 0.0), 1.0),
        min(max(x2 - x1, 0.0), 1.0),
        min(max(y2 - y1, 0.0), 1.0),
    )


def reproject_into_crop(
    bbox: Any, crop: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """Map a source-frame box into a parent crop's own coordinate space.

    Returns ``None`` when the box lands entirely outside the crop, so the
    caller can drop that item rather than write a zero-area label.
    """
    box = as_xyxy(bbox)
    if box is None:
        return None
    px1, py1, px2, py2 = box
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    rx1 = min(max((px1 - cx1) / cw, 0.0), 1.0)
    ry1 = min(max((py1 - cy1) / ch, 0.0), 1.0)
    rx2 = min(max((px2 - cx1) / cw, 0.0), 1.0)
    ry2 = min(max((py2 - cy1) / ch, 0.0), 1.0)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0, rx2 - rx1, ry2 - ry1)


def dominant(keys: list[str]) -> str:
    """Most common key, ties broken by sort order so it stays deterministic."""
    if not keys:
        return ''
    counts = Counter(keys)
    best = max(counts.values())
    return min(k for k, c in counts.items() if c == best)


__all__ = [
    'EMPTY_REGION_STATUSES',
    'HARD_NEGATIVE_REGION_STATUSES',
    'POSITIVE_REGION_STATUSES',
    'RowCollector',
    'as_xyxy',
    'dominant',
    'reproject_into_crop',
    'xyxy_to_yolo',
]
