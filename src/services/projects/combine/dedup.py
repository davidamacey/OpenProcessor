"""Cross-source dedup rules of a combine (projects plan section 6): which
images are the same file, and what happens to the boxes of a duplicate.

Pure and shared: the preview counts what :func:`decide_merges` would do and
the executor does it, so the two cannot drift.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.clients.occ_locks import _is_human_marker
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.detection.geometry import iou


if TYPE_CHECKING:
    from collections.abc import Sequence

_HASH_BLOCK = 1024 * 1024
UNCLASSED_RANK = 9


@dataclass(frozen=True)
class Fingerprint:
    """One included source image: where it came from and what it is."""

    source_index: int
    image_id: str
    imohash: str
    path: str


def trust_rank(item: dict[str, object]) -> int:
    """Lower is more trusted: human (0) > import (1) > vlm (2) > model (3).
    An item with no class has no label to trust."""
    if item.get('class_id') is None and not item.get('class_name'):
        return UNCLASSED_RANK
    markers = (item.get('label_source'), item.get('class_source'))
    if any(_is_human_marker(m) for m in markers):
        return 0
    if any(m in ('import', LABEL_IMPORT_CLASS_SOURCE) for m in markers):
        return 1
    if any('vlm' in str(m).lower() for m in markers if m):
        return 2
    return 3


def file_sha256(path: str) -> str | None:
    """Full-bytes sha256, or ``None`` when the file cannot be read."""
    digest = hashlib.sha256()
    try:
        with Path(path).open('rb') as fh:
            while block := fh.read(_HASH_BLOCK):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def find_duplicates(
    fingerprints: Sequence[Fingerprint],
) -> dict[tuple[int, str], tuple[int, str]]:
    """``{(source_index, image_id): (priority source_index, image_id)}`` for
    every image that is a byte-identical copy of one from an earlier source.

    Candidates share an ``imohash`` (which samples the file); each is
    confirmed by a full sha256, computed only for candidates. An unreadable
    file is never called a duplicate.
    """
    by_hash: dict[str, list[Fingerprint]] = {}
    for fp in fingerprints:
        if fp.imohash:
            by_hash.setdefault(fp.imohash, []).append(fp)
    out: dict[tuple[int, str], tuple[int, str]] = {}
    for group in by_hash.values():
        if len({m.source_index for m in group}) < 2:
            continue
        members = sorted(group, key=lambda m: (m.source_index, m.image_id))
        sha = {m.image_id: file_sha256(m.path) for m in members}
        for i, member in enumerate(members):
            digest = sha[member.image_id]
            if digest is None:
                continue
            priority = next(
                (
                    p
                    for p in members[:i]
                    if p.source_index != member.source_index and sha[p.image_id] == digest
                ),
                None,
            )
            if priority is not None:
                out[(member.source_index, member.image_id)] = (
                    priority.source_index,
                    priority.image_id,
                )
    return out


@dataclass(frozen=True)
class Probe:
    """One box as the merge rules see it."""

    bbox: tuple[float, float, float, float]
    target_class: str | None
    rank: int


MergeKind = Literal['union', 'merge', 'conflict']


@dataclass(frozen=True)
class MergeDecision:
    incoming: int
    kind: MergeKind
    existing: int | None = None
    take_incoming_label: bool = False
    """``merge`` only: the incoming label is the more trusted one."""


def decide_merges(
    existing: Sequence[Probe], incoming: Sequence[Probe], *, iou_min: float
) -> list[MergeDecision]:
    """One decision per incoming box, in order. A box matches the unclaimed
    existing box with the highest IoU >= ``iou_min`` (ties: lowest index).

    - no match -> ``union`` (the box is added);
    - same target class (or either has none) -> ``merge``, keeping the more
      trusted label (ties keep the existing, i.e. the higher-priority source);
    - a different class -> ``conflict``: the existing (priority) label stays
      and the pair is flagged for review.
    """
    claimed: set[int] = set()
    out: list[MergeDecision] = []
    for i, box in enumerate(incoming):
        best, best_iou = None, 0.0
        for j, other in enumerate(existing):
            if j in claimed:
                continue
            score = iou(other.bbox, box.bbox)
            if score >= iou_min and score > best_iou:
                best, best_iou = j, score
        if best is None:
            out.append(MergeDecision(i, 'union'))
            continue
        claimed.add(best)
        other = existing[best]
        if (
            box.target_class is None
            or other.target_class is None
            or box.target_class == other.target_class
        ):
            out.append(MergeDecision(i, 'merge', best, take_incoming_label=box.rank < other.rank))
        else:
            out.append(MergeDecision(i, 'conflict', best))
    return out


__all__ = [
    'UNCLASSED_RANK',
    'Fingerprint',
    'MergeDecision',
    'Probe',
    'decide_merges',
    'file_sha256',
    'find_duplicates',
    'trust_rank',
]
