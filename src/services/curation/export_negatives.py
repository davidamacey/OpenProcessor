"""Imported reviewed-negative frames, for both exporters (W10.8).

A dataset import records a frame whose label file existed and was empty as
``import_label_state: negative`` on its images doc, with ``negative_for``
naming the classes the dataset's author looked for and did not find. A
negative says "none of THESE", nothing more, so each exporter decides for
itself whether the frame is a valid background for the classes it exports:

* the multi-class exporter needs ``negative_for`` to cover EVERY class that
  has objects in the export (a frame negative only for ``car`` may hold
  unlabeled trucks and must not become a false background);
* the single-class exporter needs the exported class (or any of the
  subset's classes) in ``negative_for``.

The scroll is one function so the two can never disagree about what a
negative frame is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.export_images import image_file_stem
from src.services.curation.export_split import SPLITS
from src.services.curation.export_support import _resolve_source_path, hash_split, scroll_hits


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.config import CurationConfig
    from src.services.curation.export_split import SplitMode

NEGATIVE_STATE = 'negative'


@dataclass(frozen=True)
class NegativeFrame:
    image_id: str
    image_path: str
    negative_for: frozenset[str]
    dataset_split: str | None = None
    import_source_stem: str | None = None
    import_stratum: str | None = None
    import_hard_negative: bool = False


async def scroll_negative_frames(opensearch: Any, *, images_index: str) -> list[NegativeFrame]:
    """Every images-index doc an import marked as a reviewed negative, in
    ``image_id`` order (deterministic whatever the scroll order was)."""
    hits = await scroll_hits(
        opensearch,
        index=images_index,
        query={'term': {'import_label_state': NEGATIVE_STATE}},
        source=[
            'import_label_state',
            'image_id',
            'image_path',
            'negative_for',
            'dataset_split',
            'import_source_stem',
            'import_stratum',
            'import_hard_negative',
        ],
    )
    frames: dict[str, NegativeFrame] = {}
    for hit in hits:
        src = hit.get('_source') or {}
        if src.get('import_label_state') != NEGATIVE_STATE:
            continue
        image_id = str(src.get('image_id') or hit.get('_id') or '')
        image_path = str(src.get('image_path') or '')
        if not image_id or not image_path:
            continue
        negative_for = src.get('negative_for') or []
        if not isinstance(negative_for, list):
            negative_for = [negative_for]
        frames[image_id] = NegativeFrame(
            image_id=image_id,
            image_path=image_path,
            negative_for=frozenset(str(n) for n in negative_for),
            dataset_split=src.get('dataset_split') or None,
            import_source_stem=src.get('import_source_stem') or None,
            import_stratum=src.get('import_stratum') or None,
            import_hard_negative=bool(src.get('import_hard_negative')),
        )
    return [frames[k] for k in sorted(frames)]


def plan_multi_class_negatives(
    frames: Iterable[NegativeFrame],
    *,
    exported_class_names: set[str],
    already_exported: set[str],
    split_mode: SplitMode,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[tuple[NegativeFrame, str]], int]:
    """``([(frame, split)], skipped_partial)`` for the multi-class exporter.

    A frame is a background only when its ``negative_for`` covers every class
    the export has objects for; one that does not is counted in
    ``skipped_partial`` (it may hold unlabeled objects of an uncovered
    class). A frame already exported with objects is left alone. The split is
    the stored one under ``keep_imported``, else (or when none is stored) the
    same hash bucket a frameless item would get.
    """
    planned: list[tuple[NegativeFrame, str]] = []
    skipped = 0
    for frame in frames:
        if frame.image_id in already_exported:
            continue
        if not exported_class_names <= frame.negative_for:
            skipped += 1
            continue
        if split_mode == 'keep_imported' and frame.dataset_split in SPLITS:
            split = str(frame.dataset_split)
        else:
            split = hash_split(frame.image_id, seed, train_ratio, val_ratio)
        planned.append((frame, split))
    return planned, skipped


def write_negative_frames(
    planned: Iterable[tuple[NegativeFrame, str]],
    *,
    labels_root: Path,
    images_root: Path,
    config: CurationConfig,
    split_counts: Any,
    test_stems: list[tuple[str | None, str]],
    image_jobs: list[tuple[str, str]],
    copy_images: bool,
) -> int:
    """Write one empty label file per planned frame (and queue its image
    copy); returns how many were written. ``split_counts`` is the exporter's
    per-split image counter, ``test_stems`` its ``(import_source_stem, own
    stem)`` list for ``source_frozen_test_sha``."""
    written = 0
    for frame, split in planned:
        stem = image_file_stem(frame.image_id)
        (labels_root / split / f'{stem}.txt').write_text('')
        setattr(split_counts, split, getattr(split_counts, split) + 1)
        written += 1
        if split == 'test':
            test_stems.append((frame.import_source_stem, stem))
        if copy_images:
            src_path = _resolve_source_path(frame.image_path, config)
            if src_path is not None:
                dest = images_root / split / f'{stem}{src_path.suffix or ".jpg"}'
                image_jobs.append((str(src_path), str(dest)))
    return written


__all__ = [
    'NEGATIVE_STATE',
    'NegativeFrame',
    'plan_multi_class_negatives',
    'scroll_negative_frames',
    'write_negative_frames',
]
