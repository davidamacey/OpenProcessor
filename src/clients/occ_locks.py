"""The lock rule (W10): a human- or import-set label/box is never touched
by an automated writer.

Split out of ``src/clients/occ.py`` (which owns the OCC read/merge/write
machinery these predicates get consulted from) to stay under the
per-file LOC ratchet — ``occ.py`` imports and re-exports every name here,
so ``from src.clients.occ import is_locked_class`` (every existing call
site) keeps working unchanged.
"""

from __future__ import annotations

from typing import Any

from src.config import get_region_fields


def _is_human_marker(value: Any) -> bool:
    """A guard-field value indicates a human write iff it's a string
    containing the substring ``human``.

    Matches the in-codebase markers ``human``, ``human_move``, and
    ``vlm_human_confirmed`` (used across the crops, regions, and
    clustering write paths).
    Non-human writers use ``ingest``, ``item_model``, ``coco_yolo11``,
    ``vlm``, ``cluster_majority_agreement``, etc.
    """
    return isinstance(value, str) and 'human' in value


def _is_locked_marker(value: Any) -> bool:
    """``_is_human_marker`` plus the two import-provenance string values
    (W10). Used wherever a re-ingest/automated writer must never clobber
    an imported label the same way it must never clobber a human one."""
    from src.services.curation.ingest_class_sources import (
        LABEL_IMPORT_CLASS_SOURCE,
        LABEL_SOURCE_IMPORT,
    )

    return _is_human_marker(value) or value in (LABEL_SOURCE_IMPORT, LABEL_IMPORT_CLASS_SOURCE)


def is_locked_class(source: dict[str, Any]) -> bool:
    """Reusable class-lock guard predicate (W10; replaces the narrower
    ``is_human_owned_class``, which this supersedes — no symbol of that
    name remains anywhere in the codebase).

    True when a crop's current class state must never be touched by an
    automated writer:
      * a human already set/confirmed the class (``_is_human_marker``);
      * OR it's a *validated* imported label (``class_source ==
        LABEL_IMPORT_CLASS_SOURCE`` and ``class_validated``) — an
        unvalidated (``label_trust: suggestion``) import is NOT locked,
        by design: the machine pipeline may still relabel it;
      * OR the item is frozen into the test holdout (``test_holdout``),
        regardless of who set its class.

    Every automated CLASS writer must consult this before overwriting
    class fields. Wired into:
      * the region-detection worker's classification gate and bulk writer;
      * the VLM label-batch router endpoint (caller-supplied ``crop_ids``
        with no upstream query filter, so it's the site with no other
        protection);
      * the ingest pipeline router (defense-in-depth).
    """
    from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE

    class_source = source.get('class_source')
    if _is_human_marker(class_source):
        return True
    if class_source == LABEL_IMPORT_CLASS_SOURCE and bool(source.get('class_validated')):
        return True
    return bool(source.get('test_holdout'))


def is_locked_box(box: Any) -> bool:
    """True when a region box must never be touched by an automated
    writer: a human created/verdicted/transcribed it
    (:func:`src.services.curation.region_boxes.is_human_owned`), or it
    came from a dataset import (``source == CANDIDATE_IMPORT``)."""
    from src.config.region_source import CANDIDATE_IMPORT
    from src.services.curation.region_boxes import is_human_owned as _box_is_human_owned

    return _box_is_human_owned(box) or box.source == CANDIDATE_IMPORT


def is_locked_item(source: dict[str, Any], F: Any = None) -> bool:
    """Item-level lock test reprocess and undo use: locked class, OR any
    locked box in the item's region list, OR a validated set whose
    verifier is ``human``/``import``."""
    if is_locked_class(source):
        return True
    from src.services.curation.region_boxes import read_boxes

    fields = F or get_region_fields()
    if any(is_locked_box(box) for box in read_boxes(source, fields)):
        return True
    return bool(source.get(fields.validated)) and source.get(fields.verifier) in ('human', 'import')


__all__ = [
    '_is_human_marker',
    '_is_locked_marker',
    'is_locked_box',
    'is_locked_class',
    'is_locked_item',
]
