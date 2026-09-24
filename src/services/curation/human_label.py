"""Shared merge body for every human class-label write.

``label_crop``, ``batch_label_crops`` (``src/routers/curation/crops.py``),
``move_crops`` and ``POST /review/new_class_proposals/resolve``
(``src/routers/curation/review_resolve.py``) all set a validated class on
an item the same way — this module is the one place that does it, so a
fourth copy of the merge body never creeps in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.services.curation.ingest_class_sources import HUMAN_CLASS_SOURCE
from src.services.detection.cascade_detect import class_provenance


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def human_class_provenance() -> dict[str, Any]:
    """Class provenance for every human class write."""
    from src.services.detection.profile_registry import region_profile_or_neutral

    human = region_profile_or_neutral()
    return class_provenance(
        detector=human.human_detector_name,
        detector_version=human.human_detector_version,
        labeler='human',
    )


def human_label_update(
    current: dict[str, Any],
    *,
    class_id: int,
    class_name: str,
    label_source: str,
    writer: str,
) -> dict[str, Any]:
    """Shared merge body for every human class-label write: ``label_crop``,
    ``batch_label_crops``, and ``POST /review/new_class_proposals/resolve``.

    Snapshots the pre-write class state (``record_class_snapshot``,
    ``restorable=True``) so ``POST /crops/label/undo_batch`` /
    ``POST /crops/{id}/label/undo`` can restore it exactly, sets the class
    validated in the registry's default ensemble cluster (``cluster_id ==
    class_id``), and stamps human class provenance. Callers add anything
    write-specific (e.g. clearing ``needs_new_class``) on top of the
    returned dict.
    """
    from src.services.curation.history import record_class_snapshot

    history = record_class_snapshot(current, writer=writer, restorable=True)
    return {
        'class_id': class_id,
        'class_name': class_name,
        'class_source': HUMAN_CLASS_SOURCE,
        'class_validated': True,
        'label_source': label_source,
        'class_id_history': history,
        # cluster_id mirrors class_id in the default ensemble — see
        # label_crop's docstring for why.
        'cluster_id': class_id,
        # cluster_subid is only meaningful within its origin cluster.
        'cluster_subid': None,
        **human_class_provenance(),
        'updated_at': _now_iso(),
    }


__all__ = ['human_class_provenance', 'human_label_update']
