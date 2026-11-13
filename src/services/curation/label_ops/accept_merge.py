"""The merge of the ``accept_machine`` label operation: a person bulk-accepts machine labels.

Each eligible item's current machine class becomes a validated human label with the label source
``human_bulk_accept``. Kept apart from :mod:`~src.services.curation.accept_machine` (which counts
and queues the operation) so the job body can build the merge without importing the code that
starts it.

Which machine labels are accepted is decided here once: :func:`accept_clause` is the same rule as a
query (what the dry run counts and the start route selects) and :func:`is_accepted_source` the same
rule on one document (the write-time re-check).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.audit_math import AUDIT_SAMPLE
from src.domain.ingest_class_sources import (
    DETECTOR_CLASS_SOURCE,
    HUMAN_BULK_ACCEPT_LABEL_SOURCE,
    classifier_class_sources,
)
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.class_sources import VLM_CATEGORY_SCORE, VLM_SUGGESTION_CLASS_SOURCES
from src.services.curation.label_ops.machine_merge import machine_write_blocked
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

ACCEPT_WRITER = 'human:accept_machine'
"""Writer id of the operation: a person's decision, so the lock rule treats the result as human."""

SOURCE_DETECTOR = 'detector'
SOURCE_VLM = 'vlm'


def detector_sources() -> frozenset[str]:
    """``class_source`` of a label the detector (or an ingest classifier) set."""
    return frozenset({DETECTOR_CLASS_SOURCE, *classifier_class_sources()})


def vlm_categories(min_confidence: float) -> list[str]:
    """The VLM's confidence categories whose score reaches ``min_confidence``."""
    return sorted(c for c, score in VLM_CATEGORY_SCORE.items() if score >= min_confidence)


def accept_clause(min_confidence: float, sources: Collection[str]) -> dict[str, Any]:
    """Items carrying a class that one of ``sources`` set with at least ``min_confidence``.

    The detector's confidence is its score (``confidence``); the VLM's is its category
    (``vlm_confidence``) mapped through the shared score table."""
    should: list[dict[str, Any]] = []
    if SOURCE_DETECTOR in sources:
        should.append(
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': sorted(detector_sources())}},
                        {'range': {'confidence': {'gte': min_confidence}}},
                    ]
                }
            }
        )
    categories = vlm_categories(min_confidence)
    if SOURCE_VLM in sources and categories:
        should.append(
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': sorted(VLM_SUGGESTION_CLASS_SOURCES)}},
                        {'terms': {'vlm_confidence': categories}},
                    ]
                }
            }
        )
    return {
        'bool': {
            'filter': [{'exists': {'field': 'class_id'}}],
            'should': should,
            'minimum_should_match': 1,
        }
    }


def is_accepted_source(
    current: dict[str, Any], *, min_confidence: float, sources: Collection[str]
) -> bool:
    """:func:`accept_clause` on one item."""
    class_source = current.get('class_source')
    if SOURCE_DETECTOR in sources and class_source in detector_sources():
        score = current.get('confidence')
        return (
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and score >= min_confidence
        )
    if SOURCE_VLM in sources and class_source in VLM_SUGGESTION_CLASS_SOURCES:
        score = VLM_CATEGORY_SCORE.get(str(current.get('vlm_confidence') or ''))
        return score is not None and score >= min_confidence
    return False


def accept_merger(
    *,
    classes: Mapping[str, tuple[int, str]],
    min_confidence: float,
    sources: Collection[str],
    op_id: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``classes`` maps a class name (registry-name form) to the ``(class_id, class_name)`` of an
    active registry class. A protected item, a class not in ``classes`` (deprecated, the region
    class or gone) and a label below the bar is left as it is: an empty merge writes nothing."""

    def merge(current: dict[str, Any]) -> dict[str, Any]:
        # Re-checked here on the freshest source: a person may have labelled, excluded or held the
        # item out, or an audit may have drawn it, since the ids were snapshotted.
        if (
            machine_write_blocked(current)
            or current.get('review_dismissed_at')
            or current.get(AUDIT_SAMPLE) is True
            or current.get('class_id') is None
            or not is_accepted_source(current, min_confidence=min_confidence, sources=sources)
        ):
            return {}
        found = classes.get(normalize_class_name(str(current.get('class_name') or '')))
        if found is None:
            return {}
        return class_label_update(
            current,
            ItemLabel.human(
                class_id=found[0],
                class_name=found[1],
                label_source=HUMAN_BULK_ACCEPT_LABEL_SOURCE,
                writer=ACCEPT_WRITER,
                op_id=op_id,
            ),
        )

    return merge


__all__ = [
    'ACCEPT_WRITER',
    'SOURCE_DETECTOR',
    'SOURCE_VLM',
    'accept_clause',
    'accept_merger',
    'detector_sources',
    'is_accepted_source',
    'vlm_categories',
]
