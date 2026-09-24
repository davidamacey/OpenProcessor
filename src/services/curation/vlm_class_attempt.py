"""VLM class answers -> item updates, and recording an attempt with no class.

``class_source='vlm_unmatched'`` means "the VLM answered a label that is not
in the registry" -- it carries that label in ``vlm_raw_class`` for registry
growth. A reply with *no* label (empty, ``null``, ``-1``, an out-of-range
index, or no parseable entry at all) is a different outcome: nothing was
learned about the item's class, so its class fields -- a proposal's
proposal source, a classifier's label and provenance -- stay exactly as they
were. The attempt itself is recorded instead:

* ``vlm_class_attempted_at`` -- when a VLM was last asked for this item's
  class (any outcome).
* ``vlm_class_empty_reason`` -- why that attempt gave no class
  (:class:`EmptyClassReason`); cleared (``null``) when an attempt does
  produce an answer.

Review queues find the empty attempts by the reason field, and the VLM
selectors skip items whose last attempt was empty within
:data:`EMPTY_ANSWER_RETRY_WINDOW` so they are not re-asked in a hot loop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.services.curation.history import record_class_snapshot


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.labeling.vlm_labeler import VlmClassPrediction


VLM_CLASS_ATTEMPTED_AT_FIELD = 'vlm_class_attempted_at'
VLM_CLASS_EMPTY_REASON_FIELD = 'vlm_class_empty_reason'

EMPTY_ANSWER_RETRY_WINDOW = timedelta(hours=24)
"""How long an item whose VLM class attempt came back empty stays out of
the VLM selectors before it is eligible to be asked again."""


class EmptyClassReason(StrEnum):
    """Why a VLM class attempt produced no class."""

    NO_ANSWER = 'no_answer'
    """The reply had no class (empty, ``null``, or the class omitted)."""
    NO_MATCH = 'no_match'
    """The VLM said no catalog class fits (``-1`` / ``__new__`` with no
    proposed name) without naming an alternative."""
    INVALID_INDEX = 'invalid_index'
    """A class index outside the catalog it was shown."""
    UNPARSEABLE = 'unparseable'
    """The reply held no usable entry for this item."""


def class_attempt_fields(now: str, reason: EmptyClassReason | None = None) -> dict[str, Any]:
    """The attempt marker every VLM class write carries.

    ``reason=None`` (the attempt produced an answer) writes an explicit
    ``null`` so a later successful attempt clears an earlier empty one.
    """
    return {
        VLM_CLASS_ATTEMPTED_AT_FIELD: now,
        VLM_CLASS_EMPTY_REASON_FIELD: reason.value if reason is not None else None,
    }


def empty_answer_reason_for_index(class_id: int | None, n_classes: int) -> EmptyClassReason:
    """Reason for a combined reply whose ``class_id`` does not pick a class."""
    if class_id is None:
        return EmptyClassReason.NO_ANSWER
    if class_id < 0:
        return EmptyClassReason.NO_MATCH
    if class_id >= n_classes:
        return EmptyClassReason.INVALID_INDEX
    msg = f'class_id {class_id} picks a class; it is not an empty answer'
    raise ValueError(msg)


def recent_empty_answer_clause(now: datetime | None = None) -> dict[str, Any]:
    """Query clause matching items whose last VLM class attempt was empty
    and happened within :data:`EMPTY_ANSWER_RETRY_WINDOW` -- for a
    selector's ``must_not``."""
    cutoff = ((now or datetime.now(UTC)) - EMPTY_ANSWER_RETRY_WINDOW).isoformat()
    return {
        'bool': {
            'filter': [
                {'exists': {'field': VLM_CLASS_EMPTY_REASON_FIELD}},
                {'range': {VLM_CLASS_ATTEMPTED_AT_FIELD: {'gte': cutoff}}},
            ]
        }
    }


NEW_CLASS_SENTINEL = '__new__'


def prediction_empty_reason(pred: VlmClassPrediction) -> EmptyClassReason | None:
    """Why a batch class prediction carries no class answer; ``None`` when
    it does. A prediction whose call never completed
    (``failure='request_failed'``) is not an answer at all -- callers skip
    it before asking."""
    if pred.failure == 'unparseable':
        return EmptyClassReason.UNPARSEABLE
    name = pred.class_name.strip()
    if not name:
        return EmptyClassReason.NO_ANSWER
    if name == NEW_CLASS_SENTINEL and not pred.proposed_class:
        return EmptyClassReason.NO_MATCH
    return None


def prediction_class_update(
    pred: VlmClassPrediction,
    *,
    name_to_id: dict[str, int],
    resolve: Callable[..., str | None],
    now: str,
    provenance: dict[str, Any],
    extras: dict[str, Any] | None = None,
    set_cluster: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The item update for one open-vocabulary class prediction.

    Returns ``(update, new_class_proposal)``; ``update`` is ``None`` when
    nothing may be written (the call never completed). ``resolve(raw,
    confidence=...)`` maps an answer onto a registry name. ``extras``
    (make / model / region-visible hints) ride on every written update;
    ``set_cluster`` also moves a resolved item into its class cluster. An
    empty answer records no raw label: a reply excerpt such as ``"]"`` is
    not a label and would pollute the raw-label aggregation.
    """
    if pred.failure == 'request_failed':
        return None, None
    extra = dict(extras or {})
    empty = prediction_empty_reason(pred)
    if empty is not None:
        return {**extra, **class_attempt_fields(now, empty), 'updated_at': now}, None
    attempt = class_attempt_fields(now)
    is_new = pred.class_name == NEW_CLASS_SENTINEL
    raw_label = pred.proposed_class if is_new else pred.class_name
    base = {
        'label_source': 'vlm',
        'vlm_confidence': pred.confidence,
        'vlm_raw_label': raw_label,
        **extra,
        **attempt,
        'updated_at': now,
    }
    answer = pred.proposed_class if is_new else pred.class_name
    resolved = resolve(answer, confidence=pred.confidence)
    if resolved is None and is_new:
        return (
            {
                **base,
                'class_source': 'vlm_new_class_pending',
                'vlm_proposed_class': pred.proposed_class,
                'needs_new_class': True,
            },
            {'crop_id': pred.img_id, 'proposed_class': pred.proposed_class},
        )
    if resolved is None:
        return {**base, 'class_source': 'vlm_unmatched', 'vlm_raw_class': pred.class_name}, None
    cid = name_to_id[resolved]
    update = {
        **base,
        'class_id': cid,
        'class_name': resolved,
        'class_source': 'vlm',
        **provenance,
    }
    if is_new:
        update['vlm_raw_class'] = pred.proposed_class
    if set_cluster:
        update['cluster_id'] = cid
    return update, None


def with_class_snapshot(
    update: dict[str, Any], current: dict[str, Any], *, writer: str
) -> dict[str, Any]:
    """``update`` plus a restorable pre-write class snapshot when it
    changes the class (any ``class_source`` write); unchanged otherwise."""
    if 'class_source' not in update:
        return update
    return {
        **update,
        'class_id_history': record_class_snapshot(current, writer=writer, restorable=True),
    }


VLM_CLASS_ATTEMPT_MAPPING: dict[str, Any] = {
    VLM_CLASS_ATTEMPTED_AT_FIELD: {'type': 'date'},
    VLM_CLASS_EMPTY_REASON_FIELD: {'type': 'keyword'},
}


__all__ = [
    'EMPTY_ANSWER_RETRY_WINDOW',
    'NEW_CLASS_SENTINEL',
    'VLM_CLASS_ATTEMPTED_AT_FIELD',
    'VLM_CLASS_ATTEMPT_MAPPING',
    'VLM_CLASS_EMPTY_REASON_FIELD',
    'EmptyClassReason',
    'class_attempt_fields',
    'empty_answer_reason_for_index',
    'prediction_class_update',
    'prediction_empty_reason',
    'recent_empty_answer_clause',
    'with_class_snapshot',
]
