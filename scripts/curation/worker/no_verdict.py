"""Bounded retries for items the VLM answers without a verdict.

A reply with no verdict (the combined call's box verdict is null / the
entry is unparseable, or the visibility call answers nothing) used to
leave the item pending for the next producer poll with no bound. The VLM
runs at temperature 0, so for some inputs that answer is deterministic
and the item looped through the segmenter + VLM forever.

:class:`NoVerdictCounter` counts consecutive no-verdict replies per item
and says when the cap is reached; the runner then resolves the item
(combined: a reviewable ``verify_rejected`` with
``REJECT_REASON_NO_VERDICT``; visibility: fail open to detection).

Transport failures (the upstream call raised / the VLM is down) are NOT
no-verdict replies: the runner never records them here, so an outage
keeps retrying and never turns into a terminal write.

The count is in-process only. A worker restart resets it, which at worst
buys an item ``cap`` more attempts per restart -- still bounded, and it
needs no extra index field, mapping change or write per attempt. The
streaming runner owns one counter per stage; the per-crop cascade
(``cascade._process_crop``) shares the process-wide :func:`cascade_counter`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.verify import candidate_reject_doc
from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_NO_VERDICT
from src.core.logging import get_logger


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask


logger = get_logger('curation_worker')

DEFAULT_MAX_NO_VERDICT_ATTEMPTS = 3


def max_no_verdict_attempts() -> int:
    """``OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS`` (default 3, minimum 1)."""
    raw = (os.environ.get('OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS') or '').strip()
    if not raw:
        return DEFAULT_MAX_NO_VERDICT_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            'region_worker_no_verdict_cap_invalid',
            value=raw,
            default=DEFAULT_MAX_NO_VERDICT_ATTEMPTS,
        )
        return DEFAULT_MAX_NO_VERDICT_ATTEMPTS
    return max(1, value)


class NoVerdictCounter:
    """Per-item count of consecutive no-verdict replies at one stage.

    In-process (see the module docstring). Not thread-safe; the worker's
    stages all run on one event loop and never await between the read
    and the write of a count.
    """

    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self._counts: dict[str, int] = {}

    def record(self, item_id: str) -> bool:
        """Count one no-verdict reply; True when this one reaches the cap.

        On True the count is dropped: the caller resolves the item now.
        """
        n = self._counts.get(item_id, 0) + 1
        if n >= self.cap:
            self._counts.pop(item_id, None)
            return True
        self._counts[item_id] = n
        return False

    def clear(self, item_id: str) -> None:
        """Forget the item: it got a real verdict or was written."""
        self._counts.pop(item_id, None)

    def count(self, item_id: str) -> int:
        return self._counts.get(item_id, 0)

    def __len__(self) -> int:
        return len(self._counts)


_cascade_counter: NoVerdictCounter | None = None


def cascade_counter() -> NoVerdictCounter:
    """The process-wide count for the per-crop cascade (``_process_crop``),
    created on first use with the configured cap."""
    global _cascade_counter  # noqa: PLW0603 - one lazily-built process counter
    if _cascade_counter is None:
        _cascade_counter = NoVerdictCounter(max_no_verdict_attempts())
    return _cascade_counter


def reset_cascade_counter() -> None:
    """Drop the process-wide cascade count (tests; a config reload)."""
    global _cascade_counter  # noqa: PLW0603
    _cascade_counter = None


def no_verdict_reject_doc(
    t: _ItemTask,
    *,
    actor: str,
    detector_version: str,
    class_update: dict[str, Any] | None,
    event: str = 'combined_verify_reject',
) -> dict[str, Any]:
    """The capped combined no-verdict write: a reviewable rejected candidate.

    Same shape as a verifier reject (the box kept in the ``candidate_*``
    fields, so a human confirm promotes it and a requeue by reason retries
    it) but ``bbox_correct`` is written null -- the verifier never gave a
    box verdict -- and the reason is :data:`REJECT_REASON_NO_VERDICT`.
    ``class_update`` is the reply's class side, when a reply exists;
    ``event`` names the verify call in the detector-chain entry.
    """
    F = get_region_fields()
    t.detection_trace.append(f'{actor}:{event}:{REJECT_REASON_NO_VERDICT}')
    doc = candidate_reject_doc(
        candidate_in_source=t.candidate_in_source,
        candidate_score=t.candidate_score,
        detector=actor,
        detector_version=detector_version,
        candidate_source=t.candidate_source,
        reason=REJECT_REASON_NO_VERDICT,
        chain=t.detection_trace,
    )
    # Explicit null: an earlier pass's verdict must not outlive this one.
    doc[F.bbox_correct] = None
    if class_update:
        doc.update(class_update)
    return doc


def cascade_no_verdict(
    t: _ItemTask,
    *,
    actor: str,
    detector_version: str,
    candidate_in_source: tuple[float, float, float, float] | None,
    candidate_score: float,
    candidate_source: str,
    event: str,
    class_update: dict[str, Any] | None = None,
) -> None:
    """One no-verdict pass of the per-crop cascade on a candidate box.

    Below the cap ``t.update_doc`` is left empty (nothing is written, the
    item stays pending); at the cap it becomes the
    :func:`no_verdict_reject_doc` for this candidate.
    """
    t.update_doc = {}
    counter = cascade_counter()
    if not counter.record(t.crop_id):
        return
    logger.warning(
        'region_worker_no_verdict_cap',
        stage='cascade',
        crop_id=t.crop_id,
        attempts=counter.cap,
    )
    t.candidate_in_source = candidate_in_source
    t.candidate_score = candidate_score
    t.candidate_source = candidate_source
    t.update_doc = no_verdict_reject_doc(
        t,
        actor=actor,
        detector_version=detector_version,
        class_update=class_update,
        event=event,
    )


__all__ = [
    'DEFAULT_MAX_NO_VERDICT_ATTEMPTS',
    'NoVerdictCounter',
    'cascade_counter',
    'cascade_no_verdict',
    'max_no_verdict_attempts',
    'no_verdict_reject_doc',
    'reset_cascade_counter',
]
