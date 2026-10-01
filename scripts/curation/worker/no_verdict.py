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
streaming runner owns one counter per stage (``visible_no_verdict`` /
``combined_no_verdict`` in ``runner.py``).
"""

from __future__ import annotations

import os

from src.core.logging import get_logger


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


__all__ = [
    'DEFAULT_MAX_NO_VERDICT_ATTEMPTS',
    'NoVerdictCounter',
    'max_no_verdict_attempts',
]
