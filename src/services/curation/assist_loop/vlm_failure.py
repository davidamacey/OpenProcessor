"""The VLM did not answer: the loop's evidence is missing, not negative."""

from __future__ import annotations


STOP_REASON = 'vlm_failed'
"""``stop_reason`` of a loop that stopped because every call of a step failed."""


class VlmFailedError(RuntimeError):
    """Every VLM call of a step failed (endpoint down, unknown model, rejected request).

    A node that got no answer because of it must not be decided by the gate: the run stops
    with this error, the node stays open, and a resume asks it again."""
