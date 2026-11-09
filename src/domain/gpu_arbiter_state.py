"""The value of ``deployment_settings('vlm_paused_for_training')``: who pauses the VLM workers.

One jsonb document with one section per holder, so the arbiter tick and ``openprocessor vlm use``
change their own section and never each other's:

``arbiter``     the tick's record: the jobs that hold cards (``claims``), the services it stopped
                (``stopped``), the claims whose services are confirmed stopped (``acknowledged``),
                and the last Docker failure (``error``).
``vlm_switch``  ``{'at': <epoch seconds>}``: a model swap is in progress (``openprocessor vlm
                use``).

A train, probe or bake-off handler starts its run once its own job id is in ``acknowledged``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


KEY = 'vlm_paused_for_training'
ARBITER = 'arbiter'


@dataclass(frozen=True)
class ArbiterRecord:
    """The tick's section of the document."""

    claims: tuple[str, ...] = ()
    stopped: tuple[str, ...] = ()
    acknowledged: tuple[str, ...] = ()
    error: str | None = None
    at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            'claims': list(self.claims),
            'stopped': list(self.stopped),
            'acknowledged': list(self.acknowledged),
            'error': self.error,
            'at': self.at,
        }

    @classmethod
    def from_json(cls, raw: Any) -> ArbiterRecord:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            claims=tuple(str(c) for c in raw.get('claims') or ()),
            stopped=tuple(str(s) for s in raw.get('stopped') or ()),
            acknowledged=tuple(str(a) for a in raw.get('acknowledged') or ()),
            error=raw.get('error'),
            at=float(raw.get('at') or 0.0),
        )


def arbiter_record(value: Any) -> ArbiterRecord:
    """The arbiter section of a stored document (empty when there is none)."""
    return ArbiterRecord.from_json(value.get(ARBITER) if isinstance(value, dict) else None)
