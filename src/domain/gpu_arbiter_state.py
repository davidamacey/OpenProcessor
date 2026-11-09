"""The value of ``deployment_settings('vlm_paused_for_training')``: who pauses the VLM workers.

One jsonb document with one section per holder, so the arbiter tick and ``openprocessor vlm use``
change their own section and never each other's:

``arbiter``     the tick's record: the jobs that hold cards (``claims``), the services it stopped
                (``stopped``), the claims whose services are confirmed stopped (``acknowledged``),
                and the last Docker failure (``error``).
``vlm_switch``  ``{'at': <epoch seconds>}``: a model swap is in progress (``openprocessor vlm
                use``). A crashed command must not pause the workers forever, so a section older
                than :data:`SWITCH_TTL_S` counts for nothing.

The VLM workers idle while any holder holds (:func:`is_paused`).

A train, probe or bake-off handler starts its run once its own job id is in ``acknowledged``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


KEY = 'vlm_paused_for_training'
ARBITER = 'arbiter'
VLM_SWITCH = 'vlm_switch'
SWITCH_TTL_S = 1800.0


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


def switch_active(value: Any, *, now: float | None = None) -> bool:
    """A ``vlm use`` swap section younger than :data:`SWITCH_TTL_S`."""
    section = value.get(VLM_SWITCH) if isinstance(value, dict) else None
    if not isinstance(section, dict):
        return False
    at = float(section.get('at') or 0.0)
    return ((time.time() if now is None else now) - at) < SWITCH_TTL_S


def is_paused(value: Any, *, now: float | None = None) -> bool:
    """Whether the VLM workers must idle: a card-holding claim or a live model swap."""
    return bool(arbiter_record(value).claims) or switch_active(value, now=now)
