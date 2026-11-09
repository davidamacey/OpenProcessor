"""The GPU arbiter's database side: who holds cards, and the shared ``vlm_paused_for_training`` document.

Every function takes the caller's sync ``Session`` inside a deployment transaction and never
commits. The document is one ``deployment_settings`` row whose jsonb value has a section per
writer (:mod:`src.domain.gpu_arbiter_state`); a section is replaced with ``jsonb_set`` in one
statement, so two writers never overwrite each other's section and no read-modify-write races.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.domain import gpu_arbiter_state as state, job_states as js


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

UPDATED_BY = 'gpu_arbiter'

_LIVE = (js.RUNNING, js.CANCELLING)

_CLAIMS = text(
    "SELECT DISTINCT j.id, j.kind, j.params ->> 'gpu' AS probe_gpus, "
    "tr.spec ->> 'cuda_visible_devices' AS train_gpus "
    'FROM gpu_reservations r JOIN jobs j ON j.id = r.holder_job_id '
    'LEFT JOIN training_runs tr ON tr.job_id = j.id '
    'WHERE j.status = ANY(:live) ORDER BY j.id'
)
_RELEASE_DEAD = text(
    'DELETE FROM gpu_reservations r USING jobs j '
    'WHERE j.id = r.holder_job_id AND NOT (j.status = ANY(:live)) RETURNING r.gpu_uuid'
)
_READ = text('SELECT value FROM deployment_settings WHERE key = :key')
_SET_SECTION = text(
    'INSERT INTO deployment_settings (key, value, updated_by) '
    'VALUES (:key, jsonb_build_object(CAST(:section AS text), CAST(:body AS jsonb)), :by) '
    'ON CONFLICT (key) DO UPDATE SET value = jsonb_set('
    "CASE WHEN jsonb_typeof(deployment_settings.value) = 'object' "
    "THEN deployment_settings.value ELSE '{}'::jsonb END, "
    'ARRAY[CAST(:section AS text)], CAST(:body AS jsonb)), '
    'revision = deployment_settings.revision + 1, updated_at = now(), updated_by = :by'
)
_DROP_SECTION = text(
    'UPDATE deployment_settings SET value = value - CAST(:section AS text), '
    'revision = revision + 1, updated_at = now(), updated_by = :by '
    "WHERE key = :key AND jsonb_typeof(value) = 'object' AND (value -> CAST(:section AS text)) IS NOT NULL"
)


@dataclass(frozen=True)
class Claim:
    """One running job that holds cards, with what the arbiter needs to scope its services."""

    job_id: str
    kind: str
    cuda_visible_devices: str | None


def active_claims(session: Session) -> list[Claim]:
    """Running jobs that hold at least one card, oldest id first."""
    rows = session.execute(_CLAIMS, {'live': list(_LIVE)}).mappings()
    return [
        Claim(str(r['id']), r['kind'], r['train_gpus'] or r['probe_gpus'] or None) for r in rows
    ]


def release_dead_holders(session: Session) -> int:
    """Delete reservations whose holder is no longer running (it died or ended without releasing)."""
    return len(session.execute(_RELEASE_DEAD, {'live': list(_LIVE)}).all())


def read_value(session: Session) -> Any:
    """The stored document, ``None`` when nothing wrote it yet."""
    return session.execute(_READ, {'key': state.KEY}).scalar_one_or_none()


def set_section(session: Session, section: str, body: dict[str, Any]) -> None:
    session.execute(
        _SET_SECTION,
        {'key': state.KEY, 'section': section, 'body': json.dumps(body), 'by': UPDATED_BY},
    )


def drop_section(session: Session, section: str) -> None:
    session.execute(_DROP_SECTION, {'key': state.KEY, 'section': section, 'by': UPDATED_BY})
