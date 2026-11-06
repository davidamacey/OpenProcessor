"""The ``upgrade_state`` gate (review fix R-9a) and the ``upgrade_runs`` bookkeeping.

``upgrade_state`` is a ``deployment_settings`` row written before anything else happens. While
it says ``in_progress`` every 0.7.0 process refuses to start (``src.db.schema_check``), so
nothing can write to PostgreSQL while the operator fixes a failed verify. The step that
records ``delete_sources`` sets it to ``completed`` in the same transaction.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.db.models import DeploymentSetting, UpgradeRun


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

FROM_VERSION = '0.6.0'
TO_VERSION = '0.7.0'
STATE_KEY = 'upgrade_state'
IN_PROGRESS = 'in_progress'
COMPLETED = 'completed'
UPDATED_BY = 'upgrade_from_060'


class UpgradeStateError(RuntimeError):
    """The requested transition is not allowed from the recorded state."""


async def read_state(session: AsyncSession) -> str | None:
    """``in_progress``, ``completed`` or ``None`` when no upgrade ever started."""
    value = (
        await session.execute(
            select(DeploymentSetting.value).where(DeploymentSetting.key == STATE_KEY)
        )
    ).scalar_one_or_none()
    return None if value is None else str(value['state'])


async def _write(session: AsyncSession, state: str) -> None:
    value: dict[str, Any] = {'from': FROM_VERSION, 'state': state}
    stmt = insert(DeploymentSetting).values(key=STATE_KEY, value=value, updated_by=UPDATED_BY)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[DeploymentSetting.key],
            set_={
                'value': value,
                'revision': DeploymentSetting.revision + 1,
                'updated_at': dt.datetime.now(dt.UTC),
                'updated_by': UPDATED_BY,
            },
        )
    )


async def mark_in_progress(session: AsyncSession) -> None:
    """Step 1 of the upgrade. Idempotent while in progress; refused once completed."""
    if await read_state(session) == COMPLETED:
        msg = 'the 0.6.0 upgrade already completed; there is nothing to start'
        raise UpgradeStateError(msg)
    await _write(session, IN_PROGRESS)


async def mark_completed(session: AsyncSession) -> None:
    if await read_state(session) != IN_PROGRESS:
        msg = 'the 0.6.0 upgrade is not in progress; refusing to mark it completed'
        raise UpgradeStateError(msg)
    await _write(session, COMPLETED)


async def step_done(session: AsyncSession, step: str) -> bool:
    return (
        await session.execute(
            select(UpgradeRun.id).where(
                UpgradeRun.from_version == FROM_VERSION,
                UpgradeRun.step == step,
                UpgradeRun.status == 'completed',
            )
        )
    ).first() is not None


async def record_step(
    session: AsyncSession,
    step: str,
    *,
    status: str,
    fingerprint: str,
    report: dict[str, Any] | None = None,
) -> None:
    """Record one step of the run; a completed step is recorded once (partial unique index)."""
    now = dt.datetime.now(dt.UTC)
    session.add(
        UpgradeRun(
            from_version=FROM_VERSION,
            to_version=TO_VERSION,
            step=step,
            status=status,
            source_fingerprint=fingerprint,
            report=report,
            started_at=now,
            finished_at=None if status == 'running' else now,
        )
    )
    await session.flush()
