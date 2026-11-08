"""Budgets of an auto-label loop on the SQL reservations of ``db/repos/assist_loop.py``.

Before every upstream call the loop reserves its units (crops, one request, completion tokens
as the call's ``max_tokens``, prompt tokens as an estimate) in one fenced transaction; a
refusal on any unit leaves none reserved. After the answer the reservation is settled at what
the endpoint reported; a call with no ``usage`` is charged what was reserved. At a slice start
every reservation with no closing row is forfeited (counted spent): it may have reached the
endpoint. The counters this class keeps are a read-through cache of the rows, refreshed after
each write, so the scheduler can ask ``would_refuse`` without a query; the reservation itself
is always the authority.

The audit's second-opinion requests draw on their own pool with its own limits.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import assist_loop as repo, assist_loop_record as record_rows
from src.domain.assist_loop_budget import PROMPT_HISTORY_CALLS, call_units, settled_amounts
from src.services.curation.assist_loop.tree import AUDIT, VLM


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sqlalchemy.orm import Session

    from src.domain.vlm_usage import VlmUsage
    from src.services.curation.assist_loop.record import LoopRecord

DEFAULT_MAX_WALL_S = 6 * 3600
MAX_TOKENS_PER_IMAGE = 768
"""Upper bound of the completion tokens one classified image can cost (the classify call's
``max_tokens``)."""
PROMPT_TEXT_TOKENS = 600
"""The text part of one classify prompt, in tokens, added to the per-image estimate."""
_REASON = {
    'crops': 'budget_crops',
    'requests': 'budget_requests',
    'prompt_tokens': 'budget_tokens',
    'completion_tokens': 'budget_tokens',
    'wall_s': 'budget_time',
}


@dataclass(frozen=True)
class PoolLimits:
    max_crops: int
    max_requests: int


def _default_requests(max_crops: int, images_per_call: int, concurrency: int) -> int:
    return math.ceil(max_crops / max(1, images_per_call)) + concurrency


@dataclass(frozen=True)
class BudgetLimits:
    pools: Mapping[str, PoolLimits]
    max_wall_s: float

    @classmethod
    def from_request(
        cls,
        *,
        max_vlm_crops: int | None,
        max_vlm_requests: int | None,
        max_wall_s: float | None,
        images_per_call: int,
        vlm_concurrency: int,
        audit_max_crops: int,
    ) -> BudgetLimits:
        """The caps of a loop request. ``max_vlm_crops`` has no default: a loop
        without one is refused."""
        if max_vlm_crops is None or max_vlm_crops < 1:
            msg = 'max_vlm_crops is required and must be at least 1'
            raise ValueError(msg)
        vlm_requests = max_vlm_requests or _default_requests(
            max_vlm_crops, images_per_call, vlm_concurrency
        )
        return cls(
            pools={
                VLM: PoolLimits(max_vlm_crops, vlm_requests),
                AUDIT: PoolLimits(
                    audit_max_crops,
                    _default_requests(audit_max_crops, images_per_call, vlm_concurrency),
                ),
            },
            max_wall_s=float(max_wall_s or DEFAULT_MAX_WALL_S),
        )

    def seed(self) -> dict[tuple[str, str], int | None]:
        """The ``(pool, unit)`` limits a new run's budget rows start with."""
        out: dict[tuple[str, str], int | None] = {}
        for pool, limits in self.pools.items():
            out[pool, 'crops'] = limits.max_crops
            out[pool, 'requests'] = limits.max_requests
            out[pool, 'prompt_tokens'] = None
            out[pool, 'completion_tokens'] = None
        out[VLM, 'wall_s'] = int(self.max_wall_s)
        return out


@dataclass(frozen=True)
class Reservation:
    request_id: uuid.UUID
    pool: str
    path: str
    ids: tuple[str, ...]
    units: Mapping[str, int]

    @property
    def crops(self) -> int:
        return self.units['crops']


class LoopBudget:
    """The loop's budgets for one slice."""

    def __init__(self, record: LoopRecord, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._record = record
        self._clock = clock
        self._rows: dict[tuple[str, str], repo.BudgetRow] = {}
        self._started = clock()
        self._charged = 0.0
        self._refused: dict[str, str] = {}
        self.usage_missing = 0
        """Settled calls this slice with no usage object (charged at the reservation)."""

    async def start_slice(self) -> int:
        """Forfeit what a dead slice left reserved, then read the rows. Returns how many."""
        rec = self._record
        forfeited = await rec.with_session(
            lambda s: repo.forfeit_open(s, run_id=rec.run_id, job_id=rec.job_id, token=rec.token)
        )
        await self.refresh()
        return forfeited

    async def refresh(self) -> None:
        rec = self._record
        found = await rec.with_session(lambda s: repo.budgets(s, rec.run_id))
        self._rows = {(r.pool, r.unit): r for r in found}

    # ------------------------------------------------------------------ time

    def wall_s(self) -> float:
        """Running time of the loop over its slices so far."""
        row = self._rows.get((VLM, 'wall_s'))
        return (row.spent if row else 0) + (self._clock() - self._started) - self._charged

    def time_exhausted(self) -> bool:
        row = self._rows.get((VLM, 'wall_s'))
        return row is not None and row.limit_value is not None and self.wall_s() >= row.limit_value

    async def charge_wall(self) -> None:
        """Charge the slice's running time to ``wall_s`` (at a yield or a stop)."""
        seconds = math.floor(self._clock() - self._started - self._charged)
        if seconds < 1:
            return
        rec = self._record
        await rec.with_session(
            lambda s: repo.add_spent(
                s,
                run_id=rec.run_id,
                job_id=rec.job_id,
                token=rec.token,
                pool=VLM,
                unit='wall_s',
                amount=seconds,
            )
        )
        self._charged += seconds
        await self.refresh()

    # ------------------------------------------------------------------ reservations

    @property
    def refused(self) -> str | None:
        """Why the VLM pool last refused a reservation; ``None`` when it never did."""
        return self._refused.get(VLM)

    def _held(self, pool: str, unit: str) -> tuple[int, int | None]:
        row = self._rows.get((pool, unit))
        if row is None:
            return 0, None
        return row.spent + row.reserved, row.limit_value

    def would_refuse(self, pool: str, *, crops: int, requests: int) -> str | None:
        """Why a reservation of this size would be refused now (from the cached counters), or
        ``None``. Nothing is reserved."""
        if pool == VLM and self.time_exhausted():
            return 'budget_time'
        for unit, amount in (('crops', crops), ('requests', requests)):
            held, limit = self._held(pool, unit)
            if limit is not None and held + amount > limit:
                return _REASON[unit]
        return None

    async def reserve(self, pool: str, path: str, ids: Sequence[str]) -> Reservation | None:
        """Reserve the units of one call over ``ids``; ``None`` when a limit would be passed."""
        rec = self._record
        if pool == VLM and self.time_exhausted():
            self._refused[pool] = 'budget_time'
            return None
        history = await rec.with_session(
            lambda s: record_rows.prompt_history(s, rec.run_id, calls=PROMPT_HISTORY_CALLS)
        )
        units = call_units(
            images=len(ids),
            max_tokens=MAX_TOKENS_PER_IMAGE * len(ids),
            prompt_length_tokens=PROMPT_TEXT_TOKENS,
            recent_per_image=history,
        ).as_amounts()
        request_id = uuid.uuid4()

        def write(session: Session) -> None:
            repo.reserve(
                session,
                run_id=rec.run_id,
                job_id=rec.job_id,
                token=rec.token,
                request_id=request_id,
                pool=pool,
                amounts=units,
            )

        try:
            await rec.with_session(write)
        except repo.BudgetExhaustedError as exc:
            self._refused[pool] = _REASON[exc.unit]
            await self.refresh()
            return None
        await self.refresh()
        return Reservation(request_id, pool, path, tuple(ids), units)

    async def settle(self, res: Reservation, usage: VlmUsage) -> None:
        """The call returned (answered or failed): move its reservation to spent."""
        rec = self._record
        actual = settled_amounts(res.units, usage)
        if not usage.present:
            self.usage_missing += 1
        await rec.with_session(
            lambda s: repo.settle(
                s,
                run_id=rec.run_id,
                job_id=rec.job_id,
                token=rec.token,
                request_id=res.request_id,
                actual=actual,
            )
        )
        await self.refresh()

    # ------------------------------------------------------------------ views

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Per pool: limits, spent and reserved, by unit (a limit of ``-1`` is uncapped)."""
        out: dict[str, dict[str, int]] = {}
        for (pool, unit), row in sorted(self._rows.items()):
            if unit == 'wall_s':
                continue
            view = out.setdefault(pool, {})
            view[f'max_{unit}'] = -1 if row.limit_value is None else row.limit_value
            view[f'{unit}_spent'] = row.spent
            view[f'{unit}_reserved'] = row.reserved
        return out

    def rows(self) -> list[repo.BudgetRow]:
        return list(self._rows.values())


__all__ = [
    'AUDIT',
    'DEFAULT_MAX_WALL_S',
    'VLM',
    'BudgetLimits',
    'LoopBudget',
    'PoolLimits',
    'Reservation',
]
