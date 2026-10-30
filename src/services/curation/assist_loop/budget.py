"""Write-ahead budgets of an auto-label loop: VLM crops, requests and wall time.

Every request is covered by a reservation made, logged (``reserve`` event) and written
to ``budget.json`` before the request is sent; a reservation that would take
``spent + reserved`` past a cap is refused. A returned chunk moves its share from
reserved to spent; failed and retried requests are spent too. On resume every
reservation in the log counts as spent, settled or not, so a crash can waste the
requests that were in flight but can never take a pool past its cap.

The audit's second-opinion requests draw on their own pool with its own cap.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.assist_loop.store import AUDIT, VLM


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from src.services.curation.assist_loop.store import LoopStore

DEFAULT_MAX_WALL_S = 6 * 3600
BUDGET_FILE = 'budget.json'


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


@dataclass(frozen=True)
class Reservation:
    rid: int
    pool: str
    path: str
    ids: tuple[str, ...]
    crops: int
    requests: int


@dataclass
class _Counts:
    crops_spent: int = 0
    crops_reserved: int = 0
    requests_spent: int = 0
    requests_reserved: int = 0


class LoopBudget:
    """The loop's budgets, rebuilt from the event log of ``store``."""

    def __init__(
        self,
        store: LoopStore,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._counts = {pool: _Counts() for pool in limits.pools}
        self._open: dict[int, Reservation] = {}
        self._refused: dict[str, str] = {}
        rids = [0]
        wall = [0.0]
        for event in store.events():
            if 'wall_s' in event:
                wall.append(float(event['wall_s']))
            if event.get('event') != 'reserve':
                continue
            rids.append(int(event['rid']))
            counts = self._counts.setdefault(event['pool'], _Counts())
            counts.crops_spent += int(event['crops'])
            counts.requests_spent += int(event['requests'])
        self._next_rid = max(rids) + 1
        self._wall_before = max(wall)
        self._started = clock()

    # ------------------------------------------------------------------ time

    def wall_s(self) -> float:
        """Running time of the loop over all its sessions."""
        return self._wall_before + (self._clock() - self._started)

    def time_exhausted(self) -> bool:
        return self.wall_s() >= self._limits.max_wall_s

    # ------------------------------------------------------------------ reservations

    @property
    def refused(self) -> str | None:
        """Why the VLM pool last refused a reservation (``budget_crops``,
        ``budget_requests``, ``budget_time``); ``None`` when it never did."""
        return self._refused.get(VLM)

    def _refusal(self, pool: str, crops: int, requests: int) -> str | None:
        if pool == VLM and self.time_exhausted():
            return 'budget_time'
        limits = self._limits.pools[pool]
        counts = self._counts[pool]
        if counts.crops_spent + counts.crops_reserved + crops > limits.max_crops:
            return 'budget_crops'
        if counts.requests_spent + counts.requests_reserved + requests > limits.max_requests:
            return 'budget_requests'
        return None

    def would_refuse(self, pool: str, *, crops: int, requests: int) -> str | None:
        """Why a reservation of this size would be refused now, or ``None`` when it would
        be granted. Nothing is reserved."""
        with self._lock:
            return self._refusal(pool, crops, requests)

    def reserve(
        self, pool: str, path: str, ids: Sequence[str], *, requests: int = 1
    ) -> Reservation | None:
        """Reserve ``ids`` (one crop each) and ``requests`` for node ``path``; ``None``
        when a cap would be exceeded. The reservation is on disk when this returns; if
        writing it fails, the error propagates and no request may be sent."""
        crops = len(ids)
        with self._lock:
            reason = self._refusal(pool, crops, requests)
            if reason is not None:
                self._refused[pool] = reason
                return None
            res = Reservation(self._next_rid, pool, path, tuple(ids), crops, requests)
            self._store.append(
                'reserve',
                pool=pool,
                rid=res.rid,
                path=path,
                ids=list(ids),
                crops=crops,
                requests=requests,
                wall_s=round(self.wall_s(), 3),
            )
            self._next_rid += 1
            self._open[res.rid] = res
            counts = self._counts[pool]
            counts.crops_reserved += crops
            counts.requests_reserved += requests
            self._persist()
            return res

    def settle(self, res: Reservation) -> None:
        """The reservation's requests returned (answered or failed): its share is spent.
        A reservation this budget did not make (made before a resume) is spent already."""
        with self._lock:
            if self._open.pop(res.rid, None) is None:
                return
            counts = self._counts[res.pool]
            counts.crops_reserved -= res.crops
            counts.requests_reserved -= res.requests
            counts.crops_spent += res.crops
            counts.requests_spent += res.requests
            self._persist()

    # ------------------------------------------------------------------ views

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Per pool: caps, spent and reserved."""
        with self._lock:
            return self._view()

    def _view(self) -> dict[str, dict[str, int]]:
        return {
            pool: {
                'max_crops': self._limits.pools[pool].max_crops,
                'max_requests': self._limits.pools[pool].max_requests,
                **vars(counts),
            }
            for pool, counts in self._counts.items()
            if pool in self._limits.pools
        }

    def _persist(self) -> None:
        body: dict[str, Any] = dict(self._view())
        body['wall_s'] = round(self.wall_s(), 3)
        body['max_wall_s'] = self._limits.max_wall_s
        self._store.write_json(BUDGET_FILE, body)


__all__ = [
    'AUDIT',
    'DEFAULT_MAX_WALL_S',
    'VLM',
    'BudgetLimits',
    'LoopBudget',
    'PoolLimits',
    'Reservation',
]
