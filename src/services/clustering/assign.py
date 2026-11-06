"""Incremental placement of a new item by the serving partition (plan 5.2, 7.1; L1c).

Incremental assignment is not a job: the embedding stage of ingest places each new item at the
serving partition's nearest prototype with the calibrated bands and writes the item's cluster
fields with ``cluster_assigned_by = 'incremental'``. Prototypes are read from PostgreSQL once per
partition (an immutable row set, so a cached copy can never be stale) and kept in a small
in-process cache.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

ASSIGNED_BY = 'incremental'
PROTOTYPE_CACHE_SIZE = 8
SIM_CHUNK = 20_000


class AssignBands(BaseModel):
    """Body of a ``cluster_calibrations`` row of kind ``assign_bands``."""

    model_config = ConfigDict(extra='forbid')

    assign_min: float = Field(ge=-1.0, le=1.0)
    """Score at or above which (with the margin) an item is ``assigned``."""
    review_min: float = Field(ge=-1.0, le=1.0)
    """Score at or above which a weaker item is ``review``; below it ``unassigned``."""
    min_margin: float = Field(default=0.0, ge=0.0, le=2.0)


def band_state(score: float, margin: float, bands: AssignBands | None) -> str:
    """``assigned`` without a calibration (today's behaviour); else by the calibrated bands."""
    if bands is None:
        return 'assigned'
    if score >= bands.assign_min and margin >= bands.min_margin:
        return 'assigned'
    if score >= bands.review_min:
        return 'review'
    return 'unassigned'


@dataclass(frozen=True)
class Placement:
    item_id: str
    partition_id: uuid.UUID
    cluster_uid: uuid.UUID | None
    state: str
    score: float
    margin: float
    assigned_by: str = ASSIGNED_BY


@dataclass(frozen=True)
class PrototypeSet:
    partition_id: uuid.UUID
    uids: tuple[uuid.UUID, ...]
    matrix: np.ndarray
    """One unit-norm row per prototype; ``uids[i]`` is the owner of row ``i``."""
    bands: AssignBands | None


_cache: OrderedDict[uuid.UUID, PrototypeSet] = OrderedDict()


def forget_prototypes() -> None:
    _cache.clear()


async def serving_partition_id(
    session: AsyncSession, *, project: uuid.UUID, target: str, level: str = 'class'
) -> uuid.UUID | None:
    row = await session.execute(
        text(
            'SELECT serving_partition_id FROM cluster_activations WHERE project_id = :p '
            'AND target = :t AND level = :l ORDER BY updated_at DESC LIMIT 1'
        ),
        {'p': project, 't': target, 'l': level},
    )
    found = row.scalar_one_or_none()
    return found if found is None else uuid.UUID(str(found))


async def load_prototypes(session: AsyncSession, partition_id: uuid.UUID) -> PrototypeSet | None:
    cached = _cache.get(partition_id)
    if cached is not None:
        _cache.move_to_end(partition_id)
        return cached
    rows = (
        await session.execute(
            text(
                'SELECT pc.cluster_uid, pc.prototypes, cal.body FROM partition_clusters pc '
                'JOIN cluster_partitions cp ON cp.id = pc.partition_id '
                'LEFT JOIN cluster_calibrations cal ON cal.id = cp.calibration_id '
                'WHERE pc.partition_id = :id ORDER BY pc.display_no'
            ),
            {'id': partition_id},
        )
    ).all()
    uids: list[uuid.UUID] = []
    vectors: list[list[float]] = []
    for uid, prototypes, _ in rows:
        for prototype in prototypes or ():
            vector = prototype.get('vector')
            if vector:
                uids.append(uid)
                vectors.append(vector)
    if not vectors:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    body: Any = rows[0][2]
    bands = AssignBands.model_validate(body) if body else None
    found = PrototypeSet(partition_id, tuple(uids), matrix, bands)
    _cache[partition_id] = found
    while len(_cache) > PROTOTYPE_CACHE_SIZE:
        _cache.popitem(last=False)
    return found


def place_vectors(prototypes: PrototypeSet, vectors: np.ndarray) -> list[tuple[int, float, float]]:
    """``(best prototype row, score, margin)`` for each vector; the margin is to the best row of
    a different cluster (the top score when only one cluster exists)."""
    x = np.asarray(vectors, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    codes: dict[uuid.UUID, int] = {}
    owners = np.asarray([codes.setdefault(u, len(codes)) for u in prototypes.uids], dtype=np.int64)
    out: list[tuple[int, float, float]] = []
    for start in range(0, len(x), SIM_CHUNK):
        sims = x[start : start + SIM_CHUNK] @ prototypes.matrix.T
        best = sims.argmax(axis=1)
        top = sims[np.arange(len(sims)), best]
        other = np.where(owners[None, :] == owners[best][:, None], -np.inf, sims).max(axis=1)
        runner = np.where(np.isfinite(other), other, top - 1.0)
        out.extend(
            (int(b), float(s), float(s - r)) for b, s, r in zip(best, top, runner, strict=True)
        )
    return out


async def place_new(
    session: AsyncSession,
    *,
    project: uuid.UUID,
    target: str,
    item_ids: Sequence[str],
    vectors: np.ndarray,
) -> list[Placement]:
    """Place new items by the serving partition's prototypes with its calibrated bands.

    Empty when the project has no serving partition for ``target``: nothing to place into, and
    the caller writes no cluster field (an unclustered project behaves as it does today).
    """
    if len(item_ids) != len(vectors):
        msg = 'item_ids and vectors differ in length'
        raise ValueError(msg)
    if not item_ids:
        return []
    partition_id = await serving_partition_id(session, project=project, target=target)
    if partition_id is None:
        return []
    prototypes = await load_prototypes(session, partition_id)
    if prototypes is None:
        return []
    placements = []
    for item_id, (row, score, margin) in zip(
        item_ids, place_vectors(prototypes, vectors), strict=True
    ):
        state = band_state(score, margin, prototypes.bands)
        uid = None if state == 'unassigned' else prototypes.uids[row]
        placements.append(Placement(item_id, partition_id, uid, state, score, margin))
    return placements


__all__ = [
    'AssignBands',
    'Placement',
    'band_state',
    'forget_prototypes',
    'place_new',
    'place_vectors',
]
