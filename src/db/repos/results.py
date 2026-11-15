"""Upsert of machine results, per analyser (15a section 2.3, R4 and R5).

A result row is replaced by the next run of the same analyser and kept when it holds the same
values (``IS DISTINCT FROM``). Nothing here reads or writes a claim: a machine writer cannot undo
what a person decided, so no result write needs a lock check (R5). Callers hold the item locks.

Two rules live in the statement, not in the callers: a VLM answer without a class never replaces a
class opinion that has one (the attempt goes to the analysis ledger, 15a M17), and a row that carries
``op_id`` or ``import_id`` is not rebuildable (R2) and is only ever replaced by the same key.
Append-only result families (``text_lines``, ``item_notes``) are written by their own lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import ARRAY, BigInteger, Text, bindparam, or_, text

from src.db.models.boxes import BoxText, RegionBox
from src.db.models.gallery import EntityEdge, PhotoDerivative, PhotoPlace, PhotoSignal
from src.db.models.items import ClassOpinion, Detection, Face, ItemRegionResult, RawLabelCluster
from src.db.models.photos import PhotoCapture
from src.db.models.scores import AssistAnswer, ItemProjection, ItemScore, ProbePrediction
from src.db.repos.upserts import upsert_distinct


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement


class ResultRowError(ValueError):
    """A result row that names a table or a box that does not exist (a writer's bug)."""


@dataclass(frozen=True)
class ResultSpec:
    model: Any
    key: tuple[str, ...]


@dataclass(frozen=True)
class ResultWrite:
    """The rows one analyser run writes to one table. Rows share their columns; ``project_id`` is
    added. ``box_texts`` rows name their box by ``item_key`` and ``box_id``, not ``box_key``."""

    table: str
    rows: Sequence[Mapping[str, Any]]


# Parents before children: region_boxes before the readings that point at them.
RESULT_SPECS: dict[str, ResultSpec] = {
    'photo_capture': ResultSpec(PhotoCapture, ('photo_key',)),
    'detections': ResultSpec(Detection, ('item_key', 'analyser_id')),
    'class_opinions': ResultSpec(ClassOpinion, ('item_key', 'source')),
    'raw_label_clusters': ResultSpec(RawLabelCluster, ('project_id', 'raw_label')),
    'item_region_results': ResultSpec(ItemRegionResult, ('item_key',)),
    'region_boxes': ResultSpec(RegionBox, ('item_key', 'box_id')),
    'box_texts': ResultSpec(BoxText, ('box_key', 'reader')),
    'probe_predictions': ResultSpec(ProbePrediction, ('item_key',)),
    'item_scores': ResultSpec(ItemScore, ('item_key', 'kind')),
    'item_projections': ResultSpec(ItemProjection, ('item_key',)),
    'assist_answers': ResultSpec(AssistAnswer, ('item_key',)),
    'faces': ResultSpec(Face, ('project_id', 'face_id')),
    'entity_edges': ResultSpec(EntityEdge, ('entity_key', 'subject_kind', 'subject_key')),
    'photo_derivatives': ResultSpec(PhotoDerivative, ('photo_key',)),
    'photo_signals': ResultSpec(PhotoSignal, ('photo_key',)),
    'photo_places': ResultSpec(PhotoPlace, ('photo_key',)),
}
_ORDER = {name: n for n, name in enumerate(RESULT_SPECS)}

# Whether a write to the table can change the rank of an item among its photo's items: only a detection
# (a box, or an item appearing or going). Every table of RESULT_SPECS is in exactly one of the two sets:
# a table added without a classification fails tests/storage/test_apply_offline.py.
RANK_CHANGING_TABLES = frozenset({'detections'})
RANK_NEUTRAL_TABLES = frozenset(
    {
        'photo_capture', 'class_opinions', 'raw_label_clusters', 'item_region_results',
        'region_boxes', 'box_texts', 'probe_predictions', 'item_scores', 'item_projections',
        'assist_answers', 'faces', 'entity_edges', 'photo_derivatives', 'photo_signals',
        'photo_places',
    }
)  # fmt: skip

_BOX_KEYS = text(
    """
    SELECT r.item_key, r.box_id, r.box_key
    FROM region_boxes r
    JOIN unnest(:items, :boxes) AS t(item_key, box_id)
      ON r.item_key = t.item_key AND r.box_id = t.box_id
    """
).bindparams(bindparam('items', type_=ARRAY(BigInteger())), bindparam('boxes', type_=ARRAY(Text())))


def _opinion_guard(table: Any, excluded: Any) -> ColumnElement[bool]:
    """A class opinion without a class never replaces one that has a class (M17)."""
    return or_(excluded.class_name.is_not(None), table.c.class_name.is_(None))


async def _with_box_keys(
    session: AsyncSession, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    wanted = sorted({(r['item_key'], r['box_id']) for r in rows})
    found = {
        (r.item_key, r.box_id): r.box_key
        for r in await session.execute(
            _BOX_KEYS, {'items': [i for i, _ in wanted], 'boxes': [b for _, b in wanted]}
        )
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        box_key = found.get((row['item_key'], row['box_id']))
        if box_key is None:
            raise ResultRowError(f'box_texts: item {row["item_key"]} has no box {row["box_id"]!r}')
        out.append(
            {
                'box_key': box_key,
                **{k: v for k, v in row.items() if k not in ('item_key', 'box_id')},
            }
        )
    return out


async def upsert_results(
    session: AsyncSession, project_id: uuid.UUID, writes: Sequence[ResultWrite]
) -> int:
    """Upsert every write, parents first; the rows inserted or changed."""
    unknown = sorted({w.table for w in writes} - set(RESULT_SPECS))
    if unknown:
        raise ResultRowError(f'not a result table: {unknown}')
    changed = 0
    for write in sorted(writes, key=lambda w: _ORDER[w.table]):
        if not write.rows:
            continue
        spec = RESULT_SPECS[write.table]
        rows = [{'project_id': project_id, **r} for r in write.rows]
        if write.table == 'box_texts':
            rows = await _with_box_keys(session, rows)
        changed += await upsert_distinct(
            session,
            spec.model,
            rows,
            key=spec.key,
            guard=_opinion_guard if write.table == 'class_opinions' else None,
        )
    return changed
