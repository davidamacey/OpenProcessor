"""The fold's output: ``item_state`` and ``box_state`` rows (15a section 2.4, R7).

Both tables are derived. A fold writes only rows that differ (``IS DISTINCT FROM``), so an
unchanged item writes no tuple and a label write is HOT whenever no indexed column changes
(https://www.postgresql.org/docs/18/storage-hot.html); ``updated_at`` moves only with another column.
Callers hold the item locks: no path writes a state row without first holding its ``items`` lock.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import ARRAY, BigInteger, Text, bindparam, delete, select, text

from src.db.models.item_state import BoxStateRow, ItemStateRow
from src.db.repos.upserts import any_of, upsert_distinct
from src.domain.wire_ids import to_uuid


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.fold_state import BoxState, ItemFold, ItemState


def _uuid(value: str | None) -> uuid.UUID | None:
    return None if value is None else uuid.UUID(value)


def item_state_row(project_id: uuid.UUID, item_key: int, s: ItemState) -> dict[str, Any]:
    """The columns of 15a section 2.4. Not stored, because they are derived at read from rows that
    are: the class name (the registry), the deciding claim's labeler and profile, the VLM
    suggestion (its opinion row), ``excluded_by_human`` and the two wire lock flags."""
    return {
        'item_key': item_key,
        'project_id': project_id,
        'photo_key': s.photo_key,
        'crop_id': to_uuid(s.crop_id),
        'bbox': list(s.bbox),
        'area': s.area,
        'rank': s.rank,
        'geometry_source': s.geometry_source,
        'geometry_revision': s.geometry_revision,
        'class_id': s.class_id,
        'class_source': s.class_source,
        'label_source': s.label_source,
        'class_validated': s.class_validated,
        'class_unresolved': s.class_unresolved,
        'class_labeled_at': s.class_labeled_at,
        'class_confidence': s.class_confidence,
        'class_claim_id': s.class_claim_id,
        'class_op_id': _uuid(s.class_op_id),
        'class_opinion_source': s.class_opinion_source,
        'vlm_suggestion_dismissed': s.vlm_suggestion_dismissed,
        'needs_new_class': s.needs_new_class,
        'probe_dismissed_class': s.probe_dismissed_class,
        'excluded': s.excluded,
        'excluded_reason': s.excluded_reason,
        'excluded_at': s.excluded_at,
        'holdout': s.holdout,
        'review_dismissed_at': s.review_dismissed_at,
        'audit_sample': s.audit_sample,
        'audit_outcome': s.audit_outcome,
        'combine_conflict': s.combine_conflict,
        'import_ids': [uuid.UUID(i) for i in s.import_ids] or None,
        'region_status': s.region_status,
        'region_validated': s.region_validated,
        'region_verifier': s.region_verifier,
        'region_label_source': s.region_label_source,
        'region_count': s.region_count,
        'region_rejected_count': s.region_rejected_count,
        'region_max_score': s.region_max_score,
        'region_revision': s.region_revision,
        'region_set_complete': s.region_set_complete,
        'region_seen_seq': s.region_seen_seq,
        'geometry_pinned': s.geometry_pinned,
        'moved_partition_id': _uuid(s.moved_partition_id),
        'moved_cluster_uid': _uuid(s.moved_cluster_uid),
        'created_at': s.created_at,
        'updated_at': s.updated_at,
    }


def box_state_row(
    project_id: uuid.UUID, item_key: int, b: BoxState, box_keys: Mapping[str, int]
) -> dict[str, Any]:
    return {
        'project_id': project_id,
        'item_key': item_key,
        'box_id': b.box_id,
        'box_key': None if b.from_claims_only else box_keys.get(b.box_id),
        'bbox': list(b.bbox),
        'state': b.state,
        'score': b.score,
        'detector': b.detector,
        'detector_version': b.detector_version,
        'source': b.source,
        'bbox_correct': b.bbox_correct,
        'confidence': b.confidence,
        'rejection_reason': b.rejection_reason,
        'text': b.text,
        'text_source': b.text_source,
        'text_choice': b.text_choice,
        'text_disagreement': b.text_disagreement,
        'text_fold': b.text_fold,
        'ordinal': b.ordinal,
        'detected_at': None if b.detected_at is None else datetime.fromisoformat(b.detected_at),
    }


async def upsert_item_states(
    session: AsyncSession, project_id: uuid.UUID, folds: Mapping[int, ItemFold]
) -> int:
    """Write the item rows that differ from their fold; the rows inserted or changed."""
    return await upsert_distinct(
        session,
        ItemStateRow,
        [item_state_row(project_id, k, f.state) for k, f in sorted(folds.items())],
        key=('item_key',),
        no_compare=('updated_at',),
    )


_STALE_BOXES = (
    'DELETE FROM box_state b WHERE b.item_key = ANY(:items) AND NOT EXISTS ('
    'SELECT 1 FROM unnest(:keep_items, :keep_boxes) AS k(item_key, box_id) '
    'WHERE k.item_key = b.item_key AND k.box_id = b.box_id)'
)


async def upsert_box_states(
    session: AsyncSession,
    project_id: uuid.UUID,
    folds: Mapping[int, ItemFold],
    box_keys: Mapping[int, Mapping[str, int]],
) -> int:
    """Write the box rows that differ and delete the rows of boxes the fold no longer shows."""
    rows = [
        box_state_row(project_id, k, b, box_keys.get(k, {}))
        for k, f in sorted(folds.items())
        for b in f.boxes
    ]
    changed = await upsert_distinct(session, BoxStateRow, rows, key=('item_key', 'box_id'))
    gone = await session.execute(
        text(_STALE_BOXES).bindparams(
            bindparam('items', type_=ARRAY(BigInteger())),
            bindparam('keep_items', type_=ARRAY(BigInteger())),
            bindparam('keep_boxes', type_=ARRAY(Text())),
        ),
        {
            'items': sorted(folds),
            'keep_items': [r['item_key'] for r in rows],
            'keep_boxes': [r['box_id'] for r in rows],
        },
    )
    return changed + int(gone.rowcount)  # type: ignore[attr-defined]


async def load_ranks(session: AsyncSession, item_keys: Sequence[int]) -> dict[int, int]:
    """The stored rank of each item that has a state row."""
    if not item_keys:
        return {}
    rows = await session.execute(
        select(ItemStateRow.item_key, ItemStateRow.rank).where(
            any_of(ItemStateRow.item_key, item_keys)
        )
    )
    return {r.item_key: r.rank for r in rows}


async def delete_states(session: AsyncSession, item_keys: Sequence[int]) -> int:
    """Remove the state rows of items the fold no longer shows (neither a detection nor a live
    asserting claim). The ``items`` identity row is the attach step's to delete, not this."""
    if not item_keys:
        return 0
    boxes = await session.execute(
        delete(BoxStateRow).where(any_of(BoxStateRow.item_key, item_keys))
    )
    items = await session.execute(
        delete(ItemStateRow).where(any_of(ItemStateRow.item_key, item_keys))
    )
    return int(boxes.rowcount) + int(items.rowcount)  # type: ignore[attr-defined]
