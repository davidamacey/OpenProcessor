"""Why ``GET /clusters`` is empty, and the built partition waiting to be activated.

The list only reads the serving partition, so a project whose run finished but was never
activated shows no clusters. The grid needs that said, with the partition to activate, instead of
guessing from an empty list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.config.project_context import current_project_id
from src.db.session import project_tx
from src.services.clustering.partitions import list_partitions


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.curation.clusters import PendingPartition
    from src.services.clustering.partitions import PartitionRow


EmptyReason = Literal['no_partition', 'pending_activation', 'filtered_empty']

_MESSAGES: dict[str, str] = {
    'no_partition': 'No clustering run has been served yet. Run clustering from the Dashboard.',
    'pending_activation': (
        'A clustering run finished but is not serving yet. Activate it to see its clusters.'
    ),
    'filtered_empty': 'No cluster matches the current filters.',
}


@dataclass(frozen=True)
class PartitionState:
    serving: bool
    pending: PendingPartition | None


def partition_state(rows: Sequence[PartitionRow]) -> PartitionState:
    """From the item partitions, newest first: whether one serves, and the newest ready one that
    does not and is newer than the serving one (any ready one when none serves)."""
    pending: PendingPartition | None = None
    serving = False
    for row in rows:
        if row.serving:
            serving = True
            break
        if pending is None and row.status == 'ready':
            pending = {
                'partition_id': str(row.id),
                'clusters': int(row.n_clusters or 0),
                'items': int(row.n_items or 0),
            }
    return PartitionState(serving=serving, pending=pending)


async def read_partition_state(session: AsyncSession) -> PartitionState:
    async with project_tx(session, current_project_id()):
        rows = await list_partitions(session, target='item', limit=20)
    return partition_state(rows)


def empty_reason(
    state: PartitionState, *, total: int, filtered: bool
) -> tuple[EmptyReason | None, str | None]:
    """The served reason and text for an empty list; ``(None, None)`` when cards are served."""
    if total > 0:
        return None, None
    reason: EmptyReason
    if filtered:
        reason = 'filtered_empty'
    elif state.pending is not None and not state.serving:
        reason = 'pending_activation'
    elif not state.serving:
        reason = 'no_partition'
    else:
        reason = 'filtered_empty'
    return reason, _MESSAGES[reason]


def has_card_filters(item_filter_empty: bool, params: Any) -> bool:
    """Whether anything narrows the list: the shared item filter or a card filter."""
    return not item_filter_empty or bool(
        params.dominant_class
        or params.detector_class
        or params.min_size is not None
        or params.max_size is not None
        or params.validation_state is not None
        or params.flags
        or params.needs_review is not None
        or params.cluster_uid is not None
    )
