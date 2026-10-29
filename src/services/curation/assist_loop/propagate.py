"""Write a passed node's class onto its other eligible members.

The members written are the node's snapshot-eligible members minus every member the node
has an answer for (an agreeing one already carries the VLM's class, a dissenting one keeps
its own answer), minus the members that left the root since the snapshot (*drifted*). If
more than ``1 - SAME_SCOPE_JACCARD`` of the node's eligible members drifted, the cluster
moved under the loop and the node is ``stale``: nothing is written. Otherwise the write is
the one an explicit assist apply makes (:func:`~src.services.curation.assist.apply_write.write_class_to_ids`)
with the snapshot class-state tokens, so an item locked, validated, put in the holdout or
changed by anyone since the snapshot is skipped under OCC and counted.

On a resume (``resuming``) the members that already carry this loop's write are counted as
written, not read as drift (the write moved them to their class cluster) and not written
again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.crops import mget_crops
from src.domain.ingest_class_sources import PROPAGATED_CLASS_SOURCE
from src.services.curation.assist.apply import SAME_SCOPE_JACCARD
from src.services.curation.assist.apply_write import write_class_to_ids


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist.proposal import ClassAnswer
    from src.services.curation.assist_loop.partition import PartitionReader
    from src.services.curation.class_label import ItemLabel

SKIP_REASONS = ('dissent', 'drifted', 'locked', 'token_changed', 'conflicts')
"""Why a member of a passed node was not written."""
STALE_DRIFT = 1 - SAME_SCOPE_JACCARD
"""Share of a node's eligible members that may have left its root before it is stale."""


@dataclass
class PropagateResult:
    stale: bool
    drift: float
    written_ids: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    """``dissent``, ``drifted``, ``locked``, ``token_changed``, ``conflicts``."""


async def _written_by(opensearch: AsyncOpenSearch, ids: Sequence[str], run_id: str) -> set[str]:
    if not ids:
        return set()
    docs = await mget_crops(
        opensearch, list(ids), source_includes=['class_source', 'class_detector_version']
    )
    return {
        crop_id
        for crop_id, doc in docs.items()
        if (src := doc.get('_source') or {}).get('class_source') == PROPAGATED_CLASS_SOURCE
        and src.get('class_detector_version') == run_id
    }


async def propagate_node(
    opensearch: AsyncOpenSearch,
    reader: PartitionReader,
    *,
    root_key: str,
    eligible: Sequence[str],
    answers: Mapping[str, ClassAnswer],
    tokens: Mapping[str, tuple[Any, ...]],
    label: ItemLabel,
    resuming: bool = False,
) -> PropagateResult:
    """Propagate ``label`` over the node whose snapshot-eligible members are ``eligible``."""
    dissent = sum(1 for a in answers.values() if a.class_name != label.class_name)
    candidates = [i for i in eligible if i not in answers]
    already = await _written_by(opensearch, candidates, label.run_id or '') if resuming else set()
    rest = [i for i in candidates if i not in already]
    where = await reader.root_of(rest) if rest else {}
    drifted = [i for i in rest if where.get(i) != root_key]
    drift = len(drifted) / len(eligible) if eligible else 0.0
    skipped = dict.fromkeys(SKIP_REASONS, 0)
    skipped.update(dissent=dissent, drifted=len(drifted))
    if drift > STALE_DRIFT:
        return PropagateResult(stale=True, drift=round(drift, 4), skipped=skipped)
    ids = [i for i in rest if where.get(i) == root_key]
    written = await write_class_to_ids(
        opensearch, ids, label, tokens={i: tokens[i] for i in ids if i in tokens}
    )
    skipped['locked'] = len(written.skipped_locked)
    skipped['token_changed'] = len(written.skipped_changed)
    skipped['conflicts'] = len(written.conflicts)
    landed = set(written.updated_ids) | already
    return PropagateResult(
        stale=False,
        drift=round(drift, 4),
        written_ids=[i for i in eligible if i in landed],
        skipped=skipped,
    )


__all__ = ['SKIP_REASONS', 'STALE_DRIFT', 'PropagateResult', 'propagate_node']
