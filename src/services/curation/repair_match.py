"""Which detections a repair may create: the ones no item of the image already stands for.

A crop id is the SHA-256 of the box to 6 decimals and the detector is not bit-stable, so a
repair's detections can carry new ids for objects the image already has an item for (the items the
first, failed attempt wrote; a box a person drew or moved; an imported or open-vocabulary item).
A repair never rewrites or replaces those items; it matches its detections to them with the same
rule a re-detection and an import's propose mode use
(:func:`~src.services.curation.proposal_merge.merge_item_proposals`: greedy by IoU, a match at
:data:`~src.services.curation.proposal_merge.LABEL_IOU_MATCH` or above, a person-moved box also
owns the detector's original spot) and creates only the detections left unmatched.

The match does not compare labels, as that rule does not: a detector that names the same object
differently on another run is still the same object, and two items for it would be the duplicate
this exists to prevent.

The items read are the image's listed ones (a real-time ``mget``: an item written a moment ago is
there) and every other item of the image (a search after a refresh: drawn, imported and
open-vocabulary items).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.occ_locks import lock_includes
from src.domain.item_geometry import stored_box
from src.services.curation.proposal_merge import ExistingItem, Proposal, merge_item_proposals
from src.services.curation.reprocess_locks import item_locked
from src.services.curation.reprocess_targets import items_by_terms


if TYPE_CHECKING:
    from collections.abc import Sequence

MGET_PAGE = 1000


@dataclass(frozen=True)
class RepairPlan:
    to_create: list[dict[str, Any]]
    """The item documents no existing item stands for."""
    existing_ids: frozenset[str]
    """Every item the image has now (listed or not)."""


def _includes() -> list[str]:
    return ['image_id', 'bbox_norm', 'class_name', 'detector_bbox_norm', *lock_includes()]


async def _existing_items(
    client: Any, items_index: str, image_id: str, listed: Sequence[str]
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(listed), MGET_PAGE):
        resp = await client.mget(
            index=items_index,
            body={'ids': list(listed[start : start + MGET_PAGE])},
            _source_includes=_includes(),
        )
        for doc in resp.get('docs') or []:
            if doc.get('found'):
                found[doc['_id']] = doc.get('_source') or {}
    # A search sees only what a refresh has published (about a second after a write): an item a
    # person drew, or an import or an open-vocabulary run wrote, a moment ago would be missed and
    # its object created again. A repair is the rare path, so it publishes the index first
    # (https://docs.opensearch.org/latest/api-reference/index-apis/refresh/).
    await client.indices.refresh(index=items_index)
    for crop_id, source in await items_by_terms(
        client, 'image_id', [image_id], index=items_index, includes=_includes()
    ):
        if source.get('image_id') == image_id:
            found.setdefault(crop_id, source)
    return found


def _proposal(doc: dict[str, Any]) -> Proposal:
    box = doc['bbox_norm']
    return Proposal(
        bbox_norm=(box[0], box[1], box[2], box[3]),
        class_id=doc.get('class_id'),
        class_name=doc.get('class_name') or doc.get('proposal_name'),
        score=float(doc.get('detector_confidence') or 0.0),
    )


async def plan_repair(
    client: Any,
    items_index: str,
    image_id: str,
    listed: Sequence[str],
    crop_docs: list[dict[str, Any]],
) -> RepairPlan:
    """The documents of ``crop_docs`` to create for ``image_id``: those no existing item matches."""
    found = await _existing_items(client, items_index, image_id, listed)
    existing = [
        ExistingItem(
            crop_id=crop_id,
            bbox_norm=box,
            class_name=source.get('class_name'),
            locked=item_locked(source),
            detector_bbox_norm=stored_box(source.get('detector_bbox_norm')),
        )
        for crop_id, source in found.items()
        if (box := stored_box(source.get('bbox_norm'))) is not None
    ]
    proposals = [_proposal(doc) for doc in crop_docs]
    created = {id(p) for p in merge_item_proposals(existing, proposals).created}
    return RepairPlan(
        [doc for doc, p in zip(crop_docs, proposals, strict=True) if id(p) in created],
        frozenset(found),
    )


__all__ = ['RepairPlan', 'plan_repair']
