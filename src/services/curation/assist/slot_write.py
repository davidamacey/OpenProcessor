"""The ``assist`` slot of an item: the VLM's suggestions from the newest run.

One slot per item. A run replaces every key of the slot except ``boxes`` (box
attributes belong to box runs and survive an item run). The write is an OCC
merge: it skips an item whose stored slot is from a newer run, a frozen holdout
item, and, as for every worker write, an item a human changed since the read.
It never touches class or box fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config.curation_accessors import items_index


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.labeling.vlm_models import AttributeValue


@dataclass(frozen=True)
class RunMeta:
    """Who produced a slot."""

    run_id: str
    created_at: str
    vlm_endpoint: str
    vlm_model: str
    prompt_pack: str
    prompt_pack_revision: int | None


def build_slot(
    meta: RunMeta,
    *,
    class_name: str | None,
    class_confidence: str | None,
    attributes: dict[str, AttributeValue],
) -> dict[str, Any]:
    """The slot for one item. Every key is present (``None`` when the run did not
    produce it) so a partial update replaces what an older run wrote."""
    return {
        'run_id': meta.run_id,
        'run_created_at': meta.created_at,
        'vlm_endpoint': meta.vlm_endpoint,
        'vlm_model': meta.vlm_model,
        'prompt_pack': meta.prompt_pack,
        'prompt_pack_revision': meta.prompt_pack_revision,
        'class_name': class_name or None,
        'class_confidence': class_confidence if class_name else None,
        'attributes': [
            {'key': key, 'value': a.value, 'confidence': a.confidence}
            for key, a in attributes.items()
        ],
    }


def _older_than(stored: Any, created_at: str) -> bool:
    """Whether the stored slot predates this run (or is absent / unreadable)."""
    try:
        return datetime.fromisoformat(str(stored)) <= datetime.fromisoformat(created_at)
    except ValueError:
        return True


async def write_slots(opensearch: AsyncOpenSearch, slots: dict[str, dict[str, Any]]) -> int:
    """Write ``{crop_id: slot}``; the number of items written."""
    if not slots:
        return 0

    def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        if current.get('test_holdout'):
            return {}
        slot = slots[doc_id]
        stored = current.get('assist') or {}
        if stored.get('run_created_at') and not _older_than(
            stored['run_created_at'], slot['run_created_at']
        ):
            return {}
        if stored.get('boxes'):
            slot = {**slot, 'boxes': stored['boxes']}
        return {'assist': slot}

    result = await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=list(slots),
        merger=merge,
        index=items_index(),
        refresh=False,
        writer_id='assist_run',
    )
    return int(result.get('updated', 0))


async def write_box_attributes(
    opensearch: AsyncOpenSearch,
    meta: RunMeta,
    by_crop: dict[str, dict[str, list[dict[str, str]]]],
) -> int:
    """Write ``{crop_id: {box_id: attributes}}`` into ``assist.boxes[]``. An item that
    has no slot gets one carrying only the run; an existing slot keeps its item-level
    suggestions. Entries for other boxes are kept; the named boxes are replaced."""
    if not by_crop:
        return 0

    def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        if current.get('test_holdout'):
            return {}
        stored = dict(current.get('assist') or {})
        if not stored:
            stored = build_slot(meta, class_name=None, class_confidence=None, attributes={})
        named = by_crop[doc_id]
        kept = [b for b in stored.get('boxes') or [] if b.get('box_id') not in named]
        stored['boxes'] = [
            *kept,
            *({'box_id': box_id, 'attributes': attrs} for box_id, attrs in named.items()),
        ]
        return {'assist': stored}

    result = await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=list(by_crop),
        merger=merge,
        index=items_index(),
        refresh=False,
        writer_id='assist_run',
    )
    return int(result.get('updated', 0))


__all__ = ['RunMeta', 'build_slot', 'write_box_attributes', 'write_slots']
