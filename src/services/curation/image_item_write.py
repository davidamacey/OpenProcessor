"""A person draws a box on an image: the box becomes an item.

The item is created in one ``op_type=create`` write under an id made from the request id, so a
retry of the same request meets the create conflict, reads the stored item back and returns it
(no table, no lock; OpenSearch decides the race). The class is resolved by name, the label is the
human label every class writer uses (:class:`~src.services.curation.class_label.ItemLabel`), and
the new item inherits the image's hold-out so a new object on a test image never trains.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import ConflictError, NotFoundError

from src.config import get_curation_config
from src.core.exceptions import RequestRejectedError
from src.domain.embedding_state import DEFERRED
from src.domain.item_geometry import GEOMETRY_SOURCE_HUMAN, box_area_norm, drawn_crop_id
from src.schemas.curation.config_common import error_body
from src.services.classes.resolve import class_id_for_name, class_name_for_id
from src.services.curation.class_label import ItemLabel
from src.services.curation.edit_history import GEOMETRY_WRITER
from src.services.curation.image_counts_refresh import enqueue_image_counts
from src.services.curation.image_detail import image_source
from src.services.curation.item_doc import DetectedItem, build_item_doc, region_seed_status
from src.services.curation.item_embed_inline import EmbedOutcome, embed_item_inline
from src.services.curation.item_geometry import checked_box, rerank_image_items
from src.services.curation.reprocess_targets import items_by_terms
from src.services.curation.wire import serialize_item
from src.services.embedding_spaces.readers import (
    item_state_fields,
    item_write_fields,
    vector_source_excludes,
)
from src.services.near_duplicates import pending as near_dup_pending
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

IMAGE_FIELDS = ['image_path', 'width', 'height', 'source', 'dataset_split']


@dataclass(frozen=True)
class DrawnItem:
    source: dict[str, Any]
    """The stored item (no vectors)."""
    created: bool
    embed: EmbedOutcome


def _refuse(status: int, code: Any, message: str, **fields: Any) -> RequestRejectedError:
    return RequestRejectedError(status, error_body(code, message, **fields))


async def _active_class(class_name: str) -> tuple[int, str]:
    class_id = await class_id_for_name(class_name)
    name = None if class_id is None else await class_name_for_id(class_id)
    if class_id is None or name is None:
        raise _refuse(
            422,
            'unknown_class',
            f'there is no active class called "{class_name}"',
            class_name=class_name,
        )
    return class_id, name


def _inherited_split(
    image: dict[str, Any], siblings: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """The hold-out the image's items are in (D-11): a new item there is held out too."""
    held_out = any(src.get('test_holdout') for _cid, src in siblings)
    split = image.get('dataset_split') or next(
        (src['dataset_split'] for _cid, src in siblings if src.get('dataset_split')), None
    )
    fields: dict[str, Any] = {}
    if held_out:
        fields['test_holdout'] = True
    if split:
        fields['dataset_split'] = split
    return fields


async def _stored(
    opensearch: AsyncOpenSearch, crop_id: str, excludes: list[str]
) -> dict[str, Any] | None:
    try:
        resp = await opensearch.get(
            index=get_curation_config().items_index, id=crop_id, _source_excludes=excludes
        )
    except NotFoundError:
        return None
    return resp.get('_source') or None


async def _repeat(
    stored: dict[str, Any], crop_id: str, box: Sequence[float], class_name: str
) -> dict[str, Any]:
    """``stored`` when it is what this request asks for; 409 ``request_id_reused`` (with the
    stored item) when the same request id came with another box or class. The stored item is
    compared as it is now: a retry after a person edited it is told so rather than answered
    with a box the editor did not ask for."""
    same_box = (
        len(stored.get('bbox_norm') or []) == 4
        and max(abs(a - b) for a, b in zip(stored['bbox_norm'], box, strict=True)) < 1e-9
    )
    same_class = str(stored.get('class_name') or '').casefold() == class_name.strip().casefold()
    if same_box and same_class:
        return stored
    raise _refuse(
        409,
        'request_id_reused',
        'this request_id already created an item with another box or class; use a new request_id',
        item=serialize_item(stored, crop_id, state_fields=await item_state_fields()),
    )


async def create_drawn_item(
    opensearch: AsyncOpenSearch,
    pe: Any | None,
    *,
    image_id: str,
    bbox: Sequence[float],
    class_name: str,
    request_id: uuid.UUID,
) -> DrawnItem:
    """Create the item, or return the one this ``request_id`` already created.

    A repeat is answered from the stored item before the class and image are checked again, so a
    retry still succeeds after the class was deprecated; with another box or class it is 409
    ``request_id_reused``. Raises :class:`~src.core.exceptions.RequestRejectedError`: 422 for a
    refused box or ``unknown_class``, 404 ``image_not_found``, 409 ``request_id_reused``."""
    box = checked_box(bbox)
    cfg = get_curation_config()
    crop_id = drawn_crop_id(image_id, request_id)
    excludes = await vector_source_excludes()
    write = await item_write_fields()
    if (repeat := await _stored(opensearch, crop_id, excludes)) is not None:
        repeat = await _repeat(repeat, crop_id, box, class_name)
        return DrawnItem(repeat, False, EmbedOutcome(repeat.get(write.state_field, DEFERRED), None))
    class_id, canonical = await _active_class(class_name)
    image = await image_source(opensearch, image_id, IMAGE_FIELDS)
    if image is None:
        raise _refuse(404, 'image_not_found', f'image not found: {image_id}')
    siblings = await items_by_terms(
        opensearch,
        'image_id',
        [image_id],
        index=cfg.items_index,
        includes=['test_holdout', 'dataset_split'],
    )
    width, height = int(image.get('width') or 1), int(image.get('height') or 1)
    item = DetectedItem(
        bbox_pixel=(box[0] * width, box[1] * height, box[2] * width, box[3] * height),
        score=1.0,
        class_id=class_id,
        class_name=canonical,
        label=ItemLabel.human(
            class_id=class_id,
            class_name=canonical,
            label_source='human',
            writer=GEOMETRY_WRITER,
        ),
        embedding_state=DEFERRED,
        extra_fields={
            'geometry_source': GEOMETRY_SOURCE_HUMAN,
            'geometry_revision': 1,
            **_inherited_split(image, siblings),
        },
    )
    doc = build_item_doc(
        crop_id=crop_id,
        image_id=image_id,
        image_path=str(image.get('image_path') or ''),
        source=str(image.get('source') or ''),
        request_id=str(request_id),
        bbox_norm=list(box),
        item=item,
        now=now_iso(),
        crop_area_norm=box_area_norm(box),
        crop_rank_in_image=len(siblings) + 1,
        vector_field=write.vector_field,
        state_field=write.state_field,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
        region_status=region_seed_status(item),
    )
    created = True
    try:
        await opensearch.index(
            index=cfg.items_index, id=crop_id, body=doc, op_type='create', refresh='wait_for'
        )
    except ConflictError:
        created = False
    if not created:  # a parallel request with the same id won the create
        raced = await _repeat(
            await _stored(opensearch, crop_id, excludes) or {}, crop_id, box, class_name
        )
        return DrawnItem(raced, False, EmbedOutcome(raced.get(write.state_field, DEFERRED), None))
    await enqueue_image_counts(opensearch, [image_id])
    await rerank_image_items(opensearch, image_id)
    await near_dup_pending.enqueue_pending_bound('item', [crop_id], 'added')
    embed = await embed_item_inline(opensearch, pe, crop_id)
    return DrawnItem(await _stored(opensearch, crop_id, excludes) or {}, True, embed)
