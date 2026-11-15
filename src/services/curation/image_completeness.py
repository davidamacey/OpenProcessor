"""The one place that checks whether a stored photo is whole, and the edits of its ``item_ids``.

The rule is :mod:`src.domain.image_completeness`: the images doc lists its items, and a photo is
whole when every listed item exists. This module reads the items (:func:`missing_items`) and
decides ``complete`` / ``repair`` / ``in_flight`` from them (:func:`classify`, used by the content
lookup, the writer's 409 answer and the claim-time holder answer, so one rule decides a
``duplicate``), and makes the edits of the doc, each an OCC write through
:func:`~src.clients.occ_bulk.occ_update_bulk` (read, merge on the fresh source, conditional write
with ``if_seq_no`` / ``if_primary_term``, a 409 re-read):

* :func:`forget_items` takes the ids of items about to be deleted off their images' lists, so a
  deleted item never makes its image look incomplete;
* :func:`take_repair_lease` takes the repair lease before the repair reads anything, and
  :func:`set_repair_list` sets the list under that lease, BEFORE the repair creates anything: the
  list then names the items about to exist (the same order as an ingest), so a delete that races
  the repair takes its id off the list and nothing relists it;
* :func:`finish_repair` gives the lease back and clears the mark (best effort: the list is
  already true);
* :func:`mark_incomplete` is the writer's best-effort note on a photo it saw fail, which also
  gives its lease back.

OpenSearch documents used: multi-get (``_source=false`` and ``_source_includes``; a get is real
time, so an item written but not refreshed is found:
https://docs.opensearch.org/latest/api-reference/document-apis/multi-get/ and
https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/), the bulk API
(``if_seq_no`` / ``if_primary_term`` on an action:
https://docs.opensearch.org/latest/api-reference/document-apis/bulk/) and the update API
(a partial ``doc`` is merged into the stored one:
https://docs.opensearch.org/latest/api-reference/document-apis/update-document/).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from opensearchpy.exceptions import OpenSearchException
from sqlalchemy.exc import SQLAlchemyError

from src.clients.occ_bulk import occ_update_bulk
from src.core.logging import get_logger
from src.domain.image_completeness import (
    INCOMPLETE_MARK_FIELD,
    ITEM_IDS_FIELD,
    REPAIR_LEASE_FIELD,
    STATE_FIELDS,
    listed_item_ids,
    repair_in_progress,
    repairable,
)
from src.services.curation.ops_metrics import count_whole_event
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence


logger = get_logger(__name__)

MGET_PAGE = 1000
# 409 retry rounds of the writes a writer makes while it is the live writer of a photo; the
# repair age (``ingest_writer.REPAIR_AFTER_S``) counts their requests.
LIST_WRITE_RETRIES = 2
MARK_WRITE_RETRIES = 1

Completeness = Literal['complete', 'repair', 'in_flight']


async def missing_items(
    client: Any, items_index: str, listed: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """``{image_id: its listed items that do not exist}`` for the images that miss any.

    One real-time ``mget`` per page of :data:`MGET_PAGE` ids over the items of every image of
    ``listed``, with the source switched off."""
    wanted = sorted({item for ids in listed.values() for item in ids})
    found: set[str] = set()
    for start in range(0, len(wanted), MGET_PAGE):
        resp = await client.mget(
            index=items_index, body={'ids': wanted[start : start + MGET_PAGE]}, _source=False
        )
        found.update(d['_id'] for d in resp.get('docs') or [] if d.get('found'))
    gaps = {image_id: [i for i in ids if i not in found] for image_id, ids in listed.items()}
    return {image_id: gap for image_id, gap in gaps.items() if gap}


async def forget_items(
    client: Any, images_index: str, by_image: Mapping[str, Collection[str]]
) -> None:
    """Take ``by_image`` ``{image_id: item ids}`` off the lists of the images docs, in one bulk
    OCC update for all images; call it BEFORE the items are deleted (a crash in between leaves
    an item the list no longer names, which keeps the list true). An image without the field
    (legacy) or whose list names none of the ids is not written. Raises when a doc could not be
    updated: the delete must not go on."""
    status = await occ_update_bulk(
        client,
        index=images_index,
        ids=list(by_image),
        merge_fn=lambda image_id, source: _without(source, by_image[image_id]),
        vector_excludes=await vector_source_excludes('images'),
    )
    failed = {i: s for i, s in status.items() if s not in ('updated', 'not-found')}
    if failed:
        msg = (
            f'item_ids of {len(failed)} image(s) could not be updated: {sorted(failed.items())[:3]}'
        )
        raise OpenSearchException(msg)


def _without(source: dict[str, Any], gone: Collection[str]) -> dict[str, Any]:
    listed = listed_item_ids(source)
    if listed is None or not gone or not set(listed) & set(gone):
        return {}
    return {ITEM_IDS_FIELD: [i for i in listed if i not in set(gone)]}


async def classify(
    client: Any,
    items_index: str,
    docs: Mapping[str, dict[str, Any]],
    *,
    repair_after_s: float,
) -> dict[str, Completeness]:
    """``{image_id: complete | repair | in_flight}`` for the images docs of ``docs`` (``_source`` of
    each, with :data:`~src.domain.image_completeness.STATE_FIELDS`). One real-time ``mget`` over
    the listed items of all of them. A doc without a list, or with an empty one, is complete."""
    listed = {i: ids for i, src in docs.items() if (ids := listed_item_ids(src))}
    gaps = await missing_items(client, items_index, listed) if listed else {}
    now = datetime.now(UTC)
    out: dict[str, Completeness] = {}
    for image_id, source in docs.items():
        if image_id not in gaps:
            out[image_id] = 'complete'
        elif repairable(source, now=now, repair_after_s=repair_after_s):
            out[image_id] = 'repair'
        else:
            out[image_id] = 'in_flight'
    return out


async def holders_completeness(
    client: Any,
    images_index: str,
    items_index: str,
    holder_ids: Sequence[str],
    *,
    repair_after_s: float,
) -> dict[str, Completeness]:
    """:func:`classify` for images read by id (the holder a content claim found). A holder whose
    doc is not there yet is ``in_flight``: its writer has claimed and not written."""
    resp = await client.mget(
        index=images_index, body={'ids': list(holder_ids)}, _source_includes=list(STATE_FIELDS)
    )
    docs = {d['_id']: d.get('_source') or {} for d in resp.get('docs') or [] if d.get('found')}
    states = await classify(client, items_index, docs, repair_after_s=repair_after_s)
    return {i: states.get(i, 'in_flight') for i in holder_ids}


@dataclass(frozen=True)
class RepairLease:
    """A repair's hold on an image: the token (the start time) and the items the doc listed at the
    moment the lease was taken, which is where the repair starts reading."""

    token: str
    listed: tuple[str, ...]


async def take_repair_lease(
    client: Any, images_index: str, image_id: str, *, repair_after_s: float
) -> RepairLease | None:
    """Take the repair lease, in one OCC write; ``None`` when another repair holds it. Raises when
    the doc is gone or the write did not land.

    The lease comes FIRST: everything the repair reads afterwards (the doc's list, the items) is
    read while no other repair can change them, so a repair that waited behind another sees what
    that one created. A repair that read first and took the lease after would plan from a state
    that is gone and create a second set of items for the same objects."""
    token = datetime.now(UTC).isoformat()
    held = False
    listed: tuple[str, ...] = ()

    def merge(_id: str, source: dict[str, Any]) -> dict[str, Any]:
        nonlocal held, listed
        held = repair_in_progress(source, now=datetime.now(UTC), repair_after_s=repair_after_s)
        if held:
            return {}
        listed = tuple(listed_item_ids(source) or ())
        return {REPAIR_LEASE_FIELD: token}

    await _write_under_lease(client, images_index, image_id, merge)
    return None if held else RepairLease(token, listed)


async def set_repair_list(
    client: Any,
    images_index: str,
    image_id: str,
    *,
    token: str,
    existing_ids: Collection[str],
    planned_ids: Collection[str],
) -> bool:
    """Set the list, in one OCC write, while the doc still holds the lease ``token``; False when it
    does not (the lease ran out and another repair took it). Call it BEFORE the repair creates
    anything.

    The list becomes the items the doc names that still exist (``existing_ids`` is what the repair
    just read, and the merge runs on the fresh source, so an id a delete took off the list since
    stays off) plus the items the repair is about to create (``planned_ids``). Raises when the doc
    is gone or the write did not land."""
    ours = True

    def merge(_id: str, source: dict[str, Any]) -> dict[str, Any]:
        nonlocal ours
        ours = source.get(REPAIR_LEASE_FIELD) == token
        if not ours:
            return {}
        listed = set(listed_item_ids(source) or ()) & set(existing_ids)
        return {ITEM_IDS_FIELD: sorted(listed | set(planned_ids))}

    await _write_under_lease(client, images_index, image_id, merge)
    return ours


async def _write_under_lease(
    client: Any,
    images_index: str,
    image_id: str,
    merge: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    status = await occ_update_bulk(
        client,
        index=images_index,
        ids=[image_id],
        merge_fn=merge,
        vector_excludes=await vector_source_excludes('images'),
        max_retries=LIST_WRITE_RETRIES,
    )
    if status.get(image_id) != 'updated':
        msg = f'the repair of image {image_id} could not be written: {status.get(image_id)}'
        raise OpenSearchException(msg)


async def _release(
    client: Any, images_index: str, image_id: str, token: str | None, *, mark: bool
) -> bool:
    """One OCC write that gives back the lease ``token`` (only while the doc still holds it) and
    sets (``mark``) or clears the failure mark. True when the write landed. Never raises."""

    def merge(_id: str, source: dict[str, Any]) -> dict[str, Any]:
        update: dict[str, Any] = {}
        if token is not None and source.get(REPAIR_LEASE_FIELD) == token:
            update[REPAIR_LEASE_FIELD] = None
        if bool(source.get(INCOMPLETE_MARK_FIELD)) != mark and (
            mark or INCOMPLETE_MARK_FIELD in source
        ):
            update[INCOMPLETE_MARK_FIELD] = mark
        return update

    try:
        status = await occ_update_bulk(
            client,
            index=images_index,
            ids=[image_id],
            merge_fn=merge,
            vector_excludes=await vector_source_excludes('images'),
            max_retries=MARK_WRITE_RETRIES,
        )
    except (OpenSearchException, SQLAlchemyError) as exc:
        logger.warning('ingest_repair_note_failed', image_id=image_id, error=str(exc))
        return False
    return status.get(image_id) == 'updated'


async def mark_incomplete(
    client: Any, images_index: str, image_id: str, lease: str | None = None
) -> None:
    """Note on the images doc that its own writer saw the write fail, so the next ingest of the
    same bytes repairs it at once instead of waiting out the repair age; a repair gives its
    ``lease`` back in the same write. Best effort: a write that fails is logged and counted, never
    raised (the age rule covers it)."""
    if await _release(client, images_index, image_id, lease, mark=True):
        count_whole_event('marked')
    else:
        count_whole_event('mark_failed')


async def finish_repair(client: Any, images_index: str, image_id: str, lease: str) -> None:
    """The repair created what it planned: give the lease back and clear the failure mark.
    Best effort: the list was set when the repair began, so the photo is whole either way, and a
    lease left behind runs out."""
    if not await _release(client, images_index, image_id, lease, mark=False):
        count_whole_event('mark_failed')


__all__ = [
    'Completeness',
    'RepairLease',
    'classify',
    'finish_repair',
    'forget_items',
    'holders_completeness',
    'mark_incomplete',
    'missing_items',
    'set_repair_list',
    'take_repair_lease',
]
