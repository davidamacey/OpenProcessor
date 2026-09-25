"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import contextlib
import os
from typing import TYPE_CHECKING, Any

import httpx

from src.clients.occ import CLASS_WRITE_FIELDS, occ_skip_on_conflict_bulk, strip_class_write_fields
from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger
from src.services.curation.class_write_guard import class_write_allowed
from src.services.curation.history import merge_region_chain, record_class_snapshot
from src.services.curation.wire import region_event_payload


logger = get_logger('curation_worker')


from scripts.curation.worker.state import CURATION_ITEMS_INDEX, _ItemTask


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def _bulk_update(opensearch: AsyncOpenSearch, tasks: list[_ItemTask]) -> tuple[int, int]:
    """Apply each task's ``update_doc`` to OpenSearch with OCC semantics.

    A-PR3: per-doc OCC via :func:`occ_skip_on_conflict_bulk` — on a
    seq_no conflict (concurrent human region edit), the worker skips
    the doc and lets the human win. The next polling iteration will see
    the updated state and re-decide.

    A-PR4: the merger uses :func:`merge_region_chain` to extend the
    existing ``RegionFields.detector_chain`` rather than overwriting it,
    so concurrent chain mutations are preserved.

    A write whose doc's stored region status no longer equals the status
    the task was fetched with is dropped (stale — see ``_merge``).

    Returns ``(n_written, n_skipped)`` where ``n_skipped`` counts both
    tasks with empty ``update_doc`` and tasks that lost an OCC race.
    """
    F = get_region_fields()
    eligible: list[_ItemTask] = []
    n_skipped_empty = 0
    by_id: dict[str, _ItemTask] = {}
    for t in tasks:
        if not t.update_doc:
            n_skipped_empty += 1
            continue
        eligible.append(t)
        by_id[t.crop_id] = t
    if not eligible:
        return 0, n_skipped_empty

    def _merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        task = by_id[doc_id]
        # Idempotency backstop: the result only applies to the pending
        # state it was computed from. If the live (realtime ``_mget``) doc
        # has moved on — an earlier pass or a duplicate consumer already
        # wrote it, or a human changed it — drop this write instead of
        # re-stamping the region and re-appending the chain.
        if current.get(F.status) != task.region_status:
            logger.info(
                'region_write_stale_skip',
                crop_id=doc_id,
                fetched_status=task.region_status,
                current_status=current.get(F.status),
            )
            return {}
        update = dict(task.update_doc)
        # Item text read this pass rides on the region write; it is not
        # class data, so the human-label guard below leaves it alone.
        update.update(task.item_text_update)
        # Class fields land only on the exact class state this task was
        # read in, never on a human-owned/validated class: a human write
        # (an undo, a relabel) during this pass wins. runner.py's
        # _should_classify gates human-owned crops at read time; this is
        # the write-time half. Scoped to class fields only
        # (strip_class_write_fields) so the region write in the SAME
        # update_doc (the combined class+region write) still lands.
        if CLASS_WRITE_FIELDS & update.keys() and not class_write_allowed(
            task.class_token, current
        ):
            logger.info('class_write_stale_skip', doc_id=doc_id, writer_id='region_worker')
            update = strip_class_write_fields(update)
        # Phase 3 (b): the worker's combined-VLM path changes the class
        # without appending class_id_history unless we do it here — the
        # reset happens in the update dict itself (verify.py's
        # ``_combined_class_update``), the history snapshot happens
        # here where the pre-write ``current`` doc is available. A full,
        # restorable snapshot (also for a proposal with no class_id yet
        # and a class_source-only ``vlm_unmatched`` write).
        if 'class_source' in update:
            update['class_id_history'] = record_class_snapshot(
                current, writer='region_worker', restorable=True
            )
        # Merge this pass's entries onto whatever chain is stored (a human
        # or an earlier pass may have appended) — ordered, de-duplicated,
        # normalized to ``<actor>:<event>``, capped.
        new_entries = list(task.detection_trace) or list(update.get(F.detector_chain) or [])
        if new_entries:
            update[F.detector_chain] = merge_region_chain(
                current.get(F.detector_chain), new_entries
            )
        return update

    result = await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=[t.crop_id for t in eligible],
        merger=_merge,
        index=CURATION_ITEMS_INDEX,
        # The runner releases a crop from its in-flight set once this
        # returns; ``wait_for`` makes the write visible to the next
        # pending search first, so a refresh-lagged search can't hand the
        # same crop out again.
        refresh='wait_for',
        writer_id='region_worker',
    )
    n_written = int(result.get('updated', 0))
    n_skipped_conflict = int(result.get('skipped_due_to_conflict', 0))
    if result.get('errors'):
        logger.warning(
            'bulk_partial_errors',
            items=n_written,
            errors=len(result['errors']),
        )
    # Only publish events for tasks that actually wrote. ``occ_skip…``
    # doesn't return per-id results, so use the conservative
    # approximation: if nothing was skipped we wrote everything; if
    # there were conflicts, we still publish optimistically for the
    # cases that did write (advisory events).
    if n_written:
        await _publish_region_events(eligible)
    return n_written, n_skipped_empty + n_skipped_conflict


# Task #92 — module-level lazy client for the publish endpoint. Reused
# across calls so we don't tear down the connection pool every batch.
_EVENT_API_URL = os.environ.get('OP_EVENT_API_URL', '').rstrip('/')
_EVENT_CLIENT: httpx.AsyncClient | None = None


async def _publish_region_events(written: list[_ItemTask]) -> None:
    """Fire one ``crop.region_verified`` event per written crop.

    Disabled when ``OP_EVENT_API_URL`` is unset (e.g. test harness or
    pre-task-92 deployments). Never raises.
    """
    if not _EVENT_API_URL or not written:
        return
    global _EVENT_CLIENT  # noqa: PLW0603 — module-level lazy client
    if _EVENT_CLIENT is None:
        _EVENT_CLIENT = httpx.AsyncClient(timeout=2.0)
    F = get_region_fields()
    url = f'{_EVENT_API_URL}{get_curation_config().api_prefix}/events/publish'
    for t in written:
        region_status = (t.update_doc or {}).get(F.status)
        if not region_status:
            continue
        # Read under the storage name, publish under the fixed wire name
        # (_PublishEvent.region_status) — posting the storage key made the
        # API drop the status whenever the two differed.
        body = region_event_payload(
            t.crop_id, region_status=region_status, region_text=(t.update_doc or {}).get(F.text)
        )
        with contextlib.suppress(Exception):  # nosec B110 — advisory; never fail the worker
            await _EVENT_CLIENT.post(url, json=body, timeout=2.0)


# =============================================================================
# Sentinel + main loop
# =============================================================================
