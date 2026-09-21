"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import contextlib
import os
from typing import TYPE_CHECKING, Any

import httpx

from src.clients.occ import (
    is_human_owned_class,
    occ_skip_on_conflict_bulk,
    strip_class_write_fields,
)
from src.config import get_region_fields
from src.core.logging import get_logger
from src.services.curation.history import (
    MAX_PLATE_CHAIN_ENTRIES,
    append_plate_chain_entry,
    record_class_history,
)


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

    A-PR4: the merger uses :func:`append_plate_chain_entry` to extend
    the existing ``RegionFields.detector_chain`` rather than
    overwriting it, so concurrent chain mutations are preserved.

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
        update = dict(task.update_doc)
        # P0-2 defense-in-depth: runner.py's _should_classify already
        # prevents class fields from ever landing in task.update_doc for
        # a human-owned crop, so this should be a no-op in practice —
        # kept as a second layer against a future code path that
        # forgets to consult that gate. Scoped to class fields only
        # (strip_class_write_fields) so a region write in the SAME
        # update_doc (the combined class+region write) is never
        # dropped.
        if is_human_owned_class(current):
            update = strip_class_write_fields(update)
        # Phase 3 (b): the worker's combined-VLM path writes class_id
        # without appending class_id_history unless we do it here — the
        # reset happens in the update dict itself (verify.py's
        # ``_combined_class_update``), the history snapshot happens
        # here where the pre-write ``current`` doc is available.
        if 'class_id' in update:
            update['class_id_history'] = record_class_history(current, writer='sam_worker')
        # The worker accumulates an in-iteration detection_trace; merge
        # those new entries on top of any existing chain in OS (which
        # may have been appended to by a human or a prior worker pass)
        # so we never clobber prior provenance.
        existing_chain = current.get(F.detector_chain) or []
        if task.detection_trace:
            existing = list(existing_chain)
            existing_set = set(existing)
            chain = list(existing)
            for entry in task.detection_trace:
                if entry in existing_set:
                    continue
                # Entry shape: "<detector>:<tag>" or
                # "<detector>:<sub>:<tag>". Pass through
                # append_plate_chain_entry so the cap + drop-oldest
                # policy applies uniformly.
                if ':' in entry:
                    detector, _, outcome = entry.partition(':')
                else:
                    detector, outcome = entry, ''
                chain = append_plate_chain_entry(
                    chain,
                    detector=detector,
                    detector_version='',
                    outcome=outcome,
                )
                existing_set.add(entry)
            update[F.detector_chain] = chain
        elif F.detector_chain in update:
            # Worker built a chain in-task (e.g. final no_region_box).
            # Union with the existing OS chain to preserve concurrent
            # writes.
            new_entries = list(update[F.detector_chain])
            seen = set(existing_chain)
            merged = list(existing_chain)
            for entry in new_entries:
                if entry not in seen:
                    merged.append(entry)
                    seen.add(entry)
            if len(merged) > MAX_PLATE_CHAIN_ENTRIES:
                merged = merged[-MAX_PLATE_CHAIN_ENTRIES:]
            update[F.detector_chain] = merged
        return update

    result = await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=[t.crop_id for t in eligible],
        merger=_merge,
        index=CURATION_ITEMS_INDEX,
        refresh=False,
        writer_id='sam_worker',
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
    url = f'{_EVENT_API_URL}/curation/events/publish'
    for t in written:
        plate_status = (t.update_doc or {}).get(F.status)
        if not plate_status:
            continue
        body = {
            'type': 'crop.region_verified',
            'topic': 'region_status',
            'crop_id': t.crop_id,
            F.status: plate_status,
        }
        with contextlib.suppress(Exception):  # nosec B110 — advisory; never fail the worker
            await _EVENT_CLIENT.post(url, json=body, timeout=2.0)


# =============================================================================
# Sentinel + main loop
# =============================================================================
