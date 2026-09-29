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
from src.config import get_region_fields
from src.config.project_context import project_api_base
from src.core.logging import get_logger
from src.services.curation.class_sources import VLM_UNMATCHED_CLASS_SOURCE, unmatched_class_clear
from src.services.curation.class_write_guard import class_write_allowed
from src.services.curation.history import merge_region_chain, record_class_snapshot
from src.services.curation.region_boxes import (
    boxes_write_fields,
    derive_status,
    finalize_box_ids,
    is_human_owned,
    merge_boxes_for_write,
    read_boxes,
)
from src.services.curation.wire import region_event_payload


logger = get_logger('curation_worker')


from scripts.curation.worker.state import _ItemTask, items_index


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


def _current_config_stamp() -> tuple[str | None, int | None, str | None]:
    """``(profile_name, profile_revision, pack_stamp)`` for whatever
    project is currently bound -- called from inside ``_bulk_update``'s
    ``with bind_project(...):`` block, so this resolves the *task's own*
    project's config store (W2 sec 4.5 "item stamping"), never a
    process-wide default. ``profile_revision`` is ``None`` for an
    env/file-registered profile never activated through the store."""
    from src.services.config_store import get_config_store
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    if profile is None:
        return None, None, None
    profile_revision: int | None = None
    try:
        ref = get_config_store().current.active_profile
    except Exception:  # pragma: no cover - config_store always importable
        ref = None
    if isinstance(ref, tuple) and ref[0] == profile.name:
        profile_revision = ref[1]

    pack_stamp: str | None = None
    try:
        from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

        pack_stamp = prompt_pack_stamp(active_prompt_pack())
    except Exception as exc:  # pragma: no cover - defensive, never blocks a region write
        logger.warning('vlm_prompt_pack_stamp_failed', error=str(exc))

    return profile.name, profile_revision, pack_stamp


async def _bulk_update(opensearch: AsyncOpenSearch, tasks: list[_ItemTask]) -> tuple[int, int]:
    """Group ``tasks`` by their own project and flush one ``_bulk`` call
    per project, each issued while bound to that project (projects_plan.md
    §5.1) -- the guard rejects a call issued against project A's index
    while project B is bound, so writes for different projects can never
    share one bulk body. A task without a project is a producer bug and
    raises instead of being written unbound.
    """
    from src.config.project_context import bind_project

    if not tasks:
        return 0, 0
    by_project: dict[str, list[_ItemTask]] = {}
    projects_by_slug: dict[str, Any] = {}
    for t in tasks:
        if t.project is None:
            msg = f'item task {t.crop_id!r} has no project'
            raise ValueError(msg)
        by_project.setdefault(t.project.slug, []).append(t)
        projects_by_slug[t.project.slug] = t.project

    n_written = 0
    n_skipped = 0
    for slug, group in by_project.items():
        with bind_project(projects_by_slug[slug]):
            w, s = await _bulk_update_one_project(opensearch, group)
        n_written += w
        n_skipped += s
    return n_written, n_skipped


async def _bulk_update_one_project(
    opensearch: AsyncOpenSearch, tasks: list[_ItemTask]
) -> tuple[int, int]:
    """The original single-``_bulk``-call body of ``_bulk_update``,
    scoped to tasks that all belong to one (already-bound) project.

    A-PR3: per-doc OCC via :func:`occ_skip_on_conflict_bulk` — on a

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
    profile_name, profile_revision, pack_stamp = _current_config_stamp()
    eligible: list[_ItemTask] = []
    n_skipped_empty = 0
    by_id: dict[str, _ItemTask] = {}
    for t in tasks:
        # `t.pending_boxes is not None` is also a real write even when
        # `t.update_doc` itself is empty -- the box-list fields (status,
        # region_boxes, revision, ...) are computed below, inside
        # `_merge`, against the live doc (W8 B1/M1), not stashed onto
        # `update_doc` at task-processing time.
        if not t.update_doc and t.pending_boxes is None:
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
        # W8 B1 + M1 fix: the box list write is finished HERE, against
        # the live `current` doc this closure was just handed (re-read
        # immediately before the write by `occ_skip_on_conflict_bulk`) --
        # never against `task`'s own (possibly stale) fetch-time
        # snapshot. This is what actually fixes both bugs: B1 (a stored
        # sibling box getting silently discarded) because the merge base
        # is the CURRENT stored list, not this task's snapshot of it; M1
        # (a stale/reused revision and box_seq) because `current_src` for
        # `boxes_write_fields` -- and the ids `finalize_box_ids` mints --
        # both come from `current`, never `task.region_revision` /
        # `task.region_box_seq`.
        if task.pending_boxes is not None:
            stored_now = read_boxes(current, F)
            if task.reverify:
                # Path 1 re-verify (W8 B1): this pass only resolved the
                # stored `proposed` box(es) it re-verified -- every OTHER
                # sibling (already accepted/rejected, or a second
                # `proposed` box this pass didn't select) must survive,
                # by id.
                #
                # R-M3 fix (M1 residual, 2026-09-27 re-review):
                # `task.stored_boxes` is the fetch-time snapshot this
                # pass's candidates were actually built from and sent to
                # the VLM against -- passed as `baseline` so a box a
                # human moved or deleted DURING that VLM call is detected
                # (per-box, by comparing `baseline` to `stored_now`) and
                # never silently overwritten/resurrected by this pass's
                # now-stale verdict for it.
                merged = merge_boxes_for_write(
                    stored_now, task.pending_boxes, baseline=task.stored_boxes
                )
            elif task.pending_merge:
                # W8c M1 fix (2026-09-28 re-review): a FRESH detection
                # pass (Path 2/3, or the text-hint re-pass they can fall
                # into) is a new answer to "where are the regions?", not
                # a partial update -- "merge" (keeping a stale sibling by
                # id) was the wrong semantic here and let a stale
                # MACHINE-sourced box (e.g. a prior pass's sanity-gate
                # reject, left in place by a `clear_detection=False`
                # requeue) accumulate forever and keep overriding this
                # pass's own derived status. Keep only stored boxes a
                # human owns (`is_human_owned` -- created OR explicitly
                # accepted/rejected/transcribed via the W8a per-box edit
                # routes); replace every machine-sourced one with this
                # pass's own fresh findings.
                keep = [b for b in stored_now if is_human_owned(b)]
                merged = [*keep, *task.pending_boxes]
            else:
                merged = list(task.pending_boxes)
            merged = finalize_box_ids(
                merged, existing=stored_now, seq=int(current.get(F.box_seq) or 0)
            )
            if task.pending_status is not None:
                update[F.status] = task.pending_status
            else:
                assert task.pending_empty_status is not None, (
                    'pending_boxes set without pending_status or pending_empty_status'
                )
                update[F.status] = derive_status(merged, empty_status=task.pending_empty_status)
            update.update(boxes_write_fields(merged, current_src=current))
            # R-M1 fix (2026-09-27 re-review): correct the PROVISIONAL
            # ``F.status`` ``_box_list_doc`` / ``accept_without_vlm``
            # stamped onto ``task.update_doc`` at task-processing time
            # (computed from this task's own boxes alone, before any
            # merge with a stored sibling) to the REAL merged status
            # computed just above. ``region_embed_stage._eligible_tasks``
            # already ran (before this merge, off the provisional value --
            # see ``_box_list_doc``'s docstring for why that's safe) and
            # cannot be redone here, but ``_publish_region_events`` reads
            # ``task.update_doc`` AFTER this merge, so give it the
            # accurate value instead of the provisional one.
            task.update_doc[F.status] = update[F.status]
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
        # A vlm_unmatched write must not keep the class the VLM's answer
        # just contradicted (IT-2). Runs after the stale-write strip above
        # so a stripped-of-class-fields update (stale/locked) never gets a
        # class_id reset re-added; class_write_locked() inside the helper
        # is a second, independent guard against a locked item.
        if update.get('class_source') == VLM_UNMATCHED_CLASS_SOURCE:
            update.update(unmatched_class_clear(current))
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
        # Config-store provenance (W2 sec 4.5): every worker region write
        # stamps the profile that produced it. Read once per bulk call
        # (all `eligible` tasks share one bound project), not per task --
        # by the time this runs, the producer's quiesce-and-swap has
        # already drained every in-flight item onto the *old* runtime's
        # queues, so the store's current active refs always match
        # whatever pass actually processed this batch. Only stamp when
        # something is actually being written -- an update that stripped
        # down to empty (a documented noop, e.g. a stale/locked
        # class-only write) must stay empty, never turn into a real
        # write just because of the stamp.
        if update:
            if profile_name is not None:
                update[F.profile] = profile_name
                update[F.profile_revision] = profile_revision
            # Minor 5 (W2 review): `vlm_prompt_pack` is a per-TASK stamp,
            # not a per-batch one -- a batch's pack may be configured and
            # resolvable even when this particular task's write never
            # actually involved a VLM call (no VLM configured at all, or a
            # write path that skipped it, e.g. the high-confidence
            # secondary-segmenter auto-skip). Stamping unconditionally
            # would claim a VLM ran when it didn't.
            if pack_stamp is not None and task.vlm_called:
                update['vlm_prompt_pack'] = pack_stamp
            # Who answered: the identity the call actually went to (kept on
            # the task), never read from the store at write time -- a swap
            # between the call and this flush must not relabel the answer.
            if task.vlm_called and task.vlm_identity is not None:
                update['vlm_endpoint'] = task.vlm_identity.endpoint_ref
                update['vlm_model'] = task.vlm_identity.model
        return update

    result = await occ_skip_on_conflict_bulk(
        opensearch,
        doc_ids=[t.crop_id for t in eligible],
        merger=_merge,
        index=items_index(),
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
# S-3: fall back to whatever env var already names the API base — the
# detection worker's own OP_API_BASE_URL/OP_API (used elsewhere for the
# same host:port) is a zero-config default instead of requiring a
# fourth, worker-specific env var just for this.
_EVENT_API_URL = (
    os.environ.get('OP_EVENT_API_URL')
    or os.environ.get('OP_API_BASE_URL')
    or os.environ.get('OP_API')
    or ''
).rstrip('/')
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
    url = f'{_EVENT_API_URL}{project_api_base()}/events/publish'
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
