"""Shared state and helpers of the detection worker's streaming pipeline.

The stage tasks (``stage_a``, ``stage_sam``, ``stage_b``, ``flow``) take a
:class:`PipelineContext`; ``runner.run`` builds it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.combined_resolve import should_classify
from scripts.curation.worker.state import RegionProfileNotConfiguredError, _ItemTask
from scripts.curation.worker.verify import TaskBoxInput
from src.config import get_region_fields
from src.services.detection.cascade_detect import crop_norm_to_source_norm
from src.services.detection.region_candidates import select_region_candidates


if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from scripts.curation.worker.crop_gate import CropGate
    from scripts.curation.worker.fairness import FairnessScheduler
    from scripts.curation.worker.no_verdict import NoVerdictCounter
    from scripts.curation.worker.runtime import RuntimeHolder
    from src.config.region_state import RegionStatus
    from src.services.curation.region_boxes import RegionBox


# Wait at most this long for a VLM chunk to fill before firing it under-full.
# Keeps tail latency bounded when the queue empties out near end-of-run while
# still benefiting from batching during steady-state.
VLM_CHUNK_DRAIN_TIMEOUT = 0.10
# Aligned with the shared VLM's --limit-mm-per-prompt {"image":6}.
# Per-call work scales worse than linearly past 6 on the reference
# deployment's GPU for this prompt+image mix.
COMBINED_CHUNK = 6
# Same per-call image budget for the visibility filter (the
# upstream --limit-mm-per-prompt cap is shared across both prompts).
VISIBLE_CHUNK = 6


@dataclass
class PipelineContext:
    """Everything the streaming pipeline's tasks share: the inter-stage
    queues, the in-flight bookkeeping, the counters and the per-project
    runtime holder. Built once by ``runner.run`` and passed to every task."""

    args: argparse.Namespace
    stop_event: asyncio.Event
    opensearch: Any
    sentinel: Path
    started_at: float
    region_embed_pe: Any
    project_registry: Any
    fairness_scheduler: FairnessScheduler
    runtime_holder: RuntimeHolder
    in_q: asyncio.Queue[_ItemTask | None]
    vlm_visible_q: asyncio.Queue[_ItemTask | None]
    sam_q: asyncio.Queue[_ItemTask | None]
    combined_q: asyncio.Queue[_ItemTask | None]
    out_q: asyncio.Queue[_ItemTask | None]
    in_flight: set[str]
    in_flight_owner: dict[str, str]
    in_flight_lock: asyncio.Lock
    released_at: dict[str, float]
    metrics: dict[str, int]
    no_verdict_cap: int
    visible_no_verdict: NoVerdictCounter
    combined_no_verdict: NoVerdictCounter
    crop_gate: CropGate

    def rt_for(self, t: _ItemTask) -> Any:
        """The runtime this item's own project is currently on. Never a
        process-wide default -- a project with no runtime yet (no
        profile configured anywhere, or not synced this cycle) raises,
        which every stage's existing `except Exception` handler turns
        into "drop from in_flight, write nothing" (M2's no-profile-wait
        semantics, applied per project instead of per process)."""
        rt = self.runtime_holder.get(t.project.slug)
        if rt is None:
            msg = f"no region runtime built yet for project '{t.project.slug}'"
            raise RegionProfileNotConfiguredError(msg)
        return rt


# How long the producer remembers a released crop. Only has to outlive the
# slowest single pending search.
_RELEASED_AT_TTL_S = 300.0


def _should_classify(t: _ItemTask, *, registry_loaded: bool) -> bool:
    """Whether the combined prompt asks the VLM for this task's item class
    (:func:`~scripts.curation.worker.combined_resolve.should_classify` over
    the task's class state). Module-level so it is directly unit-testable --
    see ``tests/curation/test_write_guards.py``."""
    return should_classify(
        class_validated=t.class_validated,
        stored_class_source=t.class_source,
        test_holdout=t.test_holdout,
        class_confidence=t.class_confidence,
        registry_loaded=registry_loaded,
    )


# =============================================================================
# W8 multi-box candidate wiring helpers (module-level so they're directly
# unit-testable, same rationale as ``_should_classify``).
# =============================================================================


def _select_candidates(
    raw: list[Any],
    *,
    profile: Any,
    item_bbox_norm: tuple[float, float, float, float],
    detector: str,
    detector_version: str,
    source: str,
    min_score: float = 0.0,
) -> list[TaskBoxInput]:
    """Floor/NMS/cap ``raw`` (a detector/segmenter leg's candidate list)
    then wrap the selection as :class:`TaskBoxInput`, source-frame bbox
    projected via ``item_bbox_norm``. Every selected candidate is fresh
    (``box_id=None`` -- ``verdicts_to_boxes`` mints one).
    """
    sel = select_region_candidates(
        raw,
        min_score=min_score,
        iou=profile.region_nms_iou,
        max_n=profile.max_regions_per_item,
    )
    return [
        TaskBoxInput(
            bbox_in_crop=c.bbox_norm,
            bbox_in_source=crop_norm_to_source_norm(c.bbox_norm, item_bbox_norm),
            score=c.score,
            detector=detector,
            detector_version=detector_version,
            source=source,
        )
        for c in sel.selected
    ]


def _box_list_doc(
    t: _ItemTask,
    boxes: list[RegionBox],
    status: RegionStatus,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stash this pass's own resolved ``boxes`` (NOT yet merged with any
    concurrently-stored siblings) onto ``t`` for the writer to finish at
    write time, plus whatever non-box fields (chain, class update, ...)
    land in the same update.

    W8 B1 + M1 fix (pipeline-wiring review, 2026-09-27): the actual box
    list merge, status derivation, id finalization and
    ``region_revision``/``region_box_seq`` bump used to happen HERE,
    against this task's own fetch-time snapshot -- silently discarding
    any stored sibling box (B1) and reusing/reseting ids and the revision
    against data that may already be stale by write time (M1). Both are
    now deferred to ``bulk_writer._merge``, which re-reads the live doc
    immediately before the write (the same OCC discipline as
    ``occ_skip_on_conflict_bulk`` uses everywhere else) and merges/mints
    against THAT, never this snapshot. ``status`` here is the fallback
    :func:`~src.services.curation.region_boxes.derive_status` uses when
    the final (post-merge) box list is empty -- for every call site in
    this module that already equals what ``derive_status`` would compute
    from ``boxes`` alone, so this is a no-op for the common (no stored
    siblings) case.

    R-M1 fix (2026-09-27 re-review): also stash a PROVISIONAL ``F.status``
    directly onto ``doc`` (``t.update_doc``). The real, post-merge status
    is only known inside ``bulk_writer._merge`` (it can differ when a
    stored sibling box changes what the merged list derives to), but two
    consumers read ``t.update_doc`` BEFORE that merge ever runs --
    ``region_embed_stage._eligible_tasks`` (called from ``writer()``
    ahead of ``_bulk_update``) and ``bulk_writer._publish_region_events``
    (reads ``t.update_doc`` after the fact, never the merged dict). Using
    ``derive_status(boxes)`` here is always safe in the direction that
    matters: an accepted box in ``boxes`` guarantees the merged status is
    ALSO ``detected`` (accepted is top precedence), so eligibility can
    never be a false positive; it can only under-embed a merge-mode item
    whose OWN boxes are all rejected but whose merged status is
    ``detected`` because of an already-accepted sibling -- and that
    sibling's own pass already wrote (and this worker never re-embeds)
    its embedding, so nothing is lost.
    """
    F = get_region_fields()
    t.pending_boxes = list(boxes)
    t.pending_empty_status = status
    doc: dict[str, Any] = {F.status: status}
    if t.detection_trace:
        doc[F.detector_chain] = list(t.detection_trace)
    if extra:
        doc.update(extra)
    return doc


def _sync_singular_candidate(t: _ItemTask) -> None:
    """Point ``t.candidate_source`` at ``t.candidates[0]``'s source.

    Kept for the one remaining deliberately single-box consumer:
    ``accept_without_vlm`` (no VLM configured -- nothing can adjudicate
    between multiple candidates, so only the best one is ever written),
    which resolves its detector provenance from it. A no-op when
    ``t.candidates`` is empty.
    """
    if t.candidates:
        t.candidate_source = t.candidates[0].source


async def _drain_chunk(
    q: asyncio.Queue[_ItemTask | None],
    *,
    chunk_size: int,
    drain_timeout: float,
    carry: list[_ItemTask],
) -> tuple[list[_ItemTask], bool]:
    """Pull up to ``chunk_size`` tasks of ONE project from ``q``.

    Blocks indefinitely on the first task; subsequent tasks are
    non-blocking up to ``drain_timeout`` total. Returns
    ``(tasks, poison_received)``. When ``poison_received`` is True,
    the caller should flush ``tasks`` then exit (the poison pill
    was consumed but is not included in ``tasks``).

    Why batch like this
    -------------------
    The visibility + verify endpoints pack 4 crops per upstream
    VLM call. Pulling one task and immediately firing wastes the
    batching opportunity; waiting forever for chunk_size tasks
    creates terrible tail latency near end-of-run when the queue
    empties out. The bounded drain window balances both.

    A batched VLM call runs under one project binding, so a chunk
    never mixes projects: the first task of another project ends the
    chunk and is parked in ``carry`` (owned by the calling consumer),
    which starts that consumer's next chunk. A parked task never
    coexists with a consumed poison pill, so shutdown cannot strand it.
    """

    tasks: list[_ItemTask] = []
    first = carry.pop() if carry else None
    if first is None:
        first = await q.get()
        q.task_done()
        if first is None:
            return tasks, True
    tasks.append(first)

    deadline = asyncio.get_running_loop().time() + drain_timeout
    poisoned = False
    while len(tasks) < chunk_size:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            t = await asyncio.wait_for(q.get(), timeout=remaining)
        except TimeoutError:
            break
        if t is None:
            # Re-queue the poison so siblings can also drain. We
            # only consume one poison pill per consumer, matching
            # the existing N-poison-pills-for-N-consumers shutdown
            # contract upstream.
            q.task_done()
            poisoned = True
            break
        q.task_done()
        if t.project != first.project:
            carry.append(t)
            break
        tasks.append(t)
    return tasks, poisoned
