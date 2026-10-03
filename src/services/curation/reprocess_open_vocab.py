"""The ``open_vocab`` reprocess scope: run the project's ACTIVE prompt set on
stored images (:func:`~src.services.curation.open_vocab_run.run_open_vocab_image`).

A dry run costs nothing but a count: images selected, enabled targets, the
segmenter calls and minutes the pass will take, and the locked items it will
leave alone. A run that finds the segmenter down stops after
:data:`MAX_CONSECUTIVE_OUTAGES` images in a row instead of timing out on every
remaining one; the images it did not reach are counted, never marked done.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.config_store import get_config_store
from src.services.config_store.open_vocab import active_open_vocab_set
from src.services.curation.open_vocab_gate import active_vlm_visible, load_tracker, save_tracker
from src.services.curation.open_vocab_run import (
    GateContext,
    SegmentImage,
    run_open_vocab_image,
    stamp_open_vocab_status,
)
from src.services.curation.reprocess_locks import item_locked
from src.services.curation.reprocess_models import ReprocessScopeResult
from src.services.curation.reprocess_targets import ReprocessTargetsError, items_by_terms
from src.services.detection.segmenter_http import (
    SegmenterCallError,
    first_segmenter_url,
    segmenter_instances,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.detection.open_vocab_set import OpenVocabSet

logger = get_logger(__name__)

#: Planning figure for one segmenter call on a full image (a miss costs the
#: same as a hit); the dry run's minutes are calls x this / instances.
ESTIMATED_SECONDS_PER_CALL = 3
MAX_CONSECUTIVE_OUTAGES = 3


async def current_active_set(opensearch: AsyncOpenSearch) -> tuple[OpenVocabSet, int | None] | None:
    """The activated set and revision (read fresh from the config store), or
    ``None`` when none is active."""
    store = get_config_store()
    await store.refresh(opensearch)
    ov = active_open_vocab_set()
    ref = store.current.active_open_vocab
    if ov is None or not isinstance(ref, tuple):
        return None
    return ov, ref[1]


async def active_set_for_run(opensearch: AsyncOpenSearch) -> tuple[OpenVocabSet, int | None]:
    """:func:`current_active_set`, or :class:`ReprocessTargetsError` when none
    is active: a run never guesses or defaults a set."""
    found = await current_active_set(opensearch)
    if found is None:
        raise ReprocessTargetsError('no open-vocabulary set is active; activate one first')
    return found


async def plan_open_vocab_scope(
    opensearch: AsyncOpenSearch, image_ids: list[str], not_found: int
) -> ReprocessScopeResult:
    """The dry-run result of the scope (read-only)."""
    ov, _revision = await active_set_for_run(opensearch)
    targets = len(ov.enabled_targets)
    url = first_segmenter_url()
    instances = await segmenter_instances(url) if url else None
    calls = len(image_ids) * targets
    minutes = math.ceil(calls * ESTIMATED_SECONDS_PER_CALL / (instances or 1) / 60)
    locked = 0
    if image_ids:
        docs = await items_by_terms(
            opensearch,
            'image_id',
            image_ids,
            index=get_curation_config().items_index,
            includes=_lock_includes(),
        )
        locked = sum(1 for _, src in docs if item_locked(src))
    return ReprocessScopeResult(
        scope='open_vocab',
        selected=len(image_ids),
        not_found=not_found,
        locked_skipped=locked,
        detail={
            'enabled_targets': targets,
            'estimated_calls': calls,
            'segmenter_instances': instances or 1,
            'segmenter_reachable': int(instances is not None),
            'estimated_minutes': minutes,
        },
    )


def _lock_includes() -> list[str]:
    from src.config.region_fields import get_region_fields

    F = get_region_fields()
    return ['class_source', 'class_validated', 'test_holdout', F.boxes, F.validated, F.verifier]


class OpenVocabPass:
    """One run of the scope over many images; accumulates into ``result``."""

    def __init__(
        self,
        ov: OpenVocabSet,
        revision: int | None,
        segment: SegmentImage,
        result: ReprocessScopeResult,
        gate: GateContext,
    ) -> None:
        self.ov, self.revision, self.segment, self.result = ov, revision, segment, result
        self.gate = gate
        self._outages = 0

    @classmethod
    async def start(
        cls,
        opensearch: AsyncOpenSearch,
        ov: OpenVocabSet,
        revision: int | None,
        segment: SegmentImage,
        result: ReprocessScopeResult,
    ) -> OpenVocabPass:
        """A pass with the gate inputs the set asks for: the vision model when
        ``tier2_vlm_precheck`` is on (without one, tier 2 does not run and
        ``detail.vlm_precheck_unavailable`` says so), and the persisted hit-rate
        windows when ``tier3_hit_rate`` is on."""
        gate = GateContext()
        if ov.gating.tier2_vlm_precheck:
            gate.vlm_visible = await active_vlm_visible(opensearch)
            if gate.vlm_visible is None:
                result.detail['vlm_precheck_unavailable'] = 1
        if ov.gating.tier3_hit_rate.enabled:
            gate.tracker = await load_tracker(opensearch)
        return cls(ov, revision, segment, result, gate)

    async def finish(self, opensearch: AsyncOpenSearch) -> None:
        """Persist what the pass learned (tier-3 windows)."""
        if self.gate.tracker is not None:
            await save_tracker(opensearch, self.gate.tracker)

    @property
    def tripped(self) -> bool:
        return self._outages >= MAX_CONSECUTIVE_OUTAGES

    def _add(self, key: str, value: int) -> None:
        self.result.detail[key] = self.result.detail.get(key, 0) + value

    def skip(self) -> None:
        self._add('not_attempted_segmenter_down', 1)

    async def run_image(
        self,
        opensearch: AsyncOpenSearch,
        service: CurationIngestService,
        image_id: str,
        image_doc: dict[str, Any],
    ) -> None:
        """Run one image and stamp its ``open_vocab_status``: ``done`` on
        success (``skipped_gate`` when the gate spent no call at all), ``failed`` when the image itself could not be processed. A
        segmenter outage stamps nothing: the image stays as it was (``pending``
        when it was queued by ingest) so a later run picks it up."""
        try:
            out = await run_open_vocab_image(
                opensearch,
                service,
                image_id,
                image_doc,
                self.ov,
                revision=self.revision,
                segment=self.segment,
                gate=self.gate,
            )
        except SegmenterCallError as exc:
            self._outages += 1
            logger.warning('open_vocab_segmenter_unavailable', image_id=image_id, error=str(exc))
            self.result.failed += 1
            self._add('failed_segmenter_unavailable', 1)
            return
        except Exception as exc:
            logger.warning('open_vocab_image_failed', image_id=image_id, error=str(exc))
            self.result.failed += 1
            await stamp_open_vocab_status(opensearch, image_id, 'failed')
            return
        self._outages = 0
        skipped = sum(out.skipped.values())
        await stamp_open_vocab_status(
            opensearch, image_id, 'skipped_gate' if skipped and not out.calls else 'done'
        )
        self.result.queued += 1
        self.result.locked_skipped += out.locked_untouched
        for key, value in out.as_counts().items():
            if key != 'locked_untouched':
                self._add(key, value)


__all__ = [
    'ESTIMATED_SECONDS_PER_CALL',
    'MAX_CONSECUTIVE_OUTAGES',
    'OpenVocabPass',
    'active_set_for_run',
    'current_active_set',
    'plan_open_vocab_scope',
]
