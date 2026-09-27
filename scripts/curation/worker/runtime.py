"""The detection worker's hot-reloadable runtime (W2, any_domain_plan.md
sec 4.5; per-project per projects_plan.md sec 11 W2).

:func:`build_runtime` is the extracted build block that used to live
inline in ``runner.py`` (``runner.py:227-331`` pre-W2): given a Triton
pool, an already-resolved ``(profile, pack)`` pair and the worker's CLI
args, it builds every heavy-IO object one profile activation needs
(detector, segmenter, OCR recognizer, text rules, VLM) and returns them
as one immutable :class:`RegionRuntime`.

:class:`RuntimeHolder` keeps one built :class:`RegionRuntime` per
project slug -- "alpha activating a new profile does not touch beta's
runtime" falls out for free because each slug's entry is only ever
replaced by a swap keyed to that same slug (projects_plan.md sec 11 W2:
"the worker RegionRuntime, quiesce-and-swap and runtime:* docs are per
project").

:func:`quiesce_and_swap` is the producer loop's per-cycle hot-reload
primitive (sec 4.5 step 2): drain every queue so no in-flight item can
straddle two profiles, then swap in a freshly built runtime for one
project slug. Draining is process-wide (the queues are shared across
projects), but only the *changed* project's holder entry moves --
that is what makes the swap project-scoped rather than global.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    import argparse
    import asyncio

    from src.config import DetectionProfile
    from src.services.config_store.store import AxisRef, ConfigStore
    from src.services.labeling.vlm_prompts import PromptPack


logger = get_logger('curation_worker')


@dataclass
class RegionRuntime:
    """Everything the worker's per-item stage closures need for one
    (profile, pack) pairing -- built once per activation, swapped
    atomically at a quiesce point (sec 4.5), never mutated in place.
    """

    profile: DetectionProfile
    profile_ref: tuple[str, int | None]
    pack: PromptPack
    pack_ref: tuple[str, int | None]
    detector: Any
    segmenter: Any
    ocr_recognizer: Any
    text_rules: Any
    vlm: Any
    item_text_enabled: bool
    item_text_min_conf: float
    #: True when this profile/deployment has a VLM URL configured. Per
    #: runtime (M7 sibling issue) rather than a process-wide constant --
    #: a swap can change nothing about VLM availability today, but
    #: computing it fresh keeps it from silently drifting from `vlm`.
    vlm_available: bool
    #: sec 4.5/M7: computed from *this* runtime's profile + segmenter,
    #: not the startup ones -- a swap to a profile with different
    #: text-hint settings must not keep running the old gate.
    text_hint_on: bool


def config_wants_swap(store: ConfigStore, current_refs: tuple[AxisRef, AxisRef] | None) -> bool:
    """``True`` when the store's served active ``(profile, pack)`` refs
    differ from ``current_refs`` -- the producer loop's per-cycle check
    (sec 4.5 step 2). B2: reads ``pending_snapshot`` when one is staged
    (pinned mode after a ``refresh()``, before the next ``pin_active()``)
    so this agrees with :func:`maybe_hot_reload`'s own check; falls back
    to ``current`` (live mode, or pinned with nothing pending)."""
    snapshot = store.pending_snapshot if store.pending_snapshot is not None else store.current
    want = (snapshot.active_profile, snapshot.active_pack)
    return want != current_refs


async def build_runtime(
    pool: Any,
    profile: DetectionProfile,
    pack: PromptPack,
    args: argparse.Namespace,
    *,
    region_detector_cls: Any,
    ocr_recognizer_cls: Any,
    segmenter_cls: Any,
    vlm_cls: Any,
    profile_revision: int | None = None,
    pack_revision: int | None = None,
) -> RegionRuntime:
    """Build one :class:`RegionRuntime` for ``(profile, pack)``.

    Extracted, byte-for-byte, from the build block ``runner.py`` used to
    run inline at module-startup time (pre-W2) so it can be re-run at a
    quiesce point without duplicating the construction logic. Never
    touches OpenSearch or the OS-level pause sentinel -- those stay in
    ``runner.py``'s own startup path.

    The four heavy-IO constructors are passed in by the caller rather
    than imported fresh here: ``runner.py`` resolves them through its
    own module globals / the ``region_worker_main`` shim
    (``_wkr.SegmenterClient``, ``_wkr.VlmLabeler``, module-level
    ``RegionDetector``/``PaddleOcrTextRecognizer``) specifically so that
    ``tests/curation/test_region_worker.py`` /
    ``test_region_text_worker.py``'s ``monkeypatch.setattr(...)`` calls
    on those names are honoured on every rebuild, not just the first.
    Importing them fresh from ``src.services.detection.cascade_detect``
    here would silently bypass those patches.
    """
    from src.core.logging import get_logger as _get_logger
    from src.services.detection.region_text import validate_text_reader
    from src.services.detection.region_text_rules import region_text_rules

    _logger = _get_logger('curation_worker')

    detector = region_detector_cls(pool, profile)
    ocr_recognizer = ocr_recognizer_cls(pool, profile)

    segmenter_url = getattr(args, 'segmenter_url', '') or ''
    if segmenter_url and not profile.segmenter_text_prompt:
        _logger.warning(
            'segmenter_disabled_no_text_prompt',
            profile=profile.name,
            detail='set OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT to use the segmenter leg',
        )
        segmenter_url = ''
    segmenter = segmenter_cls(
        segmenter_url,
        text_prompt=profile.segmenter_text_prompt,
        source_name=profile.segmenter_name,
    )

    text_rules = region_text_rules(profile, pack)
    vlm_url = getattr(args, 'vlm_url', '') or ''
    vlm_available = bool(vlm_url.strip())
    validate_text_reader(profile.text_reader)
    text_hint_on = profile.text_hint_active(segmenter_enabled=bool(segmenter.enabled))

    from src.config import get_curation_config

    _cfg = get_curation_config()
    item_text_enabled = _cfg.item_text_enabled and bool(profile.ocr_pipeline_model)
    item_text_min_conf = _cfg.item_text_min_confidence
    vlm = vlm_cls(base_url=vlm_url, pack=pack) if vlm_available else None

    return RegionRuntime(
        profile=profile,
        profile_ref=(profile.name, profile_revision),
        pack=pack,
        pack_ref=(pack.name, pack_revision),
        detector=detector,
        segmenter=segmenter,
        ocr_recognizer=ocr_recognizer,
        text_rules=text_rules,
        vlm=vlm,
        item_text_enabled=item_text_enabled,
        item_text_min_conf=item_text_min_conf,
        vlm_available=vlm_available,
        text_hint_on=text_hint_on,
    )


@dataclass
class RuntimeHolder:
    """Per-project-slug :class:`RegionRuntime` registry (projects_plan.md
    sec 11 W2's ``runtimes[slug]``). Activating a profile in one project
    only ever replaces that project's own entry.

    ``_synced_refs`` is tracked separately from the runtime's own
    ``profile_ref``/``pack_ref`` (which always name *some* profile/pack,
    even the env/file default that was never activated through the
    store). The swap decision must compare against the store's own
    ``AxisRef`` shape -- ``None`` / ``'off'`` / ``(name, revision)`` --
    never against the runtime's refs, or a runtime built from an
    env-default (never activated) would spuriously "differ" from the
    store's ``None`` on every single check.
    """

    _runtimes: dict[str, RegionRuntime] = field(default_factory=dict)
    _synced_refs: dict[str, tuple[AxisRef, AxisRef]] = field(default_factory=dict)

    def get(self, slug: str) -> RegionRuntime | None:
        return self._runtimes.get(slug)

    def set(self, slug: str, runtime: RegionRuntime) -> None:
        self._runtimes[slug] = runtime

    def drop(self, slug: str) -> None:
        self._runtimes.pop(slug, None)
        self._synced_refs.pop(slug, None)

    def get_synced_refs(self, slug: str) -> tuple[AxisRef, AxisRef] | None:
        return self._synced_refs.get(slug)

    def set_synced_refs(self, slug: str, refs: tuple[AxisRef, AxisRef]) -> None:
        self._synced_refs[slug] = refs

    @property
    def current(self) -> dict[str, RegionRuntime]:
        """Read-only view -- callers must go through :meth:`get`/:meth:`set`
        to mutate."""
        return dict(self._runtimes)


def _revision_for(ref: AxisRef, *, resolved_name: str) -> int | None:
    """The activation's revision, but only when the resolved object
    actually IS the activated name (B4): a fallback to the env/file
    default (stored name no longer exists, or the axis was never
    activated) must never inherit a stale/unrelated revision number."""
    if isinstance(ref, tuple) and ref[0] == resolved_name:
        return ref[1]
    return None


async def quiesce_and_swap(
    *,
    queues: list[asyncio.Queue[Any]],
    holder: RuntimeHolder,
    slug: str,
    store: ConfigStore,
    pool: Any,
    args: argparse.Namespace,
    want: tuple[AxisRef, AxisRef],
    get_active_profile: Any,
    get_active_pack: Any,
    region_detector_cls: Any,
    ocr_recognizer_cls: Any,
    segmenter_cls: Any,
    vlm_cls: Any,
) -> RegionRuntime | None:
    """Drain every queue (sec 4.5 step 2.2 -- in-flight items finish on
    the *old* runtime, stamped with its own refs since the writer's
    ``out_q.task_done()`` no longer fires until the flush that produced
    the stamp actually ran -- see ``bulk_writer.py``), pin the store
    (B2: only after the drain, so an in-flight item's per-item
    ``region_profile()``/``active_prompt_pack()`` reads never see the
    new snapshot early), THEN resolve and build. Returns ``None`` (never
    raises) when the newly-pinned snapshot has no active profile at all
    (deactivated, or never configured) -- the caller keeps whatever
    runtime it already has, if any, and idles that project (M2).
    """
    for q in queues:
        await q.join()
    store.pin_active()

    new_profile = get_active_profile()
    if new_profile is None:
        return None
    new_pack = get_active_pack()

    profile_ref, pack_ref = want
    old = holder.get(slug)
    new_runtime = await build_runtime(
        pool,
        new_profile,
        new_pack,
        args,
        region_detector_cls=region_detector_cls,
        ocr_recognizer_cls=ocr_recognizer_cls,
        segmenter_cls=segmenter_cls,
        vlm_cls=vlm_cls,
        profile_revision=_revision_for(profile_ref, resolved_name=new_profile.name),
        pack_revision=_revision_for(pack_ref, resolved_name=new_pack.name),
    )
    holder.set(slug, new_runtime)
    if old is not None:
        with contextlib.suppress(Exception):
            await old.segmenter.aclose()
        if old.vlm is not None:
            with contextlib.suppress(Exception):
                await old.vlm.aclose()
    logger.info(
        'region_runtime_swapped',
        project=slug,
        profile=new_runtime.profile_ref,
        pack=new_runtime.pack_ref,
    )
    return new_runtime


async def maybe_hot_reload(
    *,
    store: ConfigStore,
    opensearch: Any,
    holder: RuntimeHolder,
    slug: str,
    pool: Any,
    args: argparse.Namespace,
    queues: list[asyncio.Queue[Any]],
    get_active_profile: Any,
    get_active_pack: Any,
    region_detector_cls: Any,
    ocr_recognizer_cls: Any,
    segmenter_cls: Any,
    vlm_cls: Any,
) -> RegionRuntime | None:
    """The producer loop's per-cycle hot-reload check (sec 4.5 step 2),
    as one call: refresh ``slug``'s store, compare what it now *serves*
    against what this holder last synced to, and only swap when that
    pair actually changed -- never on object identity, never on the
    runtime's own ``profile_ref``/``pack_ref`` (see :class:`RuntimeHolder`).

    B2: in ``pinned`` mode, :meth:`ConfigStore.refresh` only stages
    ``pending_snapshot`` -- ``store.current`` does not move until
    :meth:`ConfigStore.pin_active` runs. The "did anything change" check
    must read whichever of the two is freshest (``pending_snapshot`` if
    a refresh just staged one, else ``current``), or a pinned store can
    never detect a real activation past the very first cycle.

    Returns the (possibly unchanged) current :class:`RegionRuntime` for
    ``slug``, or ``None`` if none has ever been built and the store has
    no active profile either (nothing to run yet).
    """
    await store.refresh(opensearch)
    snapshot = store.pending_snapshot if store.pending_snapshot is not None else store.current
    want: tuple[AxisRef, AxisRef] = (snapshot.active_profile, snapshot.active_pack)
    last = holder.get_synced_refs(slug)
    if last is not None and want == last:
        return holder.get(slug)

    new_runtime = await quiesce_and_swap(
        queues=queues,
        holder=holder,
        slug=slug,
        store=store,
        pool=pool,
        args=args,
        want=want,
        get_active_profile=get_active_profile,
        get_active_pack=get_active_pack,
        region_detector_cls=region_detector_cls,
        ocr_recognizer_cls=ocr_recognizer_cls,
        segmenter_cls=segmenter_cls,
        vlm_cls=vlm_cls,
    )
    holder.set_synced_refs(slug, want)
    if new_runtime is None:
        logger.warning('region_profile_deactivated_mid_run', project=slug)
        return holder.get(slug)
    return new_runtime


async def upsert_project_runtime_doc(
    client: Any,
    *,
    index: str,
    hostname: str,
    project: str,
    runtime: RegionRuntime,
    config_revision: int,
) -> None:
    """Write ``runtime:detection_worker:<hostname>`` for ``project``'s
    configs index -- ``GET /region_profiles/active`` lists these under
    ``applied`` (sec 4.5)."""
    from src.services.config_store.index import upsert_runtime_doc

    await upsert_runtime_doc(
        client,
        index,
        process='detection_worker',
        hostname=hostname,
        fields={
            'project': project,
            'applied_config_revision': config_revision,
            'profile': runtime.profile_ref[0],
            'profile_revision': runtime.profile_ref[1],
            'pack': runtime.pack_ref[0],
            'pack_revision': runtime.pack_ref[1],
        },
    )
