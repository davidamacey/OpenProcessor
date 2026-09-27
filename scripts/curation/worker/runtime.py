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


def config_wants_swap(store: ConfigStore, current_refs: tuple[AxisRef, AxisRef] | None) -> bool:
    """``True`` when the store's pinned/current active
    ``(profile, pack)`` refs differ from ``current_refs`` (the running
    :class:`RegionRuntime`'s refs, or ``None`` before the first runtime
    is built) -- the producer loop's per-cycle check (sec 4.5 step 2)."""
    want = (store.current.active_profile, store.current.active_pack)
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

    from src.config import get_curation_config

    item_text_enabled = get_curation_config().item_text_enabled and bool(profile.ocr_pipeline_model)
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


async def quiesce_and_swap(
    *,
    queues: list[asyncio.Queue[Any]],
    holder: RuntimeHolder,
    slug: str,
    pool: Any,
    profile: DetectionProfile,
    pack: PromptPack,
    args: argparse.Namespace,
    region_detector_cls: Any,
    ocr_recognizer_cls: Any,
    segmenter_cls: Any,
    vlm_cls: Any,
    profile_revision: int | None = None,
    pack_revision: int | None = None,
) -> RegionRuntime:
    """Drain every queue (sec 4.5 step 2.2 -- in-flight items finish on the
    *old* runtime), build the new runtime, and swap it into ``holder`` for
    ``slug`` only. Other projects' entries in ``holder`` are untouched.
    """
    for q in queues:
        await q.join()
    old = holder.get(slug)
    new_runtime = await build_runtime(
        pool,
        profile,
        pack,
        args,
        region_detector_cls=region_detector_cls,
        ocr_recognizer_cls=ocr_recognizer_cls,
        segmenter_cls=segmenter_cls,
        vlm_cls=vlm_cls,
        profile_revision=profile_revision,
        pack_revision=pack_revision,
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


def _axis_name_and_revision(ref: AxisRef) -> tuple[str | None, int | None]:
    if isinstance(ref, tuple):
        return ref
    return None, None


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
    as one call: refresh ``slug``'s store, compare its served
    ``(active_profile, active_pack)`` ``AxisRef`` pair against what this
    holder last synced to, and only swap when that pair actually
    changed -- never on object identity, never on the runtime's own
    ``profile_ref``/``pack_ref`` (see :class:`RuntimeHolder`).

    Returns the (possibly unchanged) current :class:`RegionRuntime` for
    ``slug``, or ``None`` if none has ever been built and the store has
    no active profile either (nothing to run yet).
    """
    await store.refresh(opensearch)
    want: tuple[AxisRef, AxisRef] = (store.current.active_profile, store.current.active_pack)
    last = holder.get_synced_refs(slug)
    if last is not None and want == last:
        return holder.get(slug)

    profile_ref, pack_ref = want
    new_profile = get_active_profile()
    if new_profile is None:
        logger.warning('region_profile_deactivated_mid_run', project=slug)
        holder.set_synced_refs(slug, want)
        return holder.get(slug)

    new_pack = get_active_pack()
    _, profile_revision = _axis_name_and_revision(profile_ref)
    _, pack_revision = _axis_name_and_revision(pack_ref)
    new_runtime = await quiesce_and_swap(
        queues=queues,
        holder=holder,
        slug=slug,
        pool=pool,
        profile=new_profile,
        pack=new_pack,
        args=args,
        region_detector_cls=region_detector_cls,
        ocr_recognizer_cls=ocr_recognizer_cls,
        segmenter_cls=segmenter_cls,
        vlm_cls=vlm_cls,
        profile_revision=profile_revision,
        pack_revision=pack_revision,
    )
    holder.set_synced_refs(slug, want)
    store.pin_active()
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
