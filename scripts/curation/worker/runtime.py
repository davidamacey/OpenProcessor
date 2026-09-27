"""Scaffold for the detection worker's hot-reloadable runtime (W2,
any_domain_plan.md §4.5).

**Status: scaffold only.** :class:`RegionRuntime` documents the target
shape and :func:`build_runtime` is a thin wrapper the runner can grow
into once the ~40 ``state.region_profile()`` call sites
(``runner.py:653,668,937-948,1003,1025-1031``; ``verify.py:137,199``;
``state.py:281``) are migrated to read ``runtime_holder.current`` per
item, per §4.5's plan. That migration, plus the producer loop's
quiesce-and-swap (drain every queue, ``store.pin_active()``, rebuild,
resume) and the no-profile wait loop, are **not yet wired into
``runner.py``** -- deliberately deferred out of this wave to keep the
diff on ``runner.py`` (actively edited by the parallel
``cutover/projects-workers`` wave) surgical. See the W2 completion
report for the exact remaining steps.

:func:`config_wants_swap` is usable today: it is the cheap "did the
active profile/pack change" check the eventual producer-loop swap would
call each fetch cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.config_store.store import AxisRef, ConfigStore
    from src.services.labeling.vlm_prompts import PromptPack


@dataclass
class RegionRuntime:
    """Everything the worker's per-item stage closures need for one
    (profile, pack) pairing -- built once per activation, swapped
    atomically at a quiesce point (§4.5), never mutated in place.
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
    is built) -- the producer loop's per-cycle check (§4.5 step 2)."""
    want = (store.current.active_profile, store.current.active_pack)
    return want != current_refs
