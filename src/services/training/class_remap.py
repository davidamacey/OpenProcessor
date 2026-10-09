"""Class-remap resolution for promoting a subset-class training run.

Split out of :mod:`src.services.training.triton_promote`. Class identity
crosses the train/promote boundary by name: this module turns a run's
``class_remap`` into the post-training id -> name map for ``labels.txt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.training.promote_errors import ClassRemapUnreadableError


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassRemapResult:
    """A resolved, validated class_remap payload.

    ``mapping`` is ``{original_registry_class_id: new_dense_class_id}`` —
    the same contract the trainer's subset-dataset builder has always
    produced, just parsed into a typed object instead of a bare dict.
    ``source`` records which of the two resolution paths won
    (``'manifest'`` / ``'weights_dir'``), surfaced to the caller as
    ``class_remap_source`` so a promote result is traceable.
    """

    mapping: dict[int, int]
    names: list[str] | None
    single_cls: bool
    include_classes: list[int] | None
    source: str  # 'manifest' | 'weights_dir' | 'none'


_NONE_REMAP = ClassRemapResult(
    mapping={}, names=None, single_cls=False, include_classes=None, source='none'
)


def _parse_class_remap_payload(raw: Any, *, source: str) -> ClassRemapResult:
    """Parse a class_remap payload, real shape first.

    The trainer's subset-dataset builder writes ``{original_to_new,
    new_to_original, single_cls, names, include_classes}`` — a previous
    parser here only understood a flat ``{orig: new}`` dict or a
    ``{'mapping': {...}}`` wrapper, so ``int(k)`` failed on every real key
    and this always silently degraded to "no remap". Old flat/``mapping``
    shapes are still accepted for backward compat with anything that wrote
    them directly. Raises :class:`ClassRemapUnreadableError` — never
    returns ``None`` — on anything unparseable or empty.
    """
    if not isinstance(raw, dict):
        raise ClassRemapUnreadableError(source, f'not a JSON object: {type(raw).__name__}')

    if 'original_to_new' in raw:
        raw_mapping = raw.get('original_to_new')
        names = raw.get('names')
        single_cls = bool(raw.get('single_cls', False))
        include_classes = raw.get('include_classes')
    elif isinstance(raw.get('mapping'), dict):
        raw_mapping = raw['mapping']
        names = raw.get('names')
        single_cls = bool(raw.get('single_cls', False))
        include_classes = raw.get('include_classes')
    else:
        # Legacy flat {orig: new} shape.
        raw_mapping = raw
        names = None
        single_cls = False
        include_classes = None

    if not isinstance(raw_mapping, dict):
        raise ClassRemapUnreadableError(source, "no 'original_to_new'/'mapping' dict found")

    mapping: dict[int, int] = {}
    for k, v in raw_mapping.items():
        try:
            mapping[int(k)] = int(v)
        except (TypeError, ValueError) as exc:
            raise ClassRemapUnreadableError(source, f'non-integer key/value {k!r}: {v!r}') from exc

    if not mapping:
        raise ClassRemapUnreadableError(source, 'mapping is empty')

    return ClassRemapResult(
        mapping=mapping,
        names=list(names) if isinstance(names, list) else None,
        single_cls=single_cls,
        include_classes=[int(c) for c in include_classes]
        if isinstance(include_classes, list)
        else None,
        source=source,
    )


def resolve_class_remap(
    *,
    job_id: str,
    checkpoint_path: Path,
    manifest: dict[str, Any] | None,
) -> ClassRemapResult:
    """Resolve a job's class_remap, manifest first, then the weights dir.

    Resolution order:
        (a) ``manifest['lineage']['class_remap']`` — works for every
            already-completed run, since the trainer has always captured
            this pre-``rmtree`` at manifest-write time.
        (b) ``<checkpoint_path's dir>/class_remap.json`` — written by the
            trainer directly into the weights dir for runs after this fix.
        (c) neither present: :data:`_NONE_REMAP` (source='none') — legal
            only for a full-class run; the caller enforces that.

    A payload that exists but fails to parse is a loud failure
    (:class:`ClassRemapUnreadableError`), never a silent fall-through to
    the next source — an unreadable-but-present remap for a subset run is
    exactly the bug this rewrite fixes.
    """
    lineage = (manifest or {}).get('lineage') or {}
    manifest_remap = lineage.get('class_remap')
    if manifest_remap is not None:
        try:
            return _parse_class_remap_payload(manifest_remap, source='manifest')
        except ClassRemapUnreadableError:
            logger.error('train_promote_class_remap_manifest_unreadable', job_id=job_id)
            raise

    weights_dir_path = checkpoint_path.parent / 'class_remap.json'
    if weights_dir_path.is_file():
        try:
            raw = json.loads(weights_dir_path.read_text(encoding='utf-8'))
        except OSError as exc:
            logger.error(
                'train_promote_class_remap_weights_dir_unreadable', job_id=job_id, error=str(exc)
            )
            raise ClassRemapUnreadableError('weights_dir', str(exc)) from exc
        except ValueError as exc:
            logger.error(
                'train_promote_class_remap_weights_dir_unreadable', job_id=job_id, error=str(exc)
            )
            raise ClassRemapUnreadableError('weights_dir', f'invalid JSON: {exc}') from exc
        try:
            return _parse_class_remap_payload(raw, source='weights_dir')
        except ClassRemapUnreadableError:
            logger.error('train_promote_class_remap_weights_dir_unreadable', job_id=job_id)
            raise

    return _NONE_REMAP


def build_class_id_to_name(
    *,
    remap: ClassRemapResult,
    full_registry: dict[int, str],
) -> dict[int, str]:
    """Resolve the **post-training** class_id → name map for ``labels.txt``.

    For a full-class run (``remap.source == 'none'``), returns
    ``full_registry`` as-is. For a subset run, applies the remap to
    renumber to ``0..N-1``. ``single_cls`` collapses to a single-line
    ``labels.txt``.
    """
    if remap.source == 'none':
        return dict(full_registry)
    if remap.single_cls:
        # One source class: serve its own registry name, not the trainer's
        # generic collapsed name. Several collapsed into one keep the trainer's.
        if len(remap.mapping) == 1:
            registry_name = full_registry.get(next(iter(remap.mapping)))
            if registry_name:
                return {0: registry_name}
        return {0: remap.names[0] if remap.names else 'object'}
    out: dict[int, str] = {}
    for orig_id, new_id in remap.mapping.items():
        registry_name = full_registry.get(orig_id)
        if registry_name is None:
            continue
        out[new_id] = registry_name
    return out
