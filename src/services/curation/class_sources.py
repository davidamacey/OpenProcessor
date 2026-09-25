"""Every ``class_source`` value this deployment can write, and what each means.

``class_source`` records which writer set an item's class. Some values are
fixed writer names (``vlm``, ``human``, …); the ingest ones are derived
from the configured ingest profiles
(:mod:`src.services.curation.ingest_class_sources`). This module is the
single catalog of both, served by ``GET {prefix}/class_sources`` so a
client can render any value without hardcoding one deployment's detector
names. ``tests/curation/test_class_sources.py`` scans the codebase's
``class_source`` writes and fails if one is missing here.

It also derives the VLM class suggestion carried on every wire item
(``vlm_proposed_class_id`` / ``vlm_proposed_class_name``).
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    DEFAULT_PROPOSAL_CLASS_SOURCE,
    HUMAN_CLASS_SOURCE,
    LABEL_IMPORT_CLASS_SOURCE,
    VLM_CLASS_SOURCE,
    is_classifier_class_source,
)
from src.services.detection.region_text import VLM_TEXT_CONFIDENCE


VLM_UNMATCHED_CLASS_SOURCE = 'vlm_unmatched'
VLM_NEW_CLASS_PENDING_CLASS_SOURCE = 'vlm_new_class_pending'
# Registry reclassification of `vlm_unmatched` items (prefix 'vlm').
VLM_RECLASSIFIED_CLASS_SOURCE = 'vlm_reclassified'
HUMAN_MOVE_CLASS_SOURCE = 'human_move'
CLASS_MERGE_CLASS_SOURCE = 'class_merge'

# ``label_source`` values a human class write may carry. The server always
# writes ``class_source='human'`` for these writes itself; a client can only
# say how the human decided (typed a label vs confirmed a suggestion, or
# bulk-resolved a VLM new-class proposal), never make a human write look
# machine-made.
HumanLabelSource = Literal['human', 'human_confirmed', 'new_class_proposal']
HUMAN_LABEL_SOURCES: tuple[str, ...] = get_args(HumanLabelSource)

# Sources where the VLM picked a registry class that is still only a
# machine suggestion (until class_validated flips).
VLM_SUGGESTION_CLASS_SOURCES = frozenset({VLM_CLASS_SOURCE, VLM_RECLASSIFIED_CLASS_SOURCE})
# Every source whose class decision came from a VLM reply, so the stored
# ``vlm_confidence`` describes the current label.
VLM_CLASS_SOURCES = frozenset(
    {
        VLM_CLASS_SOURCE,
        VLM_UNMATCHED_CLASS_SOURCE,
        VLM_NEW_CLASS_PENDING_CLASS_SOURCE,
        VLM_RECLASSIFIED_CLASS_SOURCE,
    }
)
# The VLM answers with a category; one shared category -> number table
# (the same one region text uses) so a client can rank or threshold it.
VLM_CATEGORY_SCORE: dict[str, float] = dict(VLM_TEXT_CONFIDENCE)

CLASS_SOURCE_ROLES: tuple[str, ...] = (
    'proposal',
    'low_conf',
    'model',
    'vlm',
    'vlm_unmatched',
    'vlm_new_class_pending',
    'vlm_reclassified',
    'cluster',
    'human',
    'merge',
    'label_import',
)

_FIXED_ENTRIES: tuple[tuple[str, str, str, str], ...] = (
    (DEFAULT_PROPOSAL_CLASS_SOURCE, 'Unclassified proposal', 'proposal', 'Proposal'),
    (VLM_CLASS_SOURCE, 'Labeled by the VLM', 'vlm', 'VLM'),
    (
        VLM_UNMATCHED_CLASS_SOURCE,
        'VLM answer not in the class registry',
        'vlm_unmatched',
        'VLM unmatched',
    ),
    (
        VLM_NEW_CLASS_PENDING_CLASS_SOURCE,
        'VLM proposed a new class',
        'vlm_new_class_pending',
        'New class?',
    ),
    (
        VLM_RECLASSIFIED_CLASS_SOURCE,
        'VLM answer matched after the registry grew',
        'vlm_reclassified',
        'VLM rematch',
    ),
    (CLUSTER_MAJORITY_CLASS_SOURCE, 'Cluster majority agreement', 'cluster', 'Cluster'),
    (HUMAN_CLASS_SOURCE, 'Labeled by a human', 'human', 'Human'),
    (HUMAN_MOVE_CLASS_SOURCE, 'Moved to a cluster by a human', 'human', 'Human move'),
    (CLASS_MERGE_CLASS_SOURCE, 'Relabeled by a class merge', 'merge', 'Merge'),
    (LABEL_IMPORT_CLASS_SOURCE, 'Imported label', 'label_import', 'Imported'),
)


def _entry(source_id: str, label: str, role: str, short_label: str) -> dict[str, str]:
    return {'id': source_id, 'label': label, 'role': role, 'short_label': short_label}


def class_source_catalog() -> list[dict[str, str]]:
    """``[{id, label, role, short_label}, ...]`` for every ``class_source`` value this
    deployment can write: the configured ingest profiles' values first,
    then the fixed writer values. Reads the ingest env on every call."""
    out: list[dict[str, str]] = []
    primary = ingest_primary_profile()
    p_model = primary.detector_model or primary.name
    out.append(_entry(f'{primary.name}_proposal', f'Proposed by {p_model}', 'proposal', 'Proposal'))
    if primary.assigns_class:
        out.append(
            _entry(
                f'{primary.name}_low_conf',
                f'{p_model} below confidence floor',
                'low_conf',
                'Low conf',
            )
        )
        out.append(_entry(f'{primary.name}_model', f'Classified by {p_model}', 'model', 'Model'))
    secondary = ingest_secondary_profile()
    if secondary is not None:
        s_model = secondary.detector_model or secondary.name
        out.append(
            _entry(f'{secondary.name}_model', f'Classified by {s_model}', 'model', 'Classifier')
        )
    seen = {e['id'] for e in out}
    out.extend(_entry(*fixed) for fixed in _FIXED_ENTRIES if fixed[0] not in seen)
    return out


def _raw_vlm_suggestion(src: dict[str, Any]) -> tuple[int | None, str | None]:
    if src.get('class_validated'):
        return None, None
    source = src.get('class_source')
    if source == VLM_NEW_CLASS_PENDING_CLASS_SOURCE:
        return None, (src.get('vlm_proposed_class') or None)
    if source in VLM_SUGGESTION_CLASS_SOURCES:
        class_id = src.get('class_id')
        if isinstance(class_id, int) and not isinstance(class_id, bool):
            return class_id, (src.get('class_name') or None)
    return None, None


def vlm_suggestion_dismissed(src: dict[str, Any]) -> bool:
    """True when the operator rejected exactly the VLM's current suggestion
    (``POST /crops/{id}/vlm_dismiss``); a different later one is live again."""
    class_id, name = _raw_vlm_suggestion(src)
    if class_id is not None:
        return src.get('vlm_dismissed_class_id') == class_id
    return name is not None and src.get('vlm_dismissed_class_name') == name


def class_confidence(src: dict[str, Any]) -> tuple[float | None, str | None]:
    """``(class_confidence, class_confidence_source)`` for a stored item:
    the confidence of the writer that set the current label.

    * a VLM source: the stored ``vlm_confidence`` category mapped through
      :data:`VLM_CATEGORY_SCORE`, source ``'vlm'`` (``None`` for a missing
      or unknown category, never a guess);
    * a configured classifier source: the stored ``confidence`` score,
      source ``'model'``;
    * anything else (human, merge, import, cluster vote, an unclassified
      proposal): ``(None, None)`` — no machine confidence applies.

    ``confidence`` itself is always the detector/classifier score, whatever
    wrote the label.
    """
    source = src.get('class_source')
    if source in VLM_CLASS_SOURCES:
        score = VLM_CATEGORY_SCORE.get(str(src.get('vlm_confidence') or ''))
        return (score, 'vlm') if score is not None else (None, None)
    if is_classifier_class_source(source):
        raw = src.get('confidence')
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return float(raw), 'model'
    return None, None


def unmatched_class_clear(current: dict[str, Any]) -> dict[str, Any]:
    """Fields to merge onto a ``vlm_unmatched`` write's ``update_doc`` so the
    row never keeps the class the VLM's answer just contradicted.

    A ``vlm_unmatched`` write means "the VLM read a label that isn't in the
    registry" -- keeping the item's *prior* ``class_id``/``class_name``
    (usually from a proposal/classifier/earlier VLM pass) would show a class
    the write's own ``class_source`` says wasn't matched. Clearing it is
    restorable: the caller's history snapshot (``record_class_snapshot`` /
    ``with_class_snapshot``) records the prior state before this lands.

    Returns ``{}`` when the item is human-owned or class-validated
    (:func:`src.services.curation.class_write_guard.class_write_locked`) --
    an unmatched VLM answer must never reset a locked item's class out from
    under a human decision.

    When ``current``'s ``cluster_id`` is a *class* cluster (``0 <=
    cluster_id < RESIDUAL_CLUSTER_ID_OFFSET``, i.e. it mirrored the class
    that's being cleared), also resets ``cluster_id=-1`` and
    ``cluster_subid=None`` so the residual clustering pass re-clusters the
    item instead of leaving it parked in a class cluster it no longer
    belongs to. A candidate cluster (``>= RESIDUAL_CLUSTER_ID_OFFSET``) is
    left alone -- the item's residual grouping isn't a class-cluster fact.
    """
    from src.services.curation.class_write_guard import class_write_locked

    if class_write_locked(current):
        return {}
    out: dict[str, Any] = {
        'class_id': None,
        'class_name': None,
        'class_detector': None,
        'class_detector_version': None,
        'class_labeler': None,
        'class_labeled_at': None,
    }
    cluster_id = current.get('cluster_id')
    if (
        isinstance(cluster_id, int)
        and not isinstance(cluster_id, bool)
        and 0 <= cluster_id < RESIDUAL_CLUSTER_ID_OFFSET
    ):
        out['cluster_id'] = -1
        out['cluster_subid'] = None
    return out


def vlm_suggestion(src: dict[str, Any]) -> tuple[int | None, str | None]:
    """``(class_id, class_name)`` the VLM suggests for a stored item.

    * ``vlm`` / ``vlm_reclassified`` and not ``class_validated``: the
      registry class the VLM chose (the item's current class).
    * ``vlm_new_class_pending``: ``(None, <proposed new class name>)``.
    * anything else, a human-validated class, or a suggestion the operator
      dismissed: ``(None, None)``.
    """
    if vlm_suggestion_dismissed(src):
        return None, None
    return _raw_vlm_suggestion(src)


__all__ = [
    'CLASS_MERGE_CLASS_SOURCE',
    'CLASS_SOURCE_ROLES',
    'HUMAN_LABEL_SOURCES',
    'HUMAN_MOVE_CLASS_SOURCE',
    'VLM_CATEGORY_SCORE',
    'VLM_CLASS_SOURCES',
    'VLM_NEW_CLASS_PENDING_CLASS_SOURCE',
    'VLM_RECLASSIFIED_CLASS_SOURCE',
    'VLM_SUGGESTION_CLASS_SOURCES',
    'VLM_UNMATCHED_CLASS_SOURCE',
    'HumanLabelSource',
    'class_confidence',
    'class_source_catalog',
    'unmatched_class_clear',
    'vlm_suggestion',
    'vlm_suggestion_dismissed',
]
