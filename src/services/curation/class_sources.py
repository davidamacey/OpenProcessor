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

from typing import Any

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    DEFAULT_PROPOSAL_CLASS_SOURCE,
    HUMAN_CLASS_SOURCE,
    LABEL_IMPORT_CLASS_SOURCE,
    VLM_CLASS_SOURCE,
)


VLM_UNMATCHED_CLASS_SOURCE = 'vlm_unmatched'
VLM_NEW_CLASS_PENDING_CLASS_SOURCE = 'vlm_new_class_pending'
# Registry reclassification of `vlm_unmatched` items (prefix 'vlm').
VLM_RECLASSIFIED_CLASS_SOURCE = 'vlm_reclassified'
HUMAN_MOVE_CLASS_SOURCE = 'human_move'
CLASS_MERGE_CLASS_SOURCE = 'class_merge'

# Sources where the VLM picked a registry class that is still only a
# machine suggestion (until class_validated flips).
VLM_SUGGESTION_CLASS_SOURCES = frozenset({VLM_CLASS_SOURCE, VLM_RECLASSIFIED_CLASS_SOURCE})

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

_FIXED_ENTRIES: tuple[tuple[str, str, str], ...] = (
    (DEFAULT_PROPOSAL_CLASS_SOURCE, 'Unclassified proposal', 'proposal'),
    (VLM_CLASS_SOURCE, 'Labeled by the VLM', 'vlm'),
    (VLM_UNMATCHED_CLASS_SOURCE, 'VLM answer not in the class registry', 'vlm_unmatched'),
    (VLM_NEW_CLASS_PENDING_CLASS_SOURCE, 'VLM proposed a new class', 'vlm_new_class_pending'),
    (
        VLM_RECLASSIFIED_CLASS_SOURCE,
        'VLM answer matched after the registry grew',
        'vlm_reclassified',
    ),
    (CLUSTER_MAJORITY_CLASS_SOURCE, 'Cluster majority agreement', 'cluster'),
    (HUMAN_CLASS_SOURCE, 'Labeled by a human', 'human'),
    (HUMAN_MOVE_CLASS_SOURCE, 'Moved to a cluster by a human', 'human'),
    (CLASS_MERGE_CLASS_SOURCE, 'Relabeled by a class merge', 'merge'),
    (LABEL_IMPORT_CLASS_SOURCE, 'Imported label', 'label_import'),
)


def _entry(source_id: str, label: str, role: str) -> dict[str, str]:
    return {'id': source_id, 'label': label, 'role': role}


def class_source_catalog() -> list[dict[str, str]]:
    """``[{id, label, role}, ...]`` for every ``class_source`` value this
    deployment can write: the configured ingest profiles' values first,
    then the fixed writer values. Reads the ingest env on every call."""
    out: list[dict[str, str]] = []
    primary = ingest_primary_profile()
    p_model = primary.detector_model or primary.name
    out.append(_entry(f'{primary.name}_proposal', f'Proposed by {p_model}', 'proposal'))
    if primary.assigns_class:
        out.append(
            _entry(f'{primary.name}_low_conf', f'{p_model} below confidence floor', 'low_conf')
        )
        out.append(_entry(f'{primary.name}_model', f'Classified by {p_model}', 'model'))
    secondary = ingest_secondary_profile()
    if secondary is not None:
        s_model = secondary.detector_model or secondary.name
        out.append(_entry(f'{secondary.name}_model', f'Classified by {s_model}', 'model'))
    seen = {e['id'] for e in out}
    out.extend(_entry(*fixed) for fixed in _FIXED_ENTRIES if fixed[0] not in seen)
    return out


def vlm_suggestion(src: dict[str, Any]) -> tuple[int | None, str | None]:
    """``(class_id, class_name)`` the VLM suggests for a stored item.

    * ``vlm`` / ``vlm_reclassified`` and not ``class_validated``: the
      registry class the VLM chose (the item's current class).
    * ``vlm_new_class_pending``: ``(None, <proposed new class name>)``.
    * anything else, or a human-validated class: ``(None, None)``.
    """
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


__all__ = [
    'CLASS_MERGE_CLASS_SOURCE',
    'CLASS_SOURCE_ROLES',
    'HUMAN_MOVE_CLASS_SOURCE',
    'VLM_NEW_CLASS_PENDING_CLASS_SOURCE',
    'VLM_RECLASSIFIED_CLASS_SOURCE',
    'VLM_SUGGESTION_CLASS_SOURCES',
    'VLM_UNMATCHED_CLASS_SOURCE',
    'class_source_catalog',
    'vlm_suggestion',
]
