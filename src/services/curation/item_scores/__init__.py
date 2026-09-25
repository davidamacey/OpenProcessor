"""Registry of curation-score overlays.

Mirrors ``cluster_methods/__init__.py``'s ``get_method``/``available_methods``
pattern, but for the *score* axis — see ``base.py`` for why this is a
distinct registry (overlays never write ``cluster_id``).
"""

from __future__ import annotations

from typing import Any

from src.services.curation.item_scores.base import CropScorer, ScoreResult
from src.services.curation.item_scores.mistakenness import MistakennessScorer
from src.services.curation.item_scores.near_dup import NearDupScorer
from src.services.curation.item_scores.uniqueness import UniquenessScorer


_SCORERS: dict[str, type[CropScorer]] = {
    UniquenessScorer.name: UniquenessScorer,
    NearDupScorer.name: NearDupScorer,
    MistakennessScorer.name: MistakennessScorer,
}

# Static metadata feeding strategy_registry.py's /curation/methods payload. Kept
# separate from _SCORERS so /curation/methods never has to instantiate a scorer
# (which may touch disk / faiss) just to describe it.
SCORER_METADATA: dict[str, dict[str, Any]] = {
    UniquenessScorer.name: {
        'label': 'Uniqueness',
        'requires_field': 'uniqueness_score',
        'writes': UniquenessScorer.writes,
    },
    NearDupScorer.name: {
        'label': 'Near-duplicate',
        'requires_field': 'dup_group_id',
        'writes': NearDupScorer.writes,
    },
    MistakennessScorer.name: {
        'label': 'Mistakenness',
        'requires_field': 'mistakenness_score',
        'writes': MistakennessScorer.writes,
    },
}


def get_scorer(name: str, **kwargs: Any) -> CropScorer:
    """Resolve a curation scorer by name. Raises ``ValueError`` on unknown names."""
    key = name.lower()
    if key not in _SCORERS:
        raise ValueError(f'unknown crop scorer {name!r}; valid: {sorted(_SCORERS)}')
    return _SCORERS[key](**kwargs)


def available_scorers() -> list[str]:
    return sorted(_SCORERS)


__all__ = [
    'SCORER_METADATA',
    'CropScorer',
    'ScoreResult',
    'available_scorers',
    'get_scorer',
]
