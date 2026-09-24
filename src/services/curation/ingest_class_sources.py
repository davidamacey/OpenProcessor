"""``class_source`` values derived from the configured ingest profiles.

Ingest stamps items from :mod:`src.config.ingest_profiles`:

* primary, non-assigning (``assigns_class`` false): ``{primary}_proposal``
* primary, assigning, below its floor: ``{primary}_low_conf``
* primary, assigning, confident: ``{primary}_model``
* secondary classifier hit: ``{secondary}_model``

Consumers that ask "did a classifier label this?" or "is this still an
unlabeled proposal?" read these helpers instead of hardcoding one
deployment's detector names.
"""

from __future__ import annotations

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile


# Non-ingest writers these sets fold in (unchanged literals: they name the
# writer, not a configured detector).
VLM_CLASS_SOURCE = 'vlm'
HUMAN_CLASS_SOURCE = 'human'
CLUSTER_MAJORITY_CLASS_SOURCE = 'cluster_majority_agreement'
CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE = 'classifier_vlm_agreement'
# ItemDoc's default before any detector stamps a source.
DEFAULT_PROPOSAL_CLASS_SOURCE = 'unlabeled_proposal'


def unlabeled_proposal_class_sources() -> frozenset[str]:
    """Sources of items the ingest detectors left without a class."""
    name = ingest_primary_profile().name
    return frozenset({f'{name}_proposal', f'{name}_low_conf'})


def classifier_class_sources() -> frozenset[str]:
    """Sources written by a configured classifier at ingest: the secondary
    (when configured) and the primary (only when it assigns classes)."""
    sources: set[str] = set()
    secondary = ingest_secondary_profile()
    if secondary is not None:
        sources.add(f'{secondary.name}_model')
    primary = ingest_primary_profile()
    if primary.assigns_class:
        sources.add(f'{primary.name}_model')
    return frozenset(sources)


def confident_class_sources() -> tuple[str, ...]:
    """Classifier + VLM + human sources: labels trusted enough to keep an
    item out of the residual clustering pool. Sorted for stable queries."""
    return tuple(sorted(classifier_class_sources() | {VLM_CLASS_SOURCE, HUMAN_CLASS_SOURCE}))


__all__ = [
    'CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE',
    'CLUSTER_MAJORITY_CLASS_SOURCE',
    'DEFAULT_PROPOSAL_CLASS_SOURCE',
    'HUMAN_CLASS_SOURCE',
    'VLM_CLASS_SOURCE',
    'classifier_class_sources',
    'confident_class_sources',
    'unlabeled_proposal_class_sources',
]
