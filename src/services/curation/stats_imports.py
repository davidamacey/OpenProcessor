"""Dataset-import provenance in ``GET /stats/dataset``.

An imported label is validated by the user's own say-so, but it is not a
human review: ``validated_by_human`` / ``verified_by_human`` must not count
it, and a verifier that is "not human" must not read as the VLM. This module
owns the import-side aggregations and counts so ``stats.py`` stays one
dashboard assembly.
"""

from __future__ import annotations

from typing import Any

from src.config.region_source import CANDIDATE_IMPORT
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.region_boxes import box_query


#: The ``verifier`` an import stamps on a region it wrote
#: (``dataset_import.region_labels.region_state_fields``).
IMPORT_VERIFIER = CANDIDATE_IMPORT


def import_aggregations(fields: Any) -> dict[str, Any]:
    """The ``aggs`` entries ``stats_dataset`` adds for imports."""
    return {
        # Class labels an import wrote and asserted validated.
        'validated_by_import': {
            'filter': {
                'bool': {
                    'filter': [
                        {'term': {'class_validated': True}},
                        {'term': {'class_source': LABEL_IMPORT_CLASS_SOURCE}},
                    ]
                }
            }
        },
        # Regions an import wrote and asserted validated: the mirror of
        # ``regions_validated_by_human`` with the import as the actor.
        'regions_validated_by_import': {
            'filter': {
                'bool': {
                    'filter': [{'term': {fields.validated: True}}],
                    'should': [
                        box_query({'term': {f'{fields.boxes}.detector': CANDIDATE_IMPORT}}, fields),
                        {'term': {fields.verifier: IMPORT_VERIFIER}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        },
    }


def count_in(aggs: dict[str, Any], name: str) -> int:
    return int((aggs.get(name) or {}).get('doc_count', 0))


def verified_by_vlm(verifier_buckets: dict[str, int], by_human: int) -> int:
    """Verifiers that are neither the human nor an import: an import is not
    the VLM, so it must not read as one."""
    return sum(verifier_buckets.values()) - by_human - verifier_buckets.get(IMPORT_VERIFIER, 0)


def import_region_counts(
    aggs: dict[str, Any], verifier_buckets: dict[str, int], detector_buckets: dict[str, int]
) -> dict[str, int]:
    return {
        'by_import': detector_buckets.get(CANDIDATE_IMPORT, 0),
        'verified_by_import': verifier_buckets.get(IMPORT_VERIFIER, 0),
        'validated_by_import': count_in(aggs, 'regions_validated_by_import'),
    }


__all__ = [
    'IMPORT_VERIFIER',
    'count_in',
    'import_aggregations',
    'import_region_counts',
    'verified_by_vlm',
]
