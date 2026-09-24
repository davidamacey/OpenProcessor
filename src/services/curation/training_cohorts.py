"""The training-cohort catalog: curated slices of the items index a
training cycle can draw from, as data (``GET {prefix}/training_cohorts``).

Each cohort names the endpoint + params that fetch its rows and the
backend cut-offs that define it, so a client renders the catalog without
keeping definitions, descriptions or thresholds of its own. Region
cohorts are the ``GET /regions/training_candidates`` modes; that endpoint
builds its queries from the same cut-offs and reports the same
descriptions as ``selection_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.services.detection.profile_registry import get_active_region_profile


LOW_CONFIDENCE_MAX = 0.75
"""Classifier confidence below this is "unsure" (the primary_low_conf
review band and the low_confidence cohort)."""

REGION_LOW_SCORE_MAX = 0.6
"""A verified primary-detector region scored below this is a high-loss
training row (``low_conf_correct``)."""

RowKind = Literal['crop', 'region']


@dataclass(frozen=True)
class CohortSpec:
    label: str
    description: str
    endpoint: str
    params: dict[str, Any] = field(default_factory=dict)
    cutoffs: dict[str, float] = field(default_factory=dict)
    row_kind: RowKind = 'crop'


CORE_COHORTS: dict[str, CohortSpec] = {
    'validated': CohortSpec(
        'Validated',
        'Human-confirmed labels — the default training pool',
        '/crops',
        {'label_validated': True},
    ),
    'needs_labeling': CohortSpec(
        'Needs labeling',
        'Assigned a class but never human-validated',
        '/crops',
        {'label_validated': False},
    ),
    'low_confidence': CohortSpec(
        'Low confidence',
        f'Classifier confidence below {LOW_CONFIDENCE_MAX}, or no classifier '
        'prediction — high-loss rows',
        '/crops',
        {'classifier_conf_lt': LOW_CONFIDENCE_MAX},
        {'classifier_conf_lt': LOW_CONFIDENCE_MAX},
    ),
    'model_disagreements': CohortSpec(
        'Model disagreements',
        'Validated items where the latest model disagrees with the human label',
        '/review/model_disagreements',
    ),
}


def _region(label: str, description: str, cutoffs: dict[str, float] | None = None) -> CohortSpec:
    return CohortSpec(
        label, description, '/regions/training_candidates', {}, cutoffs or {}, 'region'
    )


TRAINING_CANDIDATE_MODES: dict[str, CohortSpec] = {
    'detector_blind_spots': _region(
        'Detector blind spots',
        'Primary detector missed; secondary segmenter found the region, VLM confirmed',
    ),
    'low_conf_correct': _region(
        'Low-confidence hits',
        'Primary detector hit but with low confidence; useful as high-loss training rows',
        {'region_score_lt': REGION_LOW_SCORE_MAX},
    ),
    'disagreement': _region(
        'Detector disagreement',
        'Primary detector + secondary segmenter both fired; bboxes may disagree',
    ),
    'human_corrected': _region(
        'Human corrected',
        'Human corrected a prior detector output',
    ),
    'false_positives': _region(
        'False positives',
        'Human marked a detector box as a false positive (box retained)',
    ),
}


def cohort_catalog(class_id: int | None = None) -> list[dict[str, Any]]:
    """Core cohorts, then the region cohorts when a region profile is active.
    ``class_id`` is folded into every cohort's ``params``."""
    specs: dict[str, CohortSpec] = dict(CORE_COHORTS)
    if get_active_region_profile() is not None:
        specs.update(TRAINING_CANDIDATE_MODES)
    out = []
    for cohort_id, spec in specs.items():
        params: dict[str, Any] = {}
        if class_id is not None:
            params['class_id'] = class_id
        if spec.endpoint == '/regions/training_candidates':
            params['mode'] = cohort_id
        params.update(spec.params)
        out.append(
            {
                'id': cohort_id,
                'label': spec.label,
                'description': spec.description,
                'cutoffs': dict(spec.cutoffs),
                'endpoint': spec.endpoint,
                'params': params,
                'row_kind': spec.row_kind,
            }
        )
    return out


__all__ = [
    'CORE_COHORTS',
    'LOW_CONFIDENCE_MAX',
    'REGION_LOW_SCORE_MAX',
    'TRAINING_CANDIDATE_MODES',
    'CohortSpec',
    'cohort_catalog',
]
