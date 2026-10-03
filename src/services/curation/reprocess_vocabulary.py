"""Operator-facing vocabulary of the unified reprocess, served on
``GET /config/vocabulary`` so a client renders labels instead of keeping
its own copy of them.

Every id list is derived from the type that validates it (the scope
Literal, the filter model's fields, the job statuses the job runner
writes), and ``tests/curation/test_reprocess_vocabulary.py`` asserts the
label tables below cover those sets exactly, so a new scope, filter field
or status cannot ship unlabelled.
"""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel

from src.services.curation.reprocess_models import ReprocessFilter, ReprocessScope


class VocabEntry(BaseModel):
    id: str
    label: str
    description: str


SCOPE_LABELS: dict[str, tuple[str, str]] = {
    'detect': (
        'Detection',
        'Re-run the object detector and merge its boxes with the existing ones.',
    ),
    'open_vocab': ('Open vocabulary', 'Re-run the open-vocabulary target prompts over the image.'),
    'region': ('Regions', 'Re-detect (or re-verify) the profile region on each selected item.'),
    'vlm': ('VLM labels', 'Regenerate the machine VLM labels and text reads.'),
    'embed': ('Embeddings', 'Recompute the stored image, crop and region vectors.'),
}

#: Selectors only a reprocess has (the shared item filter's own fields are
#: labelled by the list routes); an unlabelled field falls back to its id.
FILTER_FIELD_LABELS: dict[str, str] = {
    'region_status': 'Region status',
    'detector': 'Detector',
    'reason': 'Reason',
    'profile_not': 'Not produced by profile',
    'profile_revision_below': 'Profile revision below',
    'include_detected': 'Include already detected',
    'missing_status': 'Missing status',
    'missing_provenance': 'Missing provenance',
    'all_images': 'All images',
    'open_vocab_status': 'Open vocabulary status',
}

JOB_STATUS_LABELS: dict[str, str] = {
    'queued': 'Queued',
    'running': 'Running',
    'completed': 'Completed',
    'completed_with_errors': 'Completed with errors',
    'failed': 'Failed',
    'cancelled': 'Cancelled',
}

#: Why an item counts toward ``locked_skipped`` (the rule is
#: :mod:`src.services.curation.reprocess_locks`).
LOCK_REASON_LABELS: dict[str, tuple[str, str]] = {
    'human_label': (
        'Human label',
        'A person set or confirmed this label, so reprocess never overwrites it.',
    ),
    'validated': (
        'Validated',
        'A person validated this item or its region set, so there is no machine output to regenerate.',
    ),
    'imported': ('Imported', 'The label or box came from a dataset import and is trusted as-is.'),
    'test_holdout': (
        'Test holdout',
        'The item is frozen into the test holdout and is never modified.',
    ),
}


def scope_vocabulary() -> list[VocabEntry]:
    return [
        VocabEntry(id=s, label=SCOPE_LABELS[s][0], description=SCOPE_LABELS[s][1])
        for s in get_args(ReprocessScope)
    ]


def filter_field_vocabulary() -> list[VocabEntry]:
    return [
        VocabEntry(id=f, label=FILTER_FIELD_LABELS.get(f, f), description='')
        for f in ReprocessFilter.model_fields
    ]


def job_status_vocabulary() -> list[VocabEntry]:
    return [VocabEntry(id=k, label=v, description='') for k, v in JOB_STATUS_LABELS.items()]


def lock_reason_vocabulary() -> list[VocabEntry]:
    return [VocabEntry(id=k, label=v[0], description=v[1]) for k, v in LOCK_REASON_LABELS.items()]
