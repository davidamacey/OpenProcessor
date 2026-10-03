"""Served labels for the open-vocabulary pass's closed value sets.

One row per value of each ``Literal`` the wire carries (an image's
``open_vocab_status``, a hit's ``drop_reason``, a gate skip's ``reason``), so
a client renders them without a hardcoded copy. A value with no label here
raises and a stale label fails its test; the wire types are the same
``Literal`` objects, so the served vocabulary, the contract enums and the code
cannot disagree.
"""

from __future__ import annotations

from typing import get_args

from src.services.curation.open_vocab_fields import OpenVocabStatus
from src.services.detection.open_vocab_select import DropReason
from src.services.detection.segmenter_gate import GateReason


_STATUS_LABELS = {
    'pending': 'Queued: waiting for the open-vocabulary pass',
    'done': 'Done',
    'skipped_gate': 'Skipped: every target was gated off',
    'failed': 'Failed: the segmenter could not be reached or errored',
}
_DROP_REASON_LABELS = {
    'below_min_score': 'Score below the target minimum',
    'too_small': 'Box smaller than the target minimum area',
    'too_large': 'Box larger than the target maximum area',
    'nms': 'Overlapped a higher-scoring hit of the same target',
    'over_max': 'Beyond the target maximum instances per image',
    'cross_target_nms': 'Overlapped a higher-scoring hit of another target',
    'agree_existing': 'An existing box of the same class already covers it',
    'skipped_locked': 'Overlaps a human-owned box, which is never edited',
}
_GATE_REASON_LABELS = {
    'disabled': 'The target is disabled',
    'no_parent_class': 'The image holds no item of the target parent classes',
    'vlm_no': 'The vision model pre-check said the prompt is not visible',
    'hit_rate': 'Recent runs found nothing: sampling instead of running',
}


def _rows(values: tuple[str, ...], labels: dict[str, str]) -> list[dict[str, str]]:
    return [{'value': v, 'label': labels[v]} for v in values]


def open_vocab_vocabulary() -> dict[str, list[dict[str, str]]]:
    return {
        'statuses': _rows(get_args(OpenVocabStatus), _STATUS_LABELS),
        'drop_reasons': _rows(get_args(DropReason), _DROP_REASON_LABELS),
        'gate_reasons': _rows(get_args(GateReason), _GATE_REASON_LABELS),
    }


__all__ = ['open_vocab_vocabulary']
