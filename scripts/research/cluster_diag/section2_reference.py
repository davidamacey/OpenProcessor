"""The held-out rows quoted in the decision note's section 2, and a check against a fresh run.

The note's first numbers came from scratch scripts. The committed arms must reproduce them to
the three decimals the note quotes; ``reproduction`` lists every quoted cell beside its value.
"""

from __future__ import annotations

from typing import Any


# Held-out half, cut chosen on calibration: (family, target) -> view -> field.
SECTION2: dict[str, dict[str, dict[str, float]]] = {
    'global_average_ahc@0.05': {
        'all': {
            'pair_precision': 0.399,
            'pair_recall': 0.073,
            'bcubed_precision': 0.741,
            'bcubed_recall': 0.285,
            'bcubed_f': 0.412,
        },
        'non_person': {'bcubed_precision': 0.762, 'bcubed_recall': 0.381},
    },
    'within_detector_class@0.1': {
        'all': {
            'pair_precision': 0.924,
            'pair_recall': 0.129,
            'bcubed_precision': 0.943,
            'bcubed_recall': 0.357,
            'bcubed_f': 0.518,
        },
        'non_person': {'bcubed_precision': 0.925, 'bcubed_recall': 0.461},
    },
    'within_detector_class@0.2': {
        'all': {
            'pair_precision': 0.919,
            'pair_recall': 0.241,
            'bcubed_precision': 0.920,
            'bcubed_recall': 0.457,
            'bcubed_f': 0.611,
        },
        'non_person': {'bcubed_precision': 0.895, 'bcubed_recall': 0.561},
    },
    'within_detector_class@0.3': {'all': {'pair_precision': 0.906, 'pair_recall': 0.312}},
}
SECTION2_REFERENCES: dict[str, dict[str, dict[str, float]]] = {
    'ivf_refit': {
        'all': {
            'pair_precision': 0.734,
            'pair_recall': 0.029,
            'bcubed_precision': 0.792,
            'bcubed_recall': 0.187,
            'bcubed_f': 0.302,
        },
        'non_person': {'bcubed_precision': 0.814, 'bcubed_recall': 0.252},
    },
    'detector_class_only': {
        'all': {
            'pair_precision': 0.906,
            'pair_recall': 0.960,
            'bcubed_precision': 0.786,
            'bcubed_recall': 0.870,
            'bcubed_f': 0.826,
        },
        'non_person': {'bcubed_precision': 0.711, 'bcubed_recall': 0.828},
    },
}
ROUNDING = 0.0006  # the note quotes three decimals


def reproduction(fold1: dict[str, Any]) -> list[dict[str, Any]]:
    """Measured against quoted values for every cell of the section 2 table."""
    rows = []
    for name, views in SECTION2_REFERENCES.items():
        measured = fold1['references'][name]
        rows += _compare(name, views, measured)
    for name, views in SECTION2.items():
        family, target = name.split('@')
        cell = fold1['families'][family][target]
        rows += _compare(name, views, cell['views'])
    return rows


def _compare(
    name: str, expected: dict[str, dict[str, float]], measured: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            'row': name,
            'view': view,
            'field': field,
            'quoted': want,
            'measured': measured[view][field],
            'match': abs(measured[view][field] - want) <= ROUNDING,
        }
        for view, fields in expected.items()
        for field, want in fields.items()
    ]
