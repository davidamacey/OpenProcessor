"""The accuracy audit's pure parts: sample allocation, outcome and report.

The audit asks a human to label a stratified sample of machine-labelled crops
and compares what the detector said (``detector_class_name``), what the machine
label was when the crop was drawn, and what the human then said. Everything here
is arithmetic on class NAMES (compared in registry-name form); nothing reads an
index or a class id.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

AuditOutcome = Literal['agree', 'detector_wrong', 'vlm_wrong', 'both_wrong']
OUTCOMES: tuple[AuditOutcome, ...] = ('agree', 'detector_wrong', 'vlm_wrong', 'both_wrong')

_Z95 = 1.96

# Item fields the audit writes. ``audit_label_*`` snapshot the machine label the
# crop carried when it was drawn, so a later relabel cannot change what the audit
# judged; ``audit_outcome`` / ``audit_human_name`` are the human's verdict.
AUDIT_SAMPLE = 'audit_sample'
AUDIT_BATCH = 'audit_batch_id'
AUDIT_SAMPLED_AT = 'audit_sampled_at'
AUDIT_LABEL_NAME = 'audit_label_name'
AUDIT_LABEL_SOURCE = 'audit_label_source'
AUDIT_OUTCOME = 'audit_outcome'
AUDIT_HUMAN = 'audit_human_name'
AUDIT_FIELDS: dict[str, dict[str, str]] = {
    AUDIT_SAMPLE: {'type': 'boolean'},
    AUDIT_BATCH: {'type': 'keyword'},
    AUDIT_SAMPLED_AT: {'type': 'date'},
    AUDIT_LABEL_NAME: {'type': 'keyword'},
    AUDIT_LABEL_SOURCE: {'type': 'keyword'},
    AUDIT_OUTCOME: {'type': 'keyword'},
    AUDIT_HUMAN: {'type': 'keyword'},
}


def audit_outcome(*, detector: str, label: str, human: str) -> AuditOutcome:
    """Who was right: the detector, the machine label (``label``, the VLM's in the
    usual case), both or neither, judged against the human's class."""
    truth = normalize_class_name(human)
    detector_ok = normalize_class_name(detector) == truth
    label_ok = normalize_class_name(label) == truth
    if detector_ok and label_ok:
        return 'agree'
    if detector_ok:
        return 'vlm_wrong'
    if label_ok:
        return 'detector_wrong'
    return 'both_wrong'


def audit_outcome_fields(current: dict[str, Any], human_name: str) -> dict[str, Any]:
    """The verdict fields a human class write adds to a crop drawn into the audit
    (empty for any other crop). The latest human verdict wins, so a re-label
    after an undo restamps."""
    detector = current.get('detector_class_name')
    label = current.get(AUDIT_LABEL_NAME)
    if current.get(AUDIT_SAMPLE) is not True or not detector or not label:
        return {}
    return {
        AUDIT_OUTCOME: audit_outcome(detector=detector, label=label, human=human_name),
        AUDIT_HUMAN: normalize_class_name(human_name),
    }


def wilson_interval(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default). With no
    observations nothing is known, so the whole range."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def allocate_sample(sizes: Mapping[str, int], *, min_per_class: int, total: int) -> dict[str, int]:
    """How many crops to draw from each stratum.

    ``total`` is a hard budget. First every stratum is brought up toward
    ``min_per_class`` one crop per round (largest stratum first within a round, so
    a budget too small for every floor is shared evenly rather than starving the
    last strata); then what is left is spread over the remaining spare capacity in
    proportion to it (largest remainder). A stratum never gets more than it has.
    """
    order = sorted(sizes, key=lambda name: (-sizes[name], name))
    alloc = dict.fromkeys(order, 0)
    remaining = max(0, total)
    floors = {name: min(sizes[name], max(0, min_per_class)) for name in order}
    while remaining > 0:
        grew = False
        for name in order:
            if remaining > 0 and alloc[name] < floors[name]:
                alloc[name] += 1
                remaining -= 1
                grew = True
        if not grew:
            break
    spare = {name: sizes[name] - alloc[name] for name in order}
    spare_total = sum(spare.values())
    if remaining > 0 and spare_total > 0:
        if remaining >= spare_total:
            return {name: sizes[name] for name in order}
        quotas = {name: remaining * spare[name] / spare_total for name in order}
        for name in order:
            alloc[name] += int(quotas[name])
        leftover = remaining - sum(int(q) for q in quotas.values())
        by_fraction = sorted(order, key=lambda name: (-(quotas[name] % 1), order.index(name)))
        for name in by_fraction[:leftover]:
            alloc[name] += 1
    return alloc


@dataclass(frozen=True)
class AuditRow:
    """One audited crop after its human label."""

    detector: str
    label: str
    label_is_vlm: bool
    human: str
    outcome: AuditOutcome


def _precision_rows(
    groups: Mapping[str, Iterable[bool]], min_per_class: int
) -> list[dict[str, Any]]:
    out = []
    for name, group in groups.items():
        hits = list(group)
        n, correct = len(hits), sum(hits)
        low, high = wilson_interval(correct, n)
        out.append(
            {
                'name': name,
                'n': n,
                'correct': correct,
                'precision': correct / n if n else None,
                'ci_low': low,
                'ci_high': high,
                'insufficient_sample': n < min_per_class,
            }
        )
    return sorted(out, key=lambda row: (-row['n'], row['name']))


def build_report(rows: Iterable[AuditRow], *, min_per_class: int, pending: int) -> dict[str, Any]:
    """Per-class precision of the detector (grouped by its class) and of the VLM
    (grouped by its class, VLM-labelled rows only) with Wilson 95% intervals, the
    ``{detector class: {human class: n}}`` confusion matrix, the outcome counts and
    how many sampled crops still wait for a human. A class below ``min_per_class``
    audited crops is flagged ``insufficient_sample``."""
    rows = list(rows)
    detector: dict[str, list[bool]] = defaultdict(list)
    vlm: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    outcomes: Counter[str] = Counter()
    for row in rows:
        truth = normalize_class_name(row.human)
        detector_name = normalize_class_name(row.detector)
        detector[detector_name].append(detector_name == truth)
        if row.label_is_vlm:
            label_name = normalize_class_name(row.label)
            vlm[label_name].append(label_name == truth)
        confusion[detector_name][truth] += 1
        outcomes[row.outcome] += 1
    return {
        'audited': len(rows),
        'pending': pending,
        'min_per_class': min_per_class,
        'detector': _precision_rows(detector, min_per_class),
        'vlm': _precision_rows(vlm, min_per_class),
        'confusion': {name: dict(counts) for name, counts in confusion.items()},
        'outcomes': {name: outcomes.get(name, 0) for name in OUTCOMES},
    }
