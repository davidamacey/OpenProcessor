"""The dataset-level "ready to train" checklist (``GET /quality/readiness``).

The same questions Train's preflight asks of a request, asked of the dataset alone: enough checked
examples per class and a fair test set (the preflight's own ``class_balance_check`` and
``test_holdout_check``, one implementation), machine labels nobody has checked, the accuracy
audit, near-duplicate groups, unlabelled objects on exportable photos and whether the newest
export is current. Each check links to the items that need work.

:func:`build_readiness` is pure over :class:`ReadinessFacts`; :func:`gather_facts` reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.domain.ingest_class_sources import DETECTOR_CLASS_SOURCE
from src.schemas.curation.quality import (
    ClassCoverage,
    DatasetReadiness,
    ReadinessCheck,
    ReadinessClassRow,
    ReadinessLink,
    ReadinessSeverity,
    Verdict,
)
from src.services.curation import review_tabs
from src.services.curation.audit import load_report
from src.services.curation.export_readiness import (
    export_generation_check,
    items_index_generation,
    validated_export_query,
)
from src.services.curation.image_join import item_image_ids
from src.services.curation.region_class import item_classes
from src.services.embedding_spaces.readers import item_write_fields
from src.services.training.preflight_checks import (
    PreflightCheck,
    class_balance_check,
    read_export_manifest,
    test_holdout_check,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import RegistryClassEntry

MAX_CLASSES = 5000
"""Distinct classes one coverage count reads."""
TOP_CLASSES = 10
"""Classes a check names."""

_NOT_EXCLUDED = {'bool': {'must_not': [{'term': {'class_excluded': True}}]}}
_VALIDATED = {'term': {'class_validated': True}}


@dataclass(frozen=True)
class ReadinessFacts:
    coverage: list[ClassCoverage]
    class_ids: dict[str, int]
    """The registry id of each covered class name, for the preflight's own check functions."""
    holdout_items: int
    """Items in the frozen test holdout."""
    audit: dict[str, Any]
    """``GET /audit/report``: audited, pending, detector precision per class."""
    near_dup_stale: list[str]
    """``level:reason`` per stale level of the near-duplicate groups; empty when current."""
    unlabeled_on_exportable: int | None
    """Unclassed items on photos that have a validated item; ``None`` when it could not be counted."""
    export: tuple[str, str] | None
    """``(severity, message)`` of the newest export's currency; ``None`` when there is no export."""


class _Names:
    """The registry lookup the preflight's check functions take, over the covered classes."""

    def __init__(self, names: dict[int, str]) -> None:
        self._names = names

    def get(self, class_id: int) -> Any:
        name = self._names.get(class_id)
        return None if name is None else type('Entry', (), {'class_name': name})()


def _link(section: str, query: str | None = None) -> ReadinessLink:
    return ReadinessLink(section=section, query=query)


def _severity(value: str) -> ReadinessSeverity:
    """A preflight severity as a readiness one: what could not be read is a warning, never ok."""
    return 'ok' if value == 'ok' else 'block' if value == 'block' else 'warn'


def _from_preflight(check: PreflightCheck, check_id: str, link: ReadinessLink) -> ReadinessCheck:
    rows = [
        ReadinessClassRow(
            class_name=str(r['name']), count=int(r.get('validated', r.get('test_crops', 0)))
        )
        for r in (check.detail or {}).get('classes', [])
    ]
    return ReadinessCheck(
        id=check_id,
        label=check.label,
        severity=_severity(check.severity),
        message=check.message,
        count=len(rows) or None,
        link=link if check.severity != 'ok' else None,
        classes=rows[:TOP_CLASSES],
    )


def _class_balance(f: ReadinessFacts, names: _Names) -> ReadinessCheck:
    ids = [f.class_ids[c.class_name] for c in f.coverage]
    counts = {f.class_ids[c.class_name]: c.validated for c in f.coverage}
    check = class_balance_check(ids, counts, names)
    return _from_preflight(check, 'class_balance', _link('workspace', 'view=clusters'))


def _test_set(f: ReadinessFacts, names: _Names) -> ReadinessCheck:
    if not f.holdout_items:
        return ReadinessCheck(
            id='test_holdout',
            label=PreflightCheck(name='test_holdout', severity='block', message='').label,
            severity='block',
            message='The test set is not frozen: no item is held out for testing.',
            count=0,
            link=_link('export'),
            classes=[],
        )
    ids = [f.class_ids[c.class_name] for c in f.coverage]
    test = {f.class_ids[c.class_name]: c.test for c in f.coverage}
    return _from_preflight(test_holdout_check(ids, test, names), 'test_holdout', _link('export'))


def _machine_labels(f: ReadinessFacts) -> ReadinessCheck:
    rows = sorted(
        (
            ReadinessClassRow(class_name=c.class_name, count=c.machine_only + c.detector_only)
            for c in f.coverage
            if c.machine_only + c.detector_only
        ),
        key=lambda r: (-r.count, r.class_name),
    )
    total = sum(r.count for r in rows)
    queue = review_tabs.QUEUE_LABELS['check_machine'][0]
    top = ', '.join(f'{r.class_name} {r.count:,}' for r in rows[:3])
    return ReadinessCheck(
        id='machine_labels',
        label='Machine labels not checked',
        severity='warn' if total else 'ok',
        message=(
            f'{total:,} machine labels are not checked ({top}). Check them in {queue}.'
            if total
            else 'Every machine label has been checked.'
        ),
        count=total,
        link=_link('workspace', 'view=review&queue=check_machine') if total else None,
        classes=rows[:TOP_CLASSES],
    )


def _audit(f: ReadinessFacts) -> ReadinessCheck:
    report = f.audit
    audited, pending = int(report['audited']), int(report['pending'])
    rows = [
        ReadinessClassRow(
            class_name=str(s['name']),
            count=int(s['n']),
            precision=s['precision'],
            ci_low=s['ci_low'],
            ci_high=s['ci_high'],
        )
        for s in report['detector']
    ]
    rows.sort(key=lambda r: (r.precision is None, r.precision or 0.0, r.class_name))
    if not audited and not pending:
        message, count = 'No accuracy audit yet: how often the detector is right is unknown.', 0
    elif pending:
        message, count = f'{pending:,} sampled crops are waiting for a label.', pending
    else:
        message, count = f'The detector was checked on {audited:,} crops.', audited
    done = bool(audited) and not pending
    return ReadinessCheck(
        id='accuracy_audit',
        label='Accuracy audit',
        severity='ok' if done else 'warn',
        message=message,
        count=count,
        link=None if done else _link('quality/audit'),
        classes=rows[:TOP_CLASSES],
    )


def _near_duplicates(f: ReadinessFacts) -> ReadinessCheck:
    stale = f.near_dup_stale
    message = (
        f'Near-duplicate groups are out of date ({", ".join(stale)}), so a duplicate may sit on '
        'both sides of the train and test split.'
        if stale
        else 'Near-duplicate groups are up to date; no group spans train and test.'
    )
    return ReadinessCheck(
        id='near_duplicates',
        label='Near-duplicate groups',
        severity='warn' if stale else 'ok',
        message=message,
        count=len(stale),
        link=_link('workspace', 'view=gallery&near_dup=collapsed') if stale else None,
        classes=[],
    )


def _unlabeled(f: ReadinessFacts) -> ReadinessCheck:
    n = f.unlabeled_on_exportable
    if n is None:
        message = 'Unlabelled objects on exportable photos could not be counted.'
    elif n:
        message = (
            f'{n:,} detected objects on photos that will be exported have no class: the model '
            'would learn them as background.'
        )
    else:
        message = 'Every detected object on an exportable photo has a class.'
    return ReadinessCheck(
        id='unlabeled_objects',
        label='Unlabelled objects',
        severity='ok' if n == 0 else 'warn',
        message=message,
        count=n,
        link=None if n == 0 else _link('workspace', 'view=detections&queue=to_label'),
        classes=[],
    )


def _export(f: ReadinessFacts) -> ReadinessCheck:
    label = PreflightCheck(name='export_generation', severity='ok', message='').label
    if f.export is None:
        severity, message = 'warn', 'Nothing has been exported yet.'
    else:
        severity, message = _severity(f.export[0]), f.export[1]
    return ReadinessCheck(
        id='export_current',
        label=label,
        severity=severity,
        message=message,
        count=None,
        link=_link('export') if severity != 'ok' else None,
        classes=[],
    )


def build_readiness(f: ReadinessFacts, *, now: datetime | None = None) -> DatasetReadiness:
    names = _Names({f.class_ids[c.class_name]: c.class_name for c in f.coverage})
    checks = [
        _class_balance(f, names),
        _test_set(f, names),
        _machine_labels(f),
        _audit(f),
        _near_duplicates(f),
        _unlabeled(f),
        _export(f),
    ]
    verdict: Verdict = (
        'blocked'
        if any(c.severity == 'block' for c in checks)
        else 'warn'
        if any(c.severity == 'warn' for c in checks)
        else 'ready'
    )
    return DatasetReadiness(
        verdict=verdict,
        checks=checks,
        coverage=f.coverage,
        computed_at=(now or datetime.now(UTC)).astimezone(UTC).isoformat(),
    )


def coverage_body() -> dict[str, Any]:
    """One search: per class name, how many items are validated, detector-only, machine-set and
    held out (items a person excluded are not counted)."""
    unchecked = {'bool': {'must_not': [_VALIDATED]}}
    return {
        'size': 0,
        'query': {
            'bool': {
                'filter': [{'exists': {'field': 'class_id'}}],
                'must_not': [{'term': {'class_excluded': True}}],
            }
        },
        'aggs': {
            'by_class': {
                'terms': {'field': 'class_name', 'size': MAX_CLASSES},
                'aggs': {
                    'validated': {'filter': _VALIDATED},
                    'unchecked': {'filter': unchecked},
                    'detector': {
                        'filter': {
                            'bool': {
                                'filter': [{'term': {'class_source': DETECTOR_CLASS_SOURCE}}],
                                'must_not': [_VALIDATED],
                            }
                        }
                    },
                    'test': {'filter': {'term': {'test_holdout': True}}},
                },
            }
        },
    }


def coverage_rows(
    response: dict[str, Any], classes: list[RegistryClassEntry]
) -> tuple[list[ClassCoverage], dict[str, int]]:
    """A row per active whole-item class, zeros for a class with no items, and the registry id of
    each."""
    buckets = {
        str(b['key']): b
        for b in ((response.get('aggregations') or {}).get('by_class') or {}).get('buckets') or []
    }
    rows: list[ClassCoverage] = []
    ids: dict[str, int] = {}
    for entry in item_classes(classes):
        found = buckets.get(entry.class_name, {})
        detector = int((found.get('detector') or {}).get('doc_count', 0))
        unchecked = int((found.get('unchecked') or {}).get('doc_count', 0))
        rows.append(
            ClassCoverage(
                class_name=entry.class_name,
                validated=int((found.get('validated') or {}).get('doc_count', 0)),
                machine_only=max(0, unchecked - detector),
                detector_only=detector,
                test=int((found.get('test') or {}).get('doc_count', 0)),
            )
        )
        ids[entry.class_name] = entry.class_id
    return rows, ids


async def _unlabeled_on_exportable(opensearch: AsyncOpenSearch) -> int | None:
    """Unclassed items on photos that carry a validated item: the objects an export leaves as
    background. ``None`` when there are more photos than one join can hold."""
    image_ids, truncated = await item_image_ids(
        opensearch, validated_export_query(await item_write_fields())
    )
    if truncated:
        return None
    if not image_ids:
        return 0
    query = {
        'bool': {
            'filter': [{'terms': {'image_id': image_ids}}],
            'must_not': [
                {'exists': {'field': 'class_id'}},
                {'term': {'class_excluded': True}},
                {'term': {'test_holdout': True}},
                {'exists': {'field': 'review_dismissed_at'}},
            ],
        }
    }
    return int((await opensearch.count(index=items_index(), body={'query': query}))['count'])


async def _export_state(opensearch: AsyncOpenSearch) -> tuple[str, str] | None:
    from src.services.curation.export import resolve_current_export_dir

    try:
        manifest = read_export_manifest(str(resolve_current_export_dir()))
    except FileNotFoundError:
        return None
    if not manifest:
        return None
    severity, message, _detail = export_generation_check(
        manifest, await items_index_generation(opensearch, items_index())
    )
    return severity, message


async def gather_facts(
    opensearch: AsyncOpenSearch, classes: list[RegistryClassEntry], near_dup_stale: list[str]
) -> ReadinessFacts:
    """Read every fact. ``near_dup_stale`` comes from PostgreSQL, so the route reads it."""
    response = await opensearch.search(index=items_index(), body=coverage_body())
    coverage, ids = coverage_rows(response, classes)
    held = await opensearch.count(
        index=items_index(),
        body={'query': {'bool': {'filter': [{'term': {'test_holdout': True}}]}}},
    )
    return ReadinessFacts(
        coverage=coverage,
        class_ids=ids,
        holdout_items=int(held['count']),
        audit=await load_report(opensearch),
        near_dup_stale=near_dup_stale,
        unlabeled_on_exportable=await _unlabeled_on_exportable(opensearch),
        export=await _export_state(opensearch),
    )


__all__ = [
    'MAX_CLASSES',
    'ReadinessFacts',
    'build_readiness',
    'coverage_body',
    'coverage_rows',
    'gather_facts',
]
