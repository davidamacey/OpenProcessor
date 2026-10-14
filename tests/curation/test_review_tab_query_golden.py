"""Golden pin of every review-tab query body.

``build_tab_query`` is the one definition of what each ``/review/{tab}`` queue
matches; ``review_tab_catalog`` is what the frontend renders from. This test
dumps the full ``(must, must_not, reason)`` result (or the rejection raised) for
every tab x ``include_test`` x ``text`` x ``max_rank`` x ``region_status``
combination, plus the tab catalog with and without an active region profile,
and compares each canonical JSON digest with ``golden/review_tab_queries.json``.
Any change to a query body, its clause order or its reason text fails here.

Regenerate after an intended change:
``PYTHONPATH=. .venv/bin/python tests/curation/test_review_tab_query_golden.py --write``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

from src.core.exceptions import RequestRejectedError
from src.services.curation.review_queries import KNOWN_TABS, build_tab_query, review_tab_catalog


if TYPE_CHECKING:
    from collections.abc import Iterator


GOLDEN = Path(__file__).parent / 'golden' / 'review_tab_queries.json'
_REGION_STATUS_VALUES = (
    'all',
    'detected',
    'verify_rejected',
    'has_rejected_box',
)
_PROFILE = 'src.services.detection.profile_registry.get_active_region_profile'


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=False, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@contextmanager
def _region_profile(reads_text: bool | None, display_name: str | None = None) -> Iterator[None]:
    profile = (
        None
        if reads_text is None
        else SimpleNamespace(reads_text=reads_text, display_name=display_name)
    )
    with mock.patch(_PROFILE, return_value=profile):
        yield


def _query_result(tab: str, **kwargs: Any) -> Any:
    try:
        must, must_not, reason = build_tab_query(tab, **kwargs)
    except RequestRejectedError as exc:
        return {'rejected': [exc.status_code, exc.detail]}
    return {'must': must, 'must_not': must_not, 'reason': reason}


def dump_cases() -> dict[str, str]:
    statuses = [None, *sorted(_REGION_STATUS_VALUES), 'not_a_status']
    cases: dict[str, str] = {}
    for profile_state in ('text', 'no_text', 'none'):
        reads_text = {'text': True, 'no_text': False, 'none': None}[profile_state]
        with _region_profile(reads_text):
            for tab, include_test, text, max_rank, status in itertools.product(
                (*KNOWN_TABS, 'not_a_tab'),
                (True, False),
                (None, 'AB*?c'),
                (None, 1, 5),
                statuses,
            ):
                key = (
                    f'{profile_state}|{tab}|test={include_test}|text={text}'
                    f'|rank={max_rank}|status={status}'
                )
                cases[key] = _digest(
                    _query_result(
                        tab,
                        include_test=include_test,
                        text=text,
                        max_rank=max_rank,
                        region_status=status,
                    )
                )
            cases[f'{profile_state}|default_region_status'] = _digest(
                _query_result('regions', include_test=False, text='x', max_rank=None)
            )
    for label, reads_text, name in (
        ('catalog|none', None, None),
        ('catalog|text', True, None),
        ('catalog|named', False, 'Wheels'),
    ):
        with _region_profile(reads_text, name):
            cases[label] = _digest(review_tab_catalog())
    return cases


def test_every_review_tab_query_body_matches_the_golden_digests() -> None:
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    actual = dump_cases()
    assert actual.keys() == golden.keys()
    changed = sorted(k for k in golden if golden[k] != actual[k])
    assert not changed, f'{len(changed)} review query bodies changed, first: {changed[:5]}'


def test_the_golden_distinguishes_a_changed_clause() -> None:
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    key = 'text|outliers|test=False|text=None|rank=None|status=None'
    must, must_not, reason = build_tab_query(
        'outliers', include_test=False, text=None, max_rank=None
    )
    assert golden[key] == _digest({'must': must, 'must_not': must_not, 'reason': reason})
    mutated = {'must': must, 'must_not': must_not[:-1], 'reason': reason}
    assert golden[key] != _digest(mutated)


if __name__ == '__main__':
    if '--write' not in sys.argv:
        raise SystemExit('pass --write to regenerate the golden digests')
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(dump_cases(), indent=0, sort_keys=True) + '\n', encoding='utf-8')
