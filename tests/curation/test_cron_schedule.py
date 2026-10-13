"""Five-field UTC cron: next fire times, the day-of-month / day-of-week rule, rollovers
and refusal of anything that is not a cron expression."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.services.curation.bulk_ingest.schedule import (
    Cron,
    InvalidCronError,
    interval_due,
    schedule_due,
)


def at(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.mark.parametrize(
    ('expr', 'after', 'expected'),
    [
        ('*/15 * * * *', at(2026, 10, 12, 10, 7), at(2026, 10, 12, 10, 15)),
        ('*/15 * * * *', at(2026, 10, 12, 10, 45), at(2026, 10, 12, 11, 0)),
        ('* * * * *', at(2026, 12, 31, 23, 59, 30), at(2027, 1, 1, 0, 0)),
        ('30 2 * * *', at(2026, 10, 12, 2, 30), at(2026, 10, 13, 2, 30)),
        ('30 2 * * *', at(2026, 10, 12, 2, 29, 59), at(2026, 10, 12, 2, 30)),
        ('0,30 8-10 * * *', at(2026, 10, 12, 10, 30), at(2026, 10, 13, 8, 0)),
        ('0 0 1 1 *', at(2026, 1, 1, 0, 0), at(2027, 1, 1, 0, 0)),
        ('0 0 29 2 *', at(2026, 3, 1, 0, 0), at(2028, 2, 29, 0, 0)),
        ('5/20 * * * *', at(2026, 10, 12, 10, 0), at(2026, 10, 12, 10, 5)),
        ('5/20 * * * *', at(2026, 10, 12, 10, 25), at(2026, 10, 12, 10, 45)),
    ],
)
def test_next_fire(expr: str, after: datetime, expected: datetime) -> None:
    assert Cron.parse(expr).next_after(after) == expected


def test_weekdays_with_sunday_as_zero_or_seven() -> None:
    friday = at(2026, 10, 9, 9, 0)
    assert friday.weekday() == 4
    assert Cron.parse('0 9 * * 1-5').next_after(friday) == at(2026, 10, 12, 9, 0)  # Monday
    sunday = at(2026, 10, 11, 9, 0)
    for expr in ('0 9 * * 0', '0 9 * * 7'):
        assert Cron.parse(expr).next_after(at(2026, 10, 5, 0, 0)) == sunday


def test_day_of_month_and_weekday_both_restricted_match_either() -> None:
    cron = Cron.parse('0 0 13 * 5')  # the 13th, or any Friday
    assert cron.next_after(at(2026, 10, 1, 0, 0)) == at(2026, 10, 2, 0, 0)  # a Friday
    assert cron.next_after(at(2026, 10, 9, 0, 0)) == at(2026, 10, 13, 0, 0)  # the 13th, a Tuesday


@pytest.mark.parametrize(
    'bad',
    [
        '',
        '* * * *',
        '* * * * * *',
        '61 * * * *',
        '* 24 * * *',
        '* * 0 * *',
        '* * * 13 *',
        '* * * * 8',
        '*/0 * * * *',
        '5-1 * * * *',
        'a b c d e',
        '1,,2 * * * *',
        '-1 * * * *',
    ],
)
def test_expressions_that_are_not_cron_are_refused(bad: str) -> None:
    with pytest.raises(InvalidCronError):
        Cron.parse(bad)


def test_an_expression_that_never_matches_is_reported() -> None:
    with pytest.raises(InvalidCronError, match='never'):
        Cron.parse('0 0 31 2 *').next_after(at(2026, 1, 1, 0, 0))


def test_fires_missed_while_down_collapse_into_one() -> None:
    cron = Cron.parse('0 * * * *')
    last = at(2026, 10, 12, 1, 0)
    now = last + timedelta(days=3)
    assert schedule_due(cron, last, now)
    # After the one catch-up fire, stamped at `now`, the next one is the next boundary.
    assert not schedule_due(cron, now, now + timedelta(minutes=30))


def test_interval_due() -> None:
    assert interval_due(None, 60, 1000.0)
    assert not interval_due(950.0, 60, 1000.0)
    assert interval_due(940.0, 60, 1000.0)
