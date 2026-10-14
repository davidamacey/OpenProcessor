"""Cron expressions and interval checks for source schedules.

Five fields in UTC (``minute hour day-of-month month day-of-week``); each field is
``*``, a number, a range ``a-b``, a list, or any of those with a step (``*/15``,
``1-5/2``, ``10/20``). Day-of-week is 0-6 with 0 and 7 both Sunday. As in standard
cron, when both day-of-month and day-of-week are restricted a day matches if either does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class InvalidCronError(ValueError):
    """The expression is not a valid five-field cron."""


_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_NAMES = ('minute', 'hour', 'day-of-month', 'month', 'day-of-week')
_MAX_DAYS_AHEAD = 366 * 5


def _parse_field(text: str, low: int, high: int, name: str) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(','):
        body, _, step_text = part.partition('/')
        try:
            step = int(step_text) if step_text else 1
            if step < 1:
                raise ValueError
            if body == '*':
                start, end = low, high
            elif '-' in body:
                a, _, b = body.partition('-')
                start, end = int(a), int(b)
            else:
                start = int(body)
                end = high if step_text else start
        except ValueError:
            raise InvalidCronError(f'{name}: {part!r} is not a valid field') from None
        if start < low or end > high or start > end:
            raise InvalidCronError(f'{name}: {part!r} is outside {low}-{high}')
        values.update(range(start, end + 1, step))
    return frozenset(values)


@dataclass(frozen=True)
class Cron:
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    days_restricted: bool
    weekdays_restricted: bool

    @classmethod
    def parse(cls, expression: str) -> Cron:
        fields = expression.split()
        if len(fields) != 5:
            raise InvalidCronError(
                'a cron expression has five fields: minute hour day month weekday'
            )
        parsed = [
            _parse_field(text, low, high, name)
            for text, (low, high), name in zip(fields, _BOUNDS, _NAMES, strict=True)
        ]
        weekdays = frozenset(d % 7 for d in parsed[4])
        return cls(
            parsed[0],
            parsed[1],
            parsed[2],
            parsed[3],
            weekdays,
            fields[2] != '*',
            fields[4] != '*',
        )

    def _day_matches(self, day: datetime) -> bool:
        in_month = day.day in self.days
        in_week = (day.weekday() + 1) % 7 in self.weekdays
        if self.days_restricted and self.weekdays_restricted:
            return in_month or in_week
        if self.days_restricted:
            return in_month
        if self.weekdays_restricted:
            return in_week
        return True

    def next_after(self, after: datetime) -> datetime:
        """The first matching minute strictly after ``after`` (UTC)."""
        moment = after.astimezone(UTC).replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(_MAX_DAYS_AHEAD):
            if moment.month in self.months and self._day_matches(moment):
                for hour in sorted(h for h in self.hours if h >= moment.hour):
                    floor = moment.minute if hour == moment.hour else 0
                    minutes = sorted(m for m in self.minutes if m >= floor)
                    if minutes:
                        return moment.replace(hour=hour, minute=minutes[0])
            moment = (moment + timedelta(days=1)).replace(hour=0, minute=0)
        raise InvalidCronError('the expression never matches')


def schedule_due(cron: Cron, last_fire_at: datetime, now: datetime) -> bool:
    """A fire is due when the first match after the last fire is not in the future.
    Fires missed while the process was down collapse into one."""
    return cron.next_after(last_fire_at) <= now


def interval_due(last_started_at: float | None, interval_s: float, now: float) -> bool:
    return last_started_at is None or now - last_started_at >= interval_s
