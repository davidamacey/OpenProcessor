"""One stable per-issue ``id`` for every issue list a response serves.

Two issues in one response can share a ``code`` (a class unmapped in two
sources, a field invalid twice) and differ only by subject, so a client keying
rows on ``code`` collides. :func:`stamp_issue_ids` gives each issue
``<code>[:<subject>]``, suffixed ``#2``, ``#3`` for a repeat, so ids are unique
within the response and stable for the same input order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def stamp_issue_ids(issues: Iterable[Any], subject_of: Callable[[Any], str | None]) -> None:
    """Set ``issue.id`` on every issue (any object with ``code`` and a writable ``id``)."""
    seen: dict[str, int] = {}
    for issue in issues:
        subject = subject_of(issue)
        base = f'{issue.code}:{subject}' if subject else str(issue.code)
        seen[base] = seen.get(base, 0) + 1
        issue.id = base if seen[base] == 1 else f'{base}#{seen[base]}'


__all__ = ['stamp_issue_ids']
