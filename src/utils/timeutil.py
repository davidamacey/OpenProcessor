"""One UTC timestamp helper for every stored ``*_at`` string."""

from datetime import UTC, datetime


def now_iso() -> str:
    """Current UTC time as ISO-8601 with microseconds and a ``+00:00`` offset.

    Stored timestamps are compared lexically in places; every writer uses this
    one format so the strings order the same way they order in time.
    """
    return datetime.now(UTC).isoformat(timespec='microseconds')
