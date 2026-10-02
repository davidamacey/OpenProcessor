"""While OpenSearch is down the client logs a traceback per request; the
filter turns that into one backed-off warning line."""

from __future__ import annotations

import logging

from opensearchpy.exceptions import ConnectionError as OsConnectionError

from src.core.logging import OutageLogFilter


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _record(exc: BaseException | None, name: str = 'opensearch') -> logging.LogRecord:
    info = (type(exc), exc, None) if exc is not None else None
    return logging.LogRecord(name, logging.WARNING, __file__, 1, 'GET /x [status:N/A]', (), info)  # type: ignore[arg-type]


def test_a_connection_failure_is_one_line_then_backed_off() -> None:
    clock = _Clock()
    flt = OutageLogFilter(max_backoff_s=8.0, clock=clock)
    boom = OsConnectionError('N/A', 'refused', OSError('refused'))
    first = _record(boom)
    assert flt.filter(first) is True
    assert first.exc_info is None
    assert 'opensearch unreachable' in first.getMessage()
    assert flt.filter(_record(boom)) is False  # inside the 1 s window
    clock.t += 1.5
    assert flt.filter(_record(boom)) is True
    clock.t += 1.5
    assert flt.filter(_record(boom)) is False  # window doubled to 2 s
    clock.t += 100
    again = _record(boom)
    assert flt.filter(again) is True
    assert 'suppressed for 1s' in again.getMessage()  # a quiet stretch starts over


def test_other_records_pass_untouched() -> None:
    flt = OutageLogFilter()
    other = _record(ValueError('bad'))
    assert flt.filter(other) is True
    assert other.exc_info is not None
    plain = _record(None)
    assert flt.filter(plain) is True
    unrelated = _record(OsConnectionError('N/A', 'x', OSError()), name='uvicorn')
    assert flt.filter(unrelated) is True
    assert unrelated.exc_info is not None
