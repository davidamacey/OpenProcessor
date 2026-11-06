"""The SSE resume cursor: ``(xid, id)`` rendered as the event id ``<xid>-<id>``."""

from __future__ import annotations

from dataclasses import dataclass


class InvalidCursorError(ValueError):
    """A ``Last-Event-ID`` that is not ``<xid>-<id>``."""


@dataclass(frozen=True, order=True)
class Cursor:
    xid: int
    id: int

    def format(self) -> str:
        return f'{self.xid}-{self.id}'

    @classmethod
    def parse(cls, raw: str) -> Cursor:
        xid, sep, last = raw.partition('-')
        if (
            not sep
            or not xid.isascii()
            or not last.isascii()
            or not xid.isdigit()
            or not last.isdigit()
        ):
            raise InvalidCursorError(raw)
        return cls(int(xid), int(last))


START = Cursor(0, 0)
