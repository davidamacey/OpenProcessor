"""Wire ids and their stored form (15a section 1, "Wire ids").

A content-derived id travels as 32 lowercase hex digits (``image_id``, ``crop_id``, a face id) and is
stored in a ``uuid`` column: 16 bytes in the row and in every index instead of 33. PostgreSQL accepts the
hyphen-free form as input (https://www.postgresql.org/docs/18/datatype-uuid.html); this module is the
one place that renders it back, so the wire does not change.
"""

from __future__ import annotations

import re
import uuid


_WIRE = re.compile(r'^[0-9a-f]{32}$')


class WireIdError(ValueError):
    """A string that is not a wire id (never stored, never guessed at)."""


def to_uuid(wire_id: str) -> uuid.UUID:
    if not _WIRE.fullmatch(wire_id):
        msg = f'not a wire id (32 lowercase hex digits): {wire_id!r}'
        raise WireIdError(msg)
    return uuid.UUID(hex=wire_id)


def to_wire(value: uuid.UUID) -> str:
    return value.hex
