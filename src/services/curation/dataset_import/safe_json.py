"""JSON text from an untrusted dataset, parsed so the only failure is a
:class:`ValueError`: ``json.loads`` raises ``RecursionError`` (not a
``ValueError``) for a document nested past the interpreter's stack."""

from __future__ import annotations

import json
from typing import Any


def loads_json(text: str) -> Any:
    """``json.loads(text)``; a malformed or too deeply nested document is a
    :class:`ValueError`."""
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise ValueError('nested too deeply') from exc


__all__ = ['loads_json']
