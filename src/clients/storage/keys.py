"""Storage keys: ``<root>/<part>/<part>...``, relative, ``/``-separated, never leaving the root.

Postgres rows store these keys, never absolute paths (02c section 5). Every key is built with
:func:`make_key` and read back with :func:`parse_key`; both refuse the shapes that could escape
a root or name an ambiguous path.
"""

from __future__ import annotations

from typing import NamedTuple


ROOT_NAMES = frozenset({'source', 'projects', 'models', 'cache', 'upgrade'})


class ParsedKey(NamedTuple):
    root: str
    parts: tuple[str, ...]


def _check_part(part: str) -> str:
    if not part or part in {'.', '..'} or '/' in part or '\\' in part or '\x00' in part:
        raise ValueError(f'unsafe storage key part: {part!r}')
    return part


def _check_root(root: str) -> str:
    if root not in ROOT_NAMES:
        raise ValueError(f'unknown storage root: {root!r}')
    return root


def make_key(root: str, *parts: str) -> str:
    """Join a root name and path parts into a key; each part is one path segment."""
    return '/'.join([_check_root(root), *(_check_part(p) for p in parts)])


def parse_key(key: str) -> ParsedKey:
    """Split a key into its root and parts, refusing anything :func:`make_key` would refuse."""
    root, *parts = key.split('/')
    return ParsedKey(_check_root(root), tuple(_check_part(p) for p in parts))
