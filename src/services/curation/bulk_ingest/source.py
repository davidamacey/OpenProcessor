"""Where a bulk run reads its image paths from: a folder or a manifest.

Both are paths inside the API container and must lie under a configured source
root (``OP_SOURCE_ROOT`` / ``OP_SOURCE_PATH_ALIASES``), the same rule every ingest
path obeys. Entries are streamed, never materialised, so a million-image source
costs no more memory than a small one.

A manifest is a text file: one absolute image path per line, or JSON Lines whose
objects carry a ``path``. Blank lines and ``#`` comments are skipped. An entry
outside the source roots is not dropped here: it reaches the runner and is
reported as a failure with its reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.services.curation.bulk_ingest.limits import DEFAULT_EXTENSIONS
from src.services.curation.bulk_ingest.walk import iter_image_paths
from src.services.curation.image_serving import UNSERVABLE_PATH_ERROR, is_servable_image_path


if TYPE_CHECKING:
    from collections.abc import Iterator


class InvalidSourceError(ValueError):
    """The folder or manifest cannot be used; the message says why."""


@dataclass(frozen=True)
class BulkSource:
    kind: Literal['folder', 'manifest']
    path: str
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS

    def to_dict(self) -> dict[str, object]:
        return {'kind': self.kind, 'path': self.path, 'extensions': list(self.extensions)}

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> BulkSource:
        kind = raw.get('kind')
        if kind not in ('folder', 'manifest'):
            raise InvalidSourceError(f'unknown source kind {kind!r}')
        extensions = raw.get('extensions') or DEFAULT_EXTENSIONS
        return cls(kind, str(raw['path']), tuple(str(e) for e in extensions))  # type: ignore[arg-type, union-attr]

    @property
    def key(self) -> str:
        """Identity of the source: the same folder (or manifest) and extensions."""
        return f'{self.kind}:{self.path}:{",".join(sorted(self.extensions))}'


def normalise_extensions(extensions: list[str] | None) -> tuple[str, ...]:
    if not extensions:
        return DEFAULT_EXTENSIONS
    return tuple(sorted({f'.{e.strip().lstrip(".").lower()}' for e in extensions if e.strip()}))


def validate_source(source: BulkSource) -> None:
    """Raise :class:`InvalidSourceError` unless the source exists, is the right
    kind of path and lies under a configured source root."""
    if not is_servable_image_path(source.path):
        raise InvalidSourceError(f'{source.path}: {UNSERVABLE_PATH_ERROR}')
    path = Path(source.path)
    if source.kind == 'folder' and not path.is_dir():
        raise InvalidSourceError(f'{source.path}: not a directory')
    if source.kind == 'manifest' and not path.is_file():
        raise InvalidSourceError(f'{source.path}: not a file')


def _manifest_entry(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith('#'):
        return None
    if text.startswith('{'):
        try:
            value = json.loads(text).get('path')
        except (ValueError, AttributeError):
            return text
        return value if isinstance(value, str) and value else None
    return text


def _manifest_paths(path: Path) -> Iterator[str]:
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            entry = _manifest_entry(line)
            if entry is not None:
                yield entry


def iter_source_paths(source: BulkSource, *, walk_workers: int = 8) -> Iterator[str]:
    """Every image path of ``source``, in discovery order. Blocking: iterate it
    off the event loop."""
    if source.kind == 'manifest':
        yield from _manifest_paths(Path(source.path))
        return
    for found in iter_image_paths(Path(source.path), frozenset(source.extensions), walk_workers):
        yield str(found)
