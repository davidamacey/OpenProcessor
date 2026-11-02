"""Tell the JSON-based annotation formats apart from the head of a file.

Auto-detection must not parse a multi-hundred-megabyte export, so it looks at
the first 64 KiB for each format's own keys. An explicit ``source.format``
skips this and parses the file for real."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from src.services.curation.dataset_import.native_common import read_head, split_of_dirname


if TYPE_CHECKING:
    from pathlib import Path

JsonKind = Literal['labelstudio', 'createml', 'labelme']

_DATA_KEY = re.compile(r'"data"\s*:')
_IMAGE_KEY = re.compile(r'"image"\s*:')
_ANNOTATIONS_KEY = re.compile(r'"annotations"\s*:')
_PREDICTIONS_KEY = re.compile(r'"predictions"\s*:')
_SHAPES_KEY = re.compile(r'"shapes"\s*:')


def sniff_json(path: Path) -> JsonKind | None:
    head = read_head(path).lstrip()
    if head.startswith('['):
        if _DATA_KEY.search(head) and (
            _ANNOTATIONS_KEY.search(head) or _PREDICTIONS_KEY.search(head)
        ):
            return 'labelstudio'
        if _IMAGE_KEY.search(head) and _ANNOTATIONS_KEY.search(head):
            return 'createml'
    elif head.startswith('{') and _SHAPES_KEY.search(head):
        return 'labelme'
    return None


def json_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob('*.json') if p.is_file())


def split_json_files(root: Path) -> list[tuple[Path, str | None]]:
    """``(file, split)`` for the JSON files in ``root`` (no split) and in its
    ``train``/``val``/``valid``/``validation``/``test`` child directories."""
    found: list[tuple[Path, str | None]] = [(p, None) for p in json_files(root)]
    for child in sorted(root.iterdir()):
        split = split_of_dirname(child.name)
        if child.is_dir() and split is not None:
            found.extend((p, split) for p in json_files(child))
    return found


def has_json_of_kind(root: Path, kind: JsonKind) -> bool:
    return root.is_dir() and any(sniff_json(p) == kind for p, _s in split_json_files(root))


__all__ = ['JsonKind', 'has_json_of_kind', 'json_files', 'sniff_json', 'split_json_files']
