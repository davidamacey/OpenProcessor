"""The columns of the Table view and the cells built from the wire dicts of items and images.

A cell is made from the same dict the list routes serve (`serialize_item`, `ImageCard`), so its
`value` is the wire value and its `text` is the only thing formatted here. Formatting lives in
:func:`format_confidence` and :func:`format_date`; the browser shows `text` as it is.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any, Final

from src.schemas.curation.table import TableCell, TableColumn, TableRows


if TYPE_CHECKING:
    from collections.abc import Callable

Cell = TableCell
_EMPTY: Final = '—'


def format_confidence(value: Any) -> str:
    return _EMPTY if value is None else f'{float(value) * 100:.0f}%'


def format_date(value: Any) -> str:
    """`YYYY-MM-DD HH:MM` of an ISO timestamp (UTC or wall-clock); the dash when there is none."""
    if not value:
        return _EMPTY
    return str(value).replace('T', ' ')[:16]


def _col(
    id_: str, label: str, kind: str, *, sortable: bool = False, align: str | None = None
) -> TableColumn:
    default_align = 'end' if kind == 'number' else 'center' if kind == 'thumbnail' else 'start'
    return TableColumn.model_validate(
        {
            'id': id_,
            'label': label,
            'sortable': sortable,
            'align': align or default_align,
            'kind': kind,
        }
    )


DETECTION_COLUMNS: Final[list[TableColumn]] = [
    _col('thumbnail', 'Crop', 'thumbnail'),
    _col('class', 'Class', 'text', sortable=True),
    _col('detector_class', 'Detector class', 'text', sortable=True),
    _col('detector_confidence', 'Detector confidence', 'number', sortable=True),
    _col('label_source', 'Label source', 'badge'),
    _col('validated', 'Validated', 'badge', sortable=True),
    _col('cluster', 'Cluster', 'text'),
    _col('image_file', 'Image', 'text'),
    _col('taken_at', 'Taken', 'date'),
    _col('area', 'Area', 'number', sortable=True),
    _col('rank', 'Rank in image', 'number', sortable=True),
    _col('clarity', 'Clarity', 'number', sortable=True),
    _col('near_dup', 'Near-duplicate', 'badge'),
    _col('embedding_state', 'Embedding', 'badge'),
    _col('updated_at', 'Updated', 'date', sortable=True),
]
IMAGE_COLUMNS: Final[list[TableColumn]] = [
    _col('thumbnail', 'Image', 'thumbnail'),
    _col('file', 'File', 'text', sortable=True),
    _col('taken_at', 'Taken', 'date', sortable=True),
    _col('located', 'Located', 'badge'),
    _col('items', 'Items', 'number'),
    _col('labeled', 'Labeled', 'number'),
    _col('unlabeled', 'Unlabeled', 'number'),
    _col('near_dup', 'Near-duplicate', 'badge'),
    _col('indexed_at', 'Indexed', 'date', sortable=True),
]
DEFAULT_COLUMNS: Final[dict[str, list[str]]] = {
    'detections': [
        'thumbnail',
        'class',
        'detector_confidence',
        'label_source',
        'validated',
        'image_file',
        'updated_at',
    ],
    'images': ['thumbnail', 'file', 'taken_at', 'items', 'unlabeled', 'indexed_at'],
}

# The stored sort field behind each sortable column: the `/crops` sort field for detections, the
# `/images` sort id for images (the table's sort id is `<column>:<asc|desc>`).
DETECTION_SORT_FIELD: Final[dict[str, str]] = {
    'class': 'class_name',
    'detector_class': 'detector_class_name',
    'detector_confidence': 'detector_confidence',
    'validated': 'validated',
    'area': 'crop_area_norm',
    'rank': 'crop_rank_in_image',
    'clarity': 'blur_lap_ratio',
    'updated_at': 'updated_at',
}
IMAGE_SORT_FIELD: Final[dict[str, str]] = {
    'file': 'file_name',
    'taken_at': 'taken_at',
    'indexed_at': 'indexed_at',
}


def columns_of(rows: TableRows) -> list[TableColumn]:
    return DETECTION_COLUMNS if rows == 'detections' else IMAGE_COLUMNS


def _text(value: Any, tone: str | None = None) -> Cell:
    text = _EMPTY if value in (None, '') else str(value)
    return Cell.model_validate({'text': text, 'value': value or None, 'tone': tone})


def _number(value: Any, text: str | None = None) -> Cell:
    return Cell(text=_EMPTY if value is None else text or str(value), value=value)


def _file(path: Any) -> Cell:
    return _text(posixpath.basename(str(path)) if path else None)


def _near_dup_text(in_group: bool, representative: bool | None) -> Cell:
    if not in_group:
        return Cell(text='None', value=False, tone='muted')
    return Cell(text='Representative' if representative else 'Member', value=True, tone='machine')


def _class_cell(w: dict[str, Any]) -> Cell:
    name = w.get('class_name')
    if not name:
        return Cell(text='Unlabeled', tone='unlabeled')
    if w.get('class_excluded'):
        return Cell(text=name, value=name, tone='rejected')
    return Cell(text=name, value=name, tone='validated' if w.get('class_validated') else 'machine')


def _taken(w: dict[str, Any]) -> Cell:
    capture = w.get('capture') or {}
    value = capture.get('taken_utc') or capture.get('taken_local')
    return Cell(text=format_date(value), value=value)


def _cluster(w: dict[str, Any]) -> Cell:
    cluster = w.get('cluster')
    if not cluster:
        return Cell(text=_EMPTY, tone='muted')
    number = cluster.get('display_no')
    return Cell(
        text=f'#{number}' if number is not None else str(cluster.get('uid')),
        value=cluster.get('uid'),
    )


def _embedding(w: dict[str, Any]) -> Cell:
    states = sorted({str(s) for s in (w.get('embedding_states') or {}).values()})
    return _text(', '.join(states) or None)


_DETECTION_CELLS: Final[dict[str, Callable[[dict[str, Any]], Cell]]] = {
    'thumbnail': lambda w: Cell(text='', value=w.get('thumbnail_url')),
    'class': _class_cell,
    'detector_class': lambda w: _text(w.get('detector_class_name')),
    'detector_confidence': lambda w: _number(
        w.get('detector_confidence'), format_confidence(w.get('detector_confidence'))
    ),
    'label_source': lambda w: _text(w.get('label_source')),
    'validated': lambda w: Cell(
        text='Validated' if w.get('class_validated') else 'Not validated',
        value=bool(w.get('class_validated')),
        tone='validated' if w.get('class_validated') else 'muted',
    ),
    'cluster': _cluster,
    'image_file': lambda w: _file(w.get('image_path') or w.get('source_image_path')),
    'taken_at': _taken,
    'area': lambda w: _number(w.get('crop_area_norm'), format_confidence(w.get('crop_area_norm'))),
    'rank': lambda w: _number(w.get('crop_rank_in_image')),
    'clarity': lambda w: _number(
        w.get('blur_lap_ratio'),
        None if w.get('blur_lap_ratio') is None else f'{float(w["blur_lap_ratio"]):.2f}',
    ),
    'near_dup': lambda w: _near_dup_text(
        w.get('near_dup_group_id') is not None, w.get('near_dup_is_representative')
    ),
    'embedding_state': _embedding,
    'updated_at': lambda w: Cell(text=format_date(w.get('updated_at')), value=w.get('updated_at')),
}
_IMAGE_CELLS: Final[dict[str, Callable[[dict[str, Any]], Cell]]] = {
    'thumbnail': lambda w: Cell(text='', value=w.get('thumbnail_url')),
    'file': lambda w: _text(w.get('file_name')),
    'taken_at': lambda w: Cell(text=format_date(w.get('taken_at')), value=w.get('taken_at')),
    'located': lambda w: Cell(
        text='Yes' if w.get('has_location') else 'No', value=bool(w.get('has_location'))
    ),
    'items': lambda w: _number(w['counts']['items']),
    'labeled': lambda w: _number(w['counts']['labeled']),
    'unlabeled': lambda w: _number(w['counts']['unlabeled']),
    'near_dup': lambda w: _near_dup_text(
        bool(w.get('near_dup')), (w.get('near_dup') or {}).get('is_representative')
    ),
    'indexed_at': lambda w: Cell(text=format_date(w.get('indexed_at')), value=w.get('indexed_at')),
}


def detection_cells(wire: dict[str, Any], columns: list[str]) -> dict[str, TableCell]:
    """The cells of one item (its `serialize_item` dict) for ``columns``."""
    return {c: _DETECTION_CELLS[c](wire) for c in columns}


def image_cells(wire: dict[str, Any], columns: list[str]) -> dict[str, TableCell]:
    """The cells of one image (its `ImageCard` as a dict) for ``columns``."""
    return {c: _IMAGE_CELLS[c](wire) for c in columns}
