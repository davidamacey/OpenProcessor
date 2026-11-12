"""Wire models of the Table view: `GET .../table/columns` and `GET .../table`.

A row is a view model: every cell is already text, so the browser formats nothing. A cell's
`value` is the raw wire value, so what the cell shows can always be checked against the item or
image it came from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.curation.views import HiddenCounts, SortOption


TableRows = Literal['detections', 'images']
CellTone = Literal['validated', 'machine', 'unlabeled', 'holdout', 'rejected', 'muted']


class TableColumn(BaseModel):
    id: str
    label: str
    sortable: bool
    align: Literal['start', 'end', 'center']
    kind: Literal['thumbnail', 'text', 'number', 'date', 'badge']


class TableColumns(BaseModel):
    rows: TableRows
    columns: list[TableColumn]
    default: list[str] = Field(description='The column ids shown when `columns` is not given.')


class TableCell(BaseModel):
    text: str
    value: str | float | int | bool | None = Field(
        default=None, description='The raw wire value the text was made from.'
    )
    tone: CellTone | None = None


class TableRow(BaseModel):
    crop_id: str | None = Field(description='The item of a detection row; null on an image row.')
    image_id: str
    thumbnail_url: str
    cells: dict[str, TableCell]


class TablePage(BaseModel):
    rows: TableRows
    columns: list[TableColumn] = Field(description='The requested columns, in order.')
    items: list[TableRow]
    next_cursor: str | None = Field(
        default=None, description='Opaque; pass back as `cursor`. Null on the last page.'
    )
    total: int
    limit: int
    sort_applied: str
    sort_options: list[SortOption] = Field(
        description='One per sortable column and direction; `id` is `<column>:<asc|desc>`.'
    )
    hidden: HiddenCounts
    join_truncated: bool
    empty_reason: Literal['no_items', 'no_match'] | None = None
    empty_message: str | None = None
