"""The Table view: one page of detection rows or image rows with the requested columns.

Detection rows are `GET /crops` rows (`build_crop_list_query`, `serialize_item`) and image rows are
`GET /images` rows (`image_page`); the page differs only in its cells, which
``table_cells`` builds from the served wire dicts. Paging, the cursor and the filter are the
shared ones, so the table lists exactly what the list routes list under the same request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from src.config.curation_accessors import items_index
from src.schemas.curation.table import TableColumn, TablePage, TableRow, TableRows
from src.schemas.curation.views import SortOption
from src.services.curation import table_cells
from src.services.curation.cluster_cards import attach_item_display_nos
from src.services.curation.crop_listing import crop_list_body
from src.services.curation.hidden_counts import hidden_counts
from src.services.curation.image_join import join_truncated
from src.services.curation.image_view import ImageQuery, image_page
from src.services.curation.list_cursor import split_page
from src.services.curation.wire import serialize_item
from src.services.embedding_spaces.readers import item_state_fields, vector_source_excludes


if TYPE_CHECKING:
    from src.services.curation.crop_listing import CropListQuery
    from src.services.embedding_spaces.readers import WriteFields

LIMIT_MAX: Final = 200
LIMIT_DEFAULT: Final = 50
DEFAULT_SORT: Final[dict[str, str]] = {'detections': 'updated_at:desc', 'images': 'indexed_at:desc'}
_NO_ITEMS: Final = {
    'detections': 'No detections yet. Ingest images and run detection.',
    'images': 'No images yet. Ingest images.',
}


class TableQueryError(ValueError):
    """The request names a column or sort the table does not serve."""


def _sort_field(rows: TableRows) -> dict[str, str]:
    return (
        table_cells.DETECTION_SORT_FIELD if rows == 'detections' else table_cells.IMAGE_SORT_FIELD
    )


def sort_options(rows: TableRows) -> list[SortOption]:
    """One option per sortable column and direction."""
    return [
        {'id': f'{c.id}:{d}', 'label': f'{c.label} ({"ascending" if d == "asc" else "descending"})'}
        for c in table_cells.columns_of(rows)
        if c.sortable
        for d in ('asc', 'desc')
    ]


def underlying_sort(rows: TableRows, sort: str | None) -> tuple[str, str]:
    """``(table sort id, the list route's sort id)`` for ``sort`` (``<column>[:asc|desc]``)."""
    table_id = sort or DEFAULT_SORT[rows]
    column, _, direction = table_id.partition(':')
    direction = direction or 'asc'
    if column not in _sort_field(rows) or direction not in ('asc', 'desc'):
        msg = f'sort must be one of {[o["id"] for o in sort_options(rows)]}'
        raise TableQueryError(msg)
    return f'{column}:{direction}', f'{_sort_field(rows)[column]}:{direction}'


def resolve_columns(rows: TableRows, columns: str | None) -> list[TableColumn]:
    """The requested columns in order (the default set without ``columns``)."""
    known = {c.id: c for c in table_cells.columns_of(rows)}
    ids = (
        [c.strip() for c in columns.split(',') if c.strip()]
        if columns
        else table_cells.DEFAULT_COLUMNS[rows]
    )
    unknown = [c for c in ids if c not in known]
    if unknown or not ids:
        msg = f'unknown column: {", ".join(unknown) or "(none given)"}'
        raise TableQueryError(msg)
    return [known[c] for c in dict.fromkeys(ids)]


def _page(
    rows: TableRows,
    columns: list[TableColumn],
    items: list[TableRow],
    *,
    narrowed: bool,
    **rest: Any,
) -> TablePage:
    reason = None if items else ('no_match' if narrowed else 'no_items')
    return TablePage(
        rows=rows,
        columns=columns,
        items=items,
        sort_options=sort_options(rows),
        empty_reason=reason,
        empty_message=None
        if reason is None
        else ('Nothing matches the current filters.' if reason == 'no_match' else _NO_ITEMS[rows]),
        **rest,
    )


async def detections_table(
    opensearch: Any,
    listing: CropListQuery,
    columns: list[TableColumn],
    *,
    table_sort: str,
    cursor: str | None,
    limit: int,
) -> TablePage:
    """A page of the items ``listing`` matches (raises ``CursorError`` for a foreign cursor)."""
    body = crop_list_body(
        listing, cursor=cursor, limit=limit, vector_excludes=await vector_source_excludes()
    )
    resp = await opensearch.search(index=items_index(), body=body)
    hits, next_cursor = split_page(resp['hits']['hits'], sort_id=listing.sort_id, limit=limit)
    state_fields = await item_state_fields()
    wires = [
        serialize_item(h.get('_source') or {}, h.get('_id', ''), state_fields=state_fields)
        for h in hits
    ]
    await attach_item_display_nos(wires)
    ids = [c.id for c in columns]
    hidden = await hidden_counts(
        opensearch,
        listing.filter_clauses,
        include_test=False,
        include_excluded='excluded' in listing.item_filter.review_status,
    )
    return _page(
        'detections',
        columns,
        [
            TableRow(
                crop_id=str(w['crop_id']),
                image_id=str(w.get('image_id') or ''),
                thumbnail_url=str(w.get('thumbnail_url') or ''),
                cells=table_cells.detection_cells(w, ids),
            )
            for w in wires
        ],
        narrowed=not listing.item_filter.is_empty(),
        next_cursor=next_cursor,
        total=int(resp['hits']['total']['value']),
        limit=limit,
        sort_applied=table_sort,
        hidden=hidden,
        join_truncated=join_truncated(listing.image_join),
    )


async def images_table(
    opensearch: Any,
    q: ImageQuery,
    embedding: WriteFields,
    policy: str,
    columns: list[TableColumn],
) -> TablePage:
    """A page of the images ``q`` matches (``q.view.sort`` is the image list's sort id)."""
    page = await image_page(opensearch, q, embedding, policy)
    ids = [c.id for c in columns]
    return _page(
        'images',
        columns,
        [
            TableRow(
                crop_id=None,
                image_id=card.image_id,
                thumbnail_url=card.thumbnail_url,
                cells=table_cells.image_cells(card.model_dump(mode='json'), ids),
            )
            for card in page.items
        ],
        narrowed=q.narrowed(),
        next_cursor=page.next_cursor,
        total=page.total,
        limit=q.limit,
        sort_applied=page.sort_applied,
        hidden=page.hidden,
        join_truncated=page.join_truncated,
    )
