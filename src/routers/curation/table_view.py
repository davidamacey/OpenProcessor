"""`GET /table/columns` and `GET /table`: the Table view as a view model, paged by cursor."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

from fastapi import HTTPException, Query

from src.routers._errors import server_error
from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._crop_params import CropBrowse, resolve_listing
from src.routers.curation._cursor_params import CursorQ, cursor_refused
from src.routers.curation._image_query_params import ImageViewQuery
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.schemas.curation.table import TableColumns, TablePage, TableRows
from src.services.curation import table_cells
from src.services.curation.image_view import ImageQuery, ImageQueryError
from src.services.curation.list_cursor import CursorError
from src.services.curation.location_policy_store import get_location_policy
from src.services.curation.table_view import (
    LIMIT_DEFAULT,
    LIMIT_MAX,
    TableQueryError,
    detections_table,
    images_table,
    resolve_columns,
    underlying_sort,
)
from src.services.embedding_spaces.readers import item_write_fields


RowsQ = Annotated[TableRows, Query(description='Which rows the table lists.')]


@router.get('/table/columns', response_model=TableColumns, tags=['Table'])
async def table_columns(rows: RowsQ = 'detections') -> TableColumns:
    """The columns a table of `rows` can show, which ones sort, and the default set."""
    return TableColumns(
        rows=rows, columns=table_cells.columns_of(rows), default=table_cells.DEFAULT_COLUMNS[rows]
    )


@router.get('/table', response_model=TablePage, tags=['Table'])
async def table(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    view: ImageViewQuery,
    cursor: CursorQ = None,
    rows: RowsQ = 'detections',
    columns: Annotated[
        str | None,
        Query(description='Comma-separated column ids, in order; the served default when omitted.'),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=LIMIT_MAX)] = LIMIT_DEFAULT,
) -> TablePage:
    """A page of detection or image rows under the shared item filter. `sort` is
    `<column>:<asc|desc>` over the sortable columns (`sort_options`); `rows=images` also takes the
    image-only filters of `GET /images`. A bad cursor is 400 `cursor_invalid`, an unknown column
    422 `unknown_column`."""
    try:
        shown = resolve_columns(rows, columns)
        table_sort, list_sort = underlying_sort(rows, view.sort)
    except TableQueryError as exc:
        if str(exc).startswith('unknown column'):
            raise api_error(422, 'unknown_column', str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        if rows == 'images':
            q = ImageQuery(
                item_filter=item_filter,
                view=replace(view, sort=list_sort),
                cursor=cursor,
                limit=limit,
            )
            page = await images_table(
                opensearch, q, await item_write_fields(), await get_location_policy(), shown
            )
            return page.model_copy(update={'sort_applied': table_sort})
        listing = await resolve_listing(opensearch, item_filter, CropBrowse(sort=list_sort))
        return await detections_table(
            opensearch, listing, shown, table_sort=table_sort, cursor=cursor, limit=limit
        )
    except CursorError as exc:
        raise cursor_refused(exc) from exc
    except ImageQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
