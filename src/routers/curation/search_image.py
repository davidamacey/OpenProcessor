"""``POST /curation/projects/{project}/search/image``: semantic search by a picture.

The sibling of ``search.py``'s text and similar routes: same filters, default exclusions and
page, gated by the same ``OP_SEMANTIC_SEARCH_ENABLED`` flag. The query is an uploaded picture
(or an ``image_id`` already in the project), optionally cut to a normalized ``box``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import File, Form, HTTPException, Query, Request, UploadFile

from src.config import get_curation_config
from src.core.exceptions import RequestRejectedError
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._page_models import SearchImageResponse
from src.routers.curation.search import _get_pe_encoder, _semantic_search_enabled
from src.schemas.curation.config_common import ApiErrorResponse
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search_image
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.item_filter import item_filter_clauses
from src.services.rejections import rejection


@router.post(
    '/search/image',
    response_model=None,
    responses={
        200: {'model': SearchImageResponse},
        400: {
            'model': ApiErrorResponse,
            'description': 'invalid_request, bad_box, image_unreadable',
        },
        404: {'model': ApiErrorResponse, 'description': 'image_not_found (image_id)'},
        413: {'model': ApiErrorResponse, 'description': 'upload_too_large (bytes or pixels)'},
        503: {'model': ApiErrorResponse, 'description': 'image_encoder_unavailable'},
    },
)
async def search_image(
    request: Request,
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    image: Annotated[UploadFile | None, File(description='The query picture.')] = None,
    image_id: Annotated[
        str | None, Form(description='An image already in this project, instead of an upload.')
    ] = None,
    box: Annotated[
        str | None,
        Form(description='Normalized "x1,y1,x2,y2": search by this region of the picture.'),
    ] = None,
    target: Annotated[
        SearchTarget,
        Query(description='`item` ranks crop vectors; `box` ranks per-box vectors.'),
    ] = SearchTarget.ITEM,
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=200),
    cluster_id: int | None = Query(None),
    tab: str | None = Query(None, description='Optional review-tab cohort id, as /search/text.'),
    date_from: str | None = Query(None, description='ISO date/datetime lower bound on created_at.'),
    date_to: str | None = Query(None, description='ISO date/datetime upper bound on created_at.'),
    min_blur_ratio: float | None = Query(None, ge=0.0),
    hide_near_duplicates: bool = False,
    min_score: float | None = Query(
        None, ge=0.0, le=1.0, description='Drop hits below this cosine-similarity score.'
    ),
    include_test: bool = False,
) -> dict[str, Any]:
    """kNN over the PE vectors using the encoded picture as the query.

    The picture goes through the same crop preprocessing the indexed vectors were made with.
    ``semantic_score`` is the similarity to the picture; with ``target=box`` each item also
    carries ``matched_box_id``. Nothing is stored.
    """
    if not _semantic_search_enabled():
        raise HTTPException(
            status_code=400,
            detail='semantic search is disabled (OP_SEMANTIC_SEARCH_ENABLED is set off; unset it or set 1 to enable)',
        )
    try:
        item_filter_clauses(item_filter)  # a malformed band is the caller's 400, not an outage
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if (image is None) == (image_id is None):
        raise rejection(400, 'invalid_request', 'send exactly one of image and image_id')
    data: bytes | None = None
    if image is not None:
        cap = get_curation_config().search_image_max_bytes
        data = await image.read(cap + 1)
        if len(data) > cap:
            raise rejection(413, 'upload_too_large', f'picture exceeds {cap} bytes')

    await ensure_indexes(opensearch)

    try:
        return await semantic_search_image.semantic_image_search(
            opensearch=opensearch,
            pe_encoder=_get_pe_encoder(request),
            data=data,
            image_id=image_id,
            box=semantic_search_image.parse_box(box),
            page=page,
            page_size=page_size,
            target=target,
            cluster_id=cluster_id,
            tab=tab,
            date_from=date_from,
            date_to=date_to,
            item_filter=item_filter,
            min_blur_ratio=min_blur_ratio,
            hide_near_duplicates=hide_near_duplicates,
            min_score=min_score,
            include_test=include_test,
        )
    except (HTTPException, RequestRejectedError):
        raise
    except Exception as exc:
        raise server_error(503, 'image search failed', exc) from exc
