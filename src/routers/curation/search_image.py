"""``POST /curation/projects/{project}/search/image``: semantic search by a picture.

The sibling of ``search.py``'s text and similar routes: same filters, default exclusions and
page, gated by the same ``OP_SEMANTIC_SEARCH_ENABLED`` flag. The query is an uploaded picture
(or an ``image_id`` already in the project), optionally cut to a normalized ``box``.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import File, Form, HTTPException, Query, Request, UploadFile

from src.config import get_curation_config
from src.core.exceptions import RequestRejectedError
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.routers.curation._page_models import SearchImageResponse
from src.routers.curation._search_params import SearchWindowQuery, checked_filter_join
from src.routers.curation._space_refusals import space_refusals
from src.routers.curation.search import _get_pe_encoder
from src.schemas.curation.config_common import ApiErrorResponse
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search_image
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
    window: SearchWindowQuery,
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
    cluster_uid: uuid.UUID | None = Query(None),
    tab: str | None = Query(None, description='Optional review-tab cohort id, as /search/text.'),
) -> dict[str, Any]:
    """kNN over the PE vectors using the encoded picture as the query.

    The picture goes through the same crop preprocessing the indexed vectors were made with.
    ``semantic_score`` is the similarity to the picture; with ``target=box`` each item also
    carries ``matched_box_id``. Nothing is stored.
    """
    join = await checked_filter_join(opensearch, item_filter)

    if (image is None) == (image_id is None):
        raise rejection(400, 'invalid_request', 'send exactly one of image and image_id')
    data: bytes | None = None
    if image is not None:
        cap = get_curation_config().search_image_max_bytes
        data = await image.read(cap + 1)
        if len(data) > cap:
            raise rejection(413, 'upload_too_large', f'picture exceeds {cap} bytes')

    try:
        async with space_refusals():
            return await semantic_search_image.semantic_image_search(
                opensearch=opensearch,
                pe_encoder=_get_pe_encoder(request),
                data=data,
                image_id=image_id,
                box=semantic_search_image.parse_box(box),
                target=target,
                tab=tab,
                item_filter=item_filter,
                image_join=join,
                cluster_uid=None if cluster_uid is None else str(cluster_uid),
                **window.service_kwargs(),
            )
    except (HTTPException, RequestRejectedError):
        raise
    except Exception as exc:
        raise server_error(503, 'image search failed', exc) from exc
