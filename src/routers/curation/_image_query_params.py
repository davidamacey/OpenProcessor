"""The query parameters of the image list, declared once.

``GET /images`` and the image-locate and table routes take :data:`ImageViewQuery` on top of the
shared item filter, so the image-only parameters (source, position, near-duplicate state, label
state, hold-out, sort) have one set of names and one meaning.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Query

from src.routers.curation._api_error import api_error
from src.services.curation.image_view import (
    ImageCountsNotReadyError,
    ImageQueryError,
    ImageViewParams,
    LabelState,
    NearDupFilter,
)


def image_view_query(
    source: Annotated[str | None, Query(description='Ingest source tag.')] = None,
    in_holdout: Annotated[
        bool | None, Query(description='true = images with a held-out item; false = none.')
    ] = None,
    near_dup: Annotated[
        NearDupFilter | None,
        Query(
            description='member = in a group and not its representative; representative; '
            'none = in no group; collapsed = one image per group (its representative) plus every '
            'image in no group.'
        ),
    ] = None,
    label_state: Annotated[
        LabelState | None,
        Query(
            description='has_unlabeled, all_labeled (has items, none unlabeled) or no_items. '
            'Read from the stored per-image counts when every image in the list has them and no '
            'item filter or `include_test` narrows the items (they trail the items by a few '
            'seconds); otherwise resolved by a join on the items.'
        ),
    ] = None,
    sort: Annotated[
        str | None,
        Query(
            description='One of the `sort_options` ids the response serves. The unlabelled-count '
            'sorts are served once every image in the list has its stored counts; asked earlier '
            'they are refused with 422 `image_counts_not_ready`.'
        ),
    ] = None,
    include_test: Annotated[
        bool, Query(description='Count and draw held-out test items too.')
    ] = False,
) -> ImageViewParams:
    return ImageViewParams(
        source=source,
        in_holdout=in_holdout,
        near_dup=near_dup,
        label_state=label_state,
        sort=sort,
        include_test=include_test,
    )


ImageViewQuery = Annotated[ImageViewParams, Depends(image_view_query)]


def image_query_refusal(exc: ImageQueryError) -> HTTPException:
    """The refusal an image-list route answers a query it cannot serve: 422
    `image_counts_not_ready` for a stored-count sort before every image is counted, else 400."""
    if isinstance(exc, ImageCountsNotReadyError):
        return api_error(422, 'image_counts_not_ready', str(exc))
    return HTTPException(status_code=400, detail=str(exc))


__all__ = ['ImageViewQuery', 'image_query_refusal', 'image_view_query']
