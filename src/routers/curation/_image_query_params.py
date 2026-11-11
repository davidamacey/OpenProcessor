"""The query parameters of the image list, declared once.

``GET /images`` and the image-locate and table routes take :data:`ImageViewQuery` on top of the
shared item filter, so the image-only parameters (source, position, near-duplicate state, label
state, hold-out, sort) have one set of names and one meaning.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from src.services.curation.image_view import (
    DEFAULT_SORT,
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
            'none = in no group.'
        ),
    ] = None,
    label_state: Annotated[
        LabelState | None,
        Query(
            description='has_unlabeled, all_labeled (has items, none unlabeled) or no_items. '
            'Resolved by a join on the items.'
        ),
    ] = None,
    sort: Annotated[
        str, Query(description='One of the `sort_options` ids the response serves.')
    ] = DEFAULT_SORT,
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

__all__ = ['ImageViewQuery', 'image_view_query']
