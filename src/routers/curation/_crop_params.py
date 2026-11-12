"""The query parameters of ``GET /crops`` and ``GET /crops/locate`` beyond the shared item filter.

Declared once: the list and the locate route take the same filter and sort, so a located index is
the index in the list the same request would serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, HTTPException, Query

from src.routers.curation._review_params import UnsetIfEmpty
from src.services.curation.crop_listing import CropListQuery, build_crop_list_query
from src.services.curation.diverse_limits import DIVERSE_BROWSE_MAX_K
from src.services.curation.image_join import resolve_filter_join
from src.services.curation.region_text_search import TextMatch, same_text_key
from src.services.embedding_spaces.readers import item_write_fields


if TYPE_CHECKING:
    from src.services.curation.item_filter import ItemFilter


@dataclass(frozen=True)
class CropBrowse:
    """What a crops request asks for besides the item filter, the cursor and the limit."""

    sort: str | None = None
    label_source: str | None = None
    class_source: str | None = None
    label_validated: bool | None = None
    import_id: str | None = None
    dataset_split: str | None = None
    on_negative_frame: bool | None = None
    proposed_by_import: bool | None = None
    source: str | None = None
    open_vocab_set: str | None = None
    source_prompt: str | None = None
    needs_new_class: bool | None = None
    review_dismissed: bool | None = None
    include_test: bool = False
    include_excluded: bool = False
    min_blur_ratio: float | None = None
    classifier_conf_lt: float | None = None
    item_text: str | None = None
    region_text: str | None = None
    region_text_match: TextMatch = 'exact'
    same_text_as: str | None = None
    order: str = 'default'
    k: int | None = None


def crop_browse(
    sort: Annotated[
        str | None,
        Query(
            description=(
                "'<field>[:asc|desc]', default 'updated_at:desc'. The response's `sort_options` "
                'lists every id with its label. Ignored by order=outliers|diverse.'
            )
        ),
    ] = None,
    label_source: str | None = None,
    class_source: str | None = None,
    label_validated: bool | None = None,
    import_id: Annotated[
        str | None,
        Query(description='Items a dataset import labeled (`import_ids`) or proposed.'),
    ] = None,
    dataset_split: Annotated[
        str | None,
        UnsetIfEmpty,
        Query(description='The split a dataset import filed the frame under.'),
    ] = None,
    on_negative_frame: Annotated[
        bool | None,
        UnsetIfEmpty,
        Query(description='true = machine items on an imported reviewed-negative frame.'),
    ] = None,
    proposed_by_import: Annotated[
        bool | None,
        Query(description='true = items an import created from the detector; false = the rest.'),
    ] = None,
    source: Annotated[str | None, Query(description='Ingest source tag (wire `source`).')] = None,
    open_vocab_set: Annotated[
        str | None, Query(description='Items the full-image pass wrote for this prompt set.')
    ] = None,
    source_prompt: Annotated[
        str | None, Query(description='Items the full-image pass found with this exact prompt.')
    ] = None,
    needs_new_class: bool | None = None,
    review_dismissed: Annotated[
        bool | None, Query(description='true = only items hidden from review.')
    ] = None,
    include_test: bool = False,
    include_excluded: bool = False,
    min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    classifier_conf_lt: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    item_text: Annotated[
        str | None,
        Query(
            max_length=200,
            description=(
                'Text read on the item crop: every word must be a case-insensitive '
                'prefix of a stored item text token.'
            ),
        ),
    ] = None,
    region_text: Annotated[
        str | None,
        Query(
            max_length=200,
            description=(
                'Plate text on a box: the item has a box whose folded text matches this '
                '(letters and digits only, confusables folded), as `region_text_match` says.'
            ),
        ),
    ] = None,
    region_text_match: Annotated[
        TextMatch,
        Query(description='How `region_text` matches: `exact`, `fuzzy` (one edit) or `fragment`.'),
    ] = 'exact',
    same_text_as: Annotated[
        str | None,
        Query(
            description=(
                "`<item_id>:<box_id>`: items with a box whose folded text equals that box's. "
                'Exclusive with `region_text`.'
            ),
        ),
    ] = None,
    order: Annotated[
        str,
        Query(
            description=(
                "'outliers' ranks a cluster's members farthest-from-centroid first; "
                "'core_first' best-scored-first by the partition's cluster_score "
                '(both need cluster_uid). '
                "'diverse' ranks the matched pool by k-center-greedy coverage "
                '(gated on OP_SELECT_DIVERSE_ENABLED; behaves like an unrecognized '
                'order value when the flag is off).'
            )
        ),
    ] = 'default',
    k: Annotated[
        int | None,
        Query(
            ge=1,
            le=DIVERSE_BROWSE_MAX_K,
            description='order=diverse only: rank just the first k picks.',
        ),
    ] = None,
) -> CropBrowse:
    return CropBrowse(
        sort=sort,
        label_source=label_source,
        class_source=class_source,
        label_validated=label_validated,
        import_id=import_id,
        dataset_split=dataset_split,
        on_negative_frame=on_negative_frame,
        proposed_by_import=proposed_by_import,
        source=source,
        open_vocab_set=open_vocab_set,
        source_prompt=source_prompt,
        needs_new_class=needs_new_class,
        review_dismissed=review_dismissed,
        include_test=include_test,
        include_excluded=include_excluded,
        min_blur_ratio=min_blur_ratio,
        classifier_conf_lt=classifier_conf_lt,
        item_text=item_text,
        region_text=region_text,
        region_text_match=region_text_match,
        same_text_as=same_text_as,
        order=order,
        k=k,
    )


CropBrowseQuery = Annotated[CropBrowse, Depends(crop_browse)]


async def resolve_listing(opensearch: Any, item_filter: ItemFilter, p: CropBrowse) -> CropListQuery:
    """The query a crops request means, or the 400/404 that says why it cannot be built."""
    join = await resolve_filter_join(opensearch, item_filter)
    region_text, region_text_match = p.region_text, p.region_text_match
    if p.same_text_as is not None:
        if region_text is not None:
            raise HTTPException(
                status_code=400, detail='same_text_as and region_text are exclusive'
            )
        try:
            region_text = await same_text_key(opensearch, p.same_text_as)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        region_text_match = 'exact'
    try:
        return build_crop_list_query(
            embedding=await item_write_fields(),
            sort=p.sort,
            item_filter_params=item_filter.model_dump(exclude_defaults=True),
            route_params={
                'label_source': p.label_source,
                'class_source': p.class_source,
                'label_validated': p.label_validated,
                'import_id': p.import_id,
                'dataset_split': p.dataset_split,
                'on_negative_frame': p.on_negative_frame,
                'proposed_by_import': p.proposed_by_import,
                'source': p.source,
                'needs_new_class': p.needs_new_class,
                'review_dismissed': p.review_dismissed,
                'min_blur_ratio': p.min_blur_ratio,
                'classifier_conf_lt': p.classifier_conf_lt,
                'item_text': p.item_text,
                'region_text': region_text,
                'region_text_match': region_text_match,
                'open_vocab_set': p.open_vocab_set,
                'source_prompt': p.source_prompt,
            },
            include_test=p.include_test,
            include_excluded=p.include_excluded,
            image_join=join,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
