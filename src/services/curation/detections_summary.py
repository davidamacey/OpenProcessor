"""What detection stored: items per detector label, split by embedding state.

The answer to "what did the detector find and what is embedded?" over the items
a shared item filter selects. Every label count carries the same embedding
breakdown the dataset stats serve, plus a ready-to-POST ``embed`` request for
the stored detections that have no vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.services.curation.embedding_state import DEFERRED, FAILED, NOT_SELECTED
from src.services.curation.item_filter import ItemFilter, item_filter_query
from src.services.curation.reprocess_models import (
    EmbedOptions,
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)
from src.services.curation.stats_embedding import embedding_aggregations, embedding_summary


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

MAX_LABELS = 500
NO_LABEL = '(no label)'
_NAME_MISSING = '__none__'


class EmbeddingBreakdown(BaseModel):
    embedded: int
    not_embedded: int
    by_state: dict[str, int]


class LabelSummary(BaseModel):
    name: str
    count: int
    embedding: EmbeddingBreakdown


class DetectionsSummary(BaseModel):
    total: int
    embedding: EmbeddingBreakdown
    by_label: list[LabelSummary]
    # True when the project has more labels than ``by_label`` lists.
    labels_truncated: bool
    # A request to POST to ``/reprocess`` that embeds the stored detections that
    # have no vector (under the same filter); null when none are missing.
    suggested_reprocess: ReprocessRequest | None


def suggested_embed_request(flt: ItemFilter) -> ReprocessRequest:
    """The ``embed`` reprocess (dry run, only the missing) for ``flt``'s items
    that were stored without a vector for a recorded reason."""
    missing = [NOT_SELECTED, DEFERRED, FAILED]
    selector = flt.model_copy(update={'embedding_state': missing})
    return ReprocessRequest(
        targets=ReprocessTargets(filter=ReprocessFilter(**selector.model_dump())),
        scopes=['embed'],
        embed=EmbedOptions(only_missing=True),
        dry_run=True,
    )


async def detections_summary(
    opensearch: AsyncOpenSearch, index: str, flt: ItemFilter
) -> DetectionsSummary:
    body = {
        'size': 0,
        'track_total_hits': True,
        'query': item_filter_query(flt),
        'aggs': {
            **embedding_aggregations(),
            'by_label': {
                'terms': {'field': 'proposal_name', 'size': MAX_LABELS, 'missing': _NAME_MISSING},
                'aggs': embedding_aggregations(),
            },
            'n_labels': {'cardinality': {'field': 'proposal_name'}},
        },
    }
    resp = await opensearch.search(index=index, body=body)
    total = int(((resp.get('hits') or {}).get('total') or {}).get('value', 0))
    aggs = resp.get('aggregations') or {}
    labels = [
        LabelSummary(
            name=NO_LABEL if b['key'] == _NAME_MISSING else str(b['key']),
            count=int(b['doc_count']),
            embedding=EmbeddingBreakdown(**embedding_summary(b, int(b['doc_count']))),
        )
        for b in (aggs.get('by_label') or {}).get('buckets') or []
    ]
    overall = EmbeddingBreakdown(**embedding_summary(aggs, total))
    return DetectionsSummary(
        total=total,
        embedding=overall,
        by_label=labels,
        labels_truncated=int((aggs.get('n_labels') or {}).get('value', 0)) > MAX_LABELS,
        suggested_reprocess=suggested_embed_request(flt) if overall.not_embedded else None,
    )


__all__ = [
    'DetectionsSummary',
    'EmbeddingBreakdown',
    'LabelSummary',
    'detections_summary',
    'suggested_embed_request',
]
