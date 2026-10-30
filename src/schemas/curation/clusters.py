"""Wire shapes of the cluster list (``GET {P}/clusters``).

TypedDicts, not models: the route answers with the plain dicts the service builds
(one card per cluster, ~600 per request), and validating that many models per call
cost more than the aggregation itself. The route documents them through
``responses=``; the tests validate real responses against them.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import Field


class ClusterRepresentative(TypedDict):
    crop_id: str
    cluster_distance: float | None
    class_name: str | None
    cluster_subid: str | None


class ClusterCard(TypedDict):
    cluster_id: int
    cluster_kind: Literal['class', 'candidate', 'unassigned']
    size: int
    validated_count: int
    labelled_count: int
    dominant_class_id: int | None
    dominant_class_name: str | None
    dominant_count: int
    purity: Annotated[
        float | None,
        Field(description='Share of measured members whose nearest centroid is their own cluster.'),
    ]
    purity_n: int
    purity_basis: str
    purity_tier: Literal['pure', 'mixed', 'noisy'] | None
    label_agreement: Annotated[
        float | None, Field(description="The top class's share of the labelled members.")
    ]
    labelled_share: float | None
    promotable: bool
    is_unlabeled: bool
    n_subclusters: int
    subcluster_ids: list[str]
    updated_at: str | None
    representatives: list[ClusterRepresentative]
    detector_top_class: Annotated[
        str | None, Field(description="The detector's most common class among the members.")
    ]
    detector_top_share: Annotated[
        float | None,
        Field(description='That class as a share of the members that have a detector class.'),
    ]
    detector_conf_mean: float | None
    validation_state: Literal['none', 'partial', 'complete']
    review_counts: Annotated[
        dict[str, int],
        Field(description='Members each review tab would show for this cluster, by tab id.'),
    ]
    needs_review_count: Annotated[
        int, Field(description="review_counts['all']: the review queue sort key.")
    ]
    flags: Annotated[list[str], Field(description='Quality flag codes, see flag_vocabulary.')]


class ClusterSortOption(TypedDict):
    id: str
    label: str


class ClusterFlag(TypedDict):
    id: str
    label: str
    description: str


class PurityThresholds(TypedDict):
    pure_min: float
    mixed_min: float
    promote_min_members: int
    promote_min_labelled_share: float


class ClusterListResponse(TypedDict):
    items: list[ClusterCard]
    total: Annotated[
        int, Field(description='Cards after the filters, before representatives paging.')
    ]
    total_class_clusters: int
    total_candidate_clusters: int
    cluster_id_offset: int
    purity_thresholds: PurityThresholds
    core_similarity_min: float
    representatives_offset: int
    representatives_limit: int
    sort: Annotated[str, Field(description='The sort applied.')]
    sort_options: list[ClusterSortOption]
    flag_vocabulary: list[ClusterFlag]
    truncated: Annotated[
        bool,
        Field(
            description=(
                'The aggregation returned max_clusters buckets, so the list may be '
                'incomplete: raise max_clusters or narrow the filters.'
            )
        ),
    ]


class FacetCount(TypedDict):
    value: str
    count: int


class ValidationCounts(TypedDict):
    validated: int
    unvalidated: int


class ConfidenceBin(TypedDict):
    min: Annotated[float, Field(description='Send as `detector_conf_min` (inclusive).')]
    max: Annotated[
        float | None,
        Field(
            description='Send as `detector_conf_max` (exclusive); null on the last bin: no upper bound.'
        ),
    ]
    count: int


class ClusterFacets(TypedDict):
    """What a cluster's members look like, to render filter chips with counts.

    Every count is the ``total`` of the matching filtered ``GET /crops`` call:
    ``detector_classes[].value`` is ``detector_class_name``, ``class_sources[].value`` is
    ``class_source``, ``validation`` is ``label_validated`` true or false,
    ``review_tabs[tab]`` is ``review_tab``, and a ``detector_confidence`` bin is
    ``detector_conf_min`` / ``detector_conf_max``.
    """

    cluster_id: int
    total: Annotated[
        int, Field(description='Members, without the test holdout and excluded items.')
    ]
    detector_classes: Annotated[list[FacetCount], Field(description='Top 20 by count.')]
    class_sources: list[FacetCount]
    validation: ValidationCounts
    review_tabs: dict[str, int]
    detector_confidence: Annotated[
        list[ConfidenceBin], Field(description='Ten fixed bins of 0.1; members with a confidence.')
    ]
    detector_confidence_missing: Annotated[
        int, Field(description='Members with no detector confidence (in no bin).')
    ]
