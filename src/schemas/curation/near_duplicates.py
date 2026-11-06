"""Wire models of the near-duplicate service and the body of the ``near_dup_profile`` document.

The profile body is validated on write and on read (``extra='forbid'``: a misspelt option is a
422). The cuts are cosines of unit-norm vectors; the vector field is never named here: the
service reads the project's primary space of the level's target (``embedding_space``), or the
live space named by ``space_key``.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain import near_dup as rules


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class _LevelCuts(_Strict):
    space_key: str | None = Field(
        default=None, description='A live space of the level; null: the primary space.'
    )
    dedup_cosine: float = Field(le=1.0, description='Members at or above it are duplicates.')
    leak_cosine: float = Field(
        ge=rules.MIN_LEAK_COSINE, description='Members at or above it must stay on one split side.'
    )
    max_group_size: int = Field(default=rules.DEFAULT_MAX_GROUP_SIZE, ge=rules.MIN_GROUP_SIZE_GUARD)

    @model_validator(mode='after')
    def _leak_is_not_stricter_than_dedup(self) -> Self:
        if self.leak_cosine > self.dedup_cosine:
            msg = 'leak_cosine must not exceed dedup_cosine'
            raise ValueError(msg)
        return self


class NearDupImageProfile(_LevelCuts):
    dedup_cosine: float = Field(default=rules.DEFAULT_IMAGE_DEDUP_COSINE, le=1.0)
    leak_cosine: float = Field(default=rules.DEFAULT_IMAGE_DEDUP_COSINE, ge=rules.MIN_LEAK_COSINE)
    k: int = Field(default=rules.DEFAULT_K, ge=rules.MIN_K, le=rules.MAX_K)


class NearDupItemProfile(_LevelCuts):
    dedup_cosine: float = Field(default=rules.DEFAULT_ITEM_DEDUP_COSINE, le=1.0)
    leak_cosine: float = Field(default=rules.DEFAULT_ITEM_DEDUP_COSINE, ge=rules.MIN_LEAK_COSINE)
    scope: Literal['image_leak_groups'] = 'image_leak_groups'
    same_detector_class: bool = True


class NearDupProfileBody(_Strict):
    version: Literal[1] = 1
    image: NearDupImageProfile = Field(default_factory=NearDupImageProfile)
    item: NearDupItemProfile = Field(default_factory=NearDupItemProfile)


# --- the wire of the routes -----------------------------------------------------------------

Level = Literal['image', 'item']
Purpose = Literal['dedup', 'leak']
StaleReason = Literal['pending', 'rebuild_required', 'no_run']


class NearDupRunView(BaseModel):
    id: str
    level: str
    mode: str
    status: str
    space: str
    algorithm_version: str
    profile_revision: int | None
    params: dict[str, Any]
    members_scored: int
    dedup_groups: int
    leak_groups: int
    grouped_members: int
    largest_group: int
    oversized_split: int
    holdout_absorbed: int
    started_at: str
    finished_at: str | None


class NearDupRunPage(BaseModel):
    items: list[NearDupRunView]


class NearDupLevelStatus(BaseModel):
    level: Level
    current_run: NearDupRunView | None
    pending_count: int
    stale_reasons: list[StaleReason]


class NearDupActiveProfile(BaseModel):
    name: str
    revision: int


class NearDupStatus(BaseModel):
    profile: NearDupActiveProfile | None
    levels: list[NearDupLevelStatus]


class NearDupRunRequest(_Strict):
    level: Level
    mode: Literal['full', 'incremental'] = 'full'


class NearDupRunStarted(BaseModel):
    job_id: str
    kind: str


class NearDupGroupSummary(BaseModel):
    group_id: str = Field(description='The lineage id: stable while the group keeps its members.')
    purpose: Purpose
    size: int
    representative_id: str
    min_edge_cosine: float
    min_cosine_to_rep: float
    has_holdout: bool


class NearDupGroupPage(BaseModel):
    items: list[NearDupGroupSummary]
    next_cursor: str | None


class NearDupMemberView(BaseModel):
    member_id: str
    is_representative: bool
    cosine_to_rep: float


class NearDupGroupDetail(BaseModel):
    group: NearDupGroupSummary
    members: list[NearDupMemberView]


class NearDupDecisionIn(_Strict):
    level: Level
    a_id: str = Field(min_length=1)
    b_id: str = Field(min_length=1)
    decision: Literal['same', 'different']

    @model_validator(mode='after')
    def _two_members(self) -> Self:
        if self.a_id == self.b_id:
            msg = 'a decision names two different members'
            raise ValueError(msg)
        return self


class NearDupDecisionView(BaseModel):
    id: str
    level: Level
    a_id: str
    b_id: str
    decision: Literal['same', 'different']
    decided_by: str
    decided_at: str
