"""Wire models of the near-duplicate service and the body of the ``near_dup_profile`` document.

The profile body is validated on write and on read (``extra='forbid'``: a misspelt option is a
422). The cuts are cosines of unit-norm vectors; the vector field is never named here: the
service reads the project's primary space of the level's target (``embedding_space``), or the
live space named by ``space_key``.
"""

from __future__ import annotations

from typing import Literal, Self

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
