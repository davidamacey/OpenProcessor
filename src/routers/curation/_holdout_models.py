"""Wire models for `POST /holdout/freeze`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Runtime import: pydantic resolves the Literal annotation from module globals.


class TestHoldoutFreezeRequest(BaseModel):
    """Selection is deterministic (SHA1 of each ``crop_id``, per class — see
    ``src/services/curation/holdout.py``), so there is no seed: an unknown
    field such as ``seed`` is a ``422`` rather than silently ignored."""

    model_config = {'extra': 'forbid'}

    percent: int = Field(default=10, ge=1, le=50)


class TestHoldoutFreezeResponse(BaseModel):
    n_frozen: int = Field(description='Items held out: the selection plus its near-duplicates.')
    moved_by_near_dup: int = Field(
        description='Items of n_frozen pulled in as near-duplicates of a selected item.'
    )
    n_classes_covered: int
    test_holdout_sha: str
    per_class_counts: dict[str, int] = Field(default_factory=dict)
    selection: Literal['sha1_per_class'] = Field(
        default='sha1_per_class',
        description='Per class, the crops with the smallest sha1(crop_id); no seed.',
    )
    percent: int = Field(description='Target holdout percent per class (from the request).')
    min_per_class: int = Field(
        description='Floor per class (all of a class smaller than this is frozen).'
    )
