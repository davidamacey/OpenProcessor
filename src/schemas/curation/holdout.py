"""Wire shape of ``GET {P}/test_holdout/stats``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HoldoutClassBucket(BaseModel):
    # An OpenSearch terms bucket plus ``deficient``; any further bucket key is kept.
    model_config = ConfigDict(extra='allow')

    key: int
    doc_count: int
    deficient: bool


class HoldoutStats(BaseModel):
    total: int
    by_class: list[HoldoutClassBucket]
    min_test_per_class: int
