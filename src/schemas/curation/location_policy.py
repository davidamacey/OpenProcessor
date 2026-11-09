"""Wire models of ``GET/PUT /curation/projects/{project}/location_policy``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LocationPolicyName = Literal['drop', 'store_coarse', 'store']


class LocationPolicyResponse(BaseModel):
    location_policy: LocationPolicyName
    revision: int = Field(
        description='The project settings revision; send it back as `expected_revision`.'
    )
    purge_job_id: str | None = Field(
        default=None,
        description='The `source_backfill` job (step `location_purge`) the change started, '
        'when the new policy is stricter than (or equal to) the old one and keeps less than '
        '`store`; `GET .../jobs/{job_id}` reports its counts.',
    )


class LocationPolicyUpdate(BaseModel):
    """`drop` stores and serves no coordinate and clears the stored ones; `store_coarse` keeps
    latitude/longitude rounded to 0.01 degree (about 1 km); `store` keeps the fix as read.
    Exports of a `drop` project strip the GPS from the copied files; otherwise they keep it."""

    location_policy: LocationPolicyName
    expected_revision: int = Field(ge=0)
