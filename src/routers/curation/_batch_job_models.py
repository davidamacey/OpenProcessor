"""The 202 body of a batch route whose work was queued as a job."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BatchJobAccepted(BaseModel):
    """The queued job; read its outcome at ``GET /jobs/{job_id}``."""

    job_id: str = Field(description='The queued job; poll GET /jobs/{job_id}.')
    status: str
