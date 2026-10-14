"""Helpers shared by the trainer conformance tests."""

from __future__ import annotations

import asyncio
from typing import Any

from src.services.training import jobs as train_jobs
from src.services.training.job_models import TrainJobSpec


def _write_job(**overrides: Any) -> str:
    """Submit a job through the *API's* writer and return the job id."""
    spec = TrainJobSpec(**overrides)
    return asyncio.run(train_jobs.write_job(spec))
