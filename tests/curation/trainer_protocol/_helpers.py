"""Helpers shared by the trainer conformance tests."""

from __future__ import annotations

from typing import Any

from job_spec import JobSpec, parse_and_validate_spec

from src.services.training import lineage
from src.services.training.job_models import TrainJobSpec


def make_spec(job_id: str = 'run-test', **overrides: Any) -> JobSpec:
    """The trainer's view of a run started with ``overrides`` (the stored ``training_runs.spec``)."""
    spec = TrainJobSpec(**overrides)
    if spec.dataset_export_dir and spec.frozen_test_sha is None:
        # What ``start_run`` stores from the export's manifest.
        identity = lineage.read_export_identity(spec.dataset_export_dir)
        spec.dataset_sha = identity.dataset_sha
        spec.frozen_test_sha = identity.frozen_test_sha
        spec.test_label_sha = identity.test_label_sha
    stored = spec.model_dump(mode='json', exclude_none=False)
    return parse_and_validate_spec(job_id, {**stored, 'job_id': job_id})
