"""The served ``/ingest/bulk*`` contract matches what the runner writes and what the
committed OpenAPI says, so the wire models cannot drift from either side."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from src.routers.curation._ingest_bulk_models import (
    BulkIngestProgress,
    BulkIngestRequest,
    BulkIngestRun,
    BulkIngestStatus,
)
from src.schemas.jobs import JobView
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.runner import BulkIngestRun as Runner, BulkOptions
from src.services.curation.bulk_ingest.source_runs import (
    ACTIVE_RUN_STATUSES,
    RESUMABLE_RUN_STATUSES,
)


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}/ingest/bulk'


def test_every_status_a_run_job_can_reach_is_on_the_wire() -> None:
    # A job a run is paused cannot be: bulk_ingest_run is not pausable.
    reachable = set(get_args(JobView.model_fields['status'].annotation)) - {'paused'}
    assert set(get_args(BulkIngestStatus)) == reachable
    assert (ACTIVE_RUN_STATUSES | RESUMABLE_RUN_STATUSES | {'completed'}) - {'paused'} <= reachable


def test_the_runner_counters_fill_the_progress_model_exactly() -> None:
    run = Runner(
        sender=None,  # type: ignore[arg-type]
        ledger=None,  # type: ignore[arg-type]
        reporter=None,  # type: ignore[arg-type]
        options=BulkOptions(),
        run_id='r',
    )
    counters = run.counters(0.0)
    served = set(BulkIngestProgress.model_fields)
    written = set(counters) - {'stage_s'}
    # elapsed_s, eta_s, images_per_s(_recent) are written; everything served is written.
    # run_wire fills ``processed`` from the job's own counter and ``skipped`` is fixed at 0.
    assert served - {'processed', 'skipped'} == written


def test_committed_openapi_serves_the_bulk_routes() -> None:
    spec = json.loads(CONTRACT.read_text())
    paths = spec['paths']
    assert set(paths[PREFIX]) == {'get', 'post'}
    assert set(paths[f'{PREFIX}/config']) == {'get'}
    assert set(paths[f'{PREFIX}/{{run_id}}']) == {'get'}
    assert set(paths[f'{PREFIX}/{{run_id}}/failures']) == {'get'}
    assert set(paths[f'{PREFIX}/{{run_id}}/cancel']) == {'post'}
    assert paths[PREFIX]['post']['responses']['202']
    schemas = spec['components']['schemas']
    assert schemas['BulkIngestRequest']['additionalProperties'] is False
    assert set(schemas['BulkIngestRun']['properties']) == set(BulkIngestRun.model_fields)


def test_request_defaults_are_the_served_defaults() -> None:
    request = BulkIngestRequest(folder='/x')
    assert (request.batch_size, request.concurrency) == (
        limits.DEFAULT_BATCH_SIZE,
        limits.DEFAULT_CONCURRENCY,
    )


def test_default_concurrency_reaches_the_gpu_bound_plateau() -> None:
    # One A6000 reached its plateau (~29 images/s) at 16 batches in flight and lost 5 % at 8
    # and 24 % at 4 (docs/PERFORMANCE.md, "Stage 2 tuning").
    assert limits.DEFAULT_CONCURRENCY == 16
    assert limits.DEFAULT_CONCURRENCY <= limits.MAX_CONCURRENCY
