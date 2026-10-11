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
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.runner import BulkIngestRun as Runner, BulkOptions
from src.services.curation.bulk_ingest.source import BulkSource
from src.services.curation.bulk_ingest.store import ACTIVE_STATUSES, FINISHED_STATUSES, BulkStore


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}/ingest/bulk'


def test_every_status_the_runner_can_write_is_on_the_wire_and_nothing_else() -> None:
    assert set(get_args(BulkIngestStatus)) == ACTIVE_STATUSES | FINISHED_STATUSES


def test_the_runner_snapshot_fills_the_progress_model_exactly(tmp_path: Path) -> None:
    run = Runner(
        sender=None,  # type: ignore[arg-type]
        store=BulkStore(tmp_path),
        source=BulkSource('folder', '/x'),
        options=BulkOptions(),
    )
    snapshot = run._snapshot(0.0)
    served = set(BulkIngestProgress.model_fields)
    written = set(snapshot) - {'stage_s', 'updated_at'}
    # elapsed_s, eta_s, images_per_s(_recent) are written; everything served is written.
    assert served == written


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
