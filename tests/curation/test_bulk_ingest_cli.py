"""``scripts/curation/bulk_ingest.py`` against a canned API: the CLI is a client of the run
routes, so its output and exit codes are checked on the wire models they serve (the routes
themselves are covered by ``ingest_db/test_ingest_runs_routes.py``)."""

from __future__ import annotations

import contextlib
import json
import types
from typing import Any

import httpx
import pytest

from src.routers.curation._ingest_bulk_models import BulkIngestProgress, BulkIngestRun


RUN_ID = '0b1c2d3e-0000-4000-8000-000000000001'
BASE = 'http://testserver/curation/projects/default/ingest/bulk'


def _run(status: str, **progress: Any) -> dict[str, Any]:
    fields = {
        'discovered': 6,
        'discovery_complete': True,
        'skipped': 0,
        'ingested': 6,
        'duplicates': 0,
        'failed': 0,
        'processed': 6,
        'crops_indexed': 6,
        'n_embedded': 6,
        'n_not_embedded': 0,
        'n_embed_failed': 0,
        'in_flight': 0,
        'images_per_s': 12.0,
        'elapsed_s': 1.0,
        **progress,
    }
    live = status in {'queued', 'running'}
    return BulkIngestRun(
        project='default',
        run_id=RUN_ID,
        status=status,  # type: ignore[arg-type]
        source={'kind': 'folder', 'path': '/imgs', 'extensions': ['.jpg']},
        options={'source_tag': 'cli_test', 'batch_size': 8, 'concurrency': 2},
        progress=BulkIngestProgress(**fields),
        stage_s={'read': 0.5},
        can_cancel=live,
        can_resume=status in {'failed', 'cancelled', 'interrupted', 'completed_with_errors'},
    ).model_dump()


class Api:
    """The handful of routes the CLI calls, answering from ``routes``."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, Any]] = {}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path.removeprefix('/curation/projects/default'))
        self.calls.append(key)
        status, body = self.routes.get(key, (404, {'detail': 'no such route'}))
        return httpx.Response(status, content=json.dumps(body))


@pytest.fixture
def api() -> Api:
    return Api()


@pytest.fixture
def cli(api: Api, monkeypatch: pytest.MonkeyPatch) -> Any:
    from scripts.curation import bulk_ingest

    client = httpx.Client(transport=httpx.MockTransport(api))
    monkeypatch.setattr(bulk_ingest, 'bind_script_project', lambda _slug: None)
    monkeypatch.setattr(
        bulk_ingest,
        'httpx',
        types.SimpleNamespace(
            Client=lambda: contextlib.nullcontext(client), HTTPError=httpx.HTTPError
        ),
    )
    return bulk_ingest


def _args(*extra: str) -> list[str]:
    return ['--api-base', 'http://testserver/curation', '--poll-interval', '0.01', *extra]


def test_a_folder_run_prints_the_outcome_and_exits_zero(
    cli: Any, api: Api, capsys: pytest.CaptureFixture[str]
) -> None:
    api.routes[('POST', '/ingest/bulk')] = (202, {**_run('queued', ingested=0), 'resumed': False})
    api.routes[('GET', f'/ingest/bulk/{RUN_ID}')] = (200, _run('completed'))
    api.routes[('GET', f'/ingest/bulk/{RUN_ID}/failures')] = (
        200,
        {'items': [], 'total': 0, 'offset': 0, 'limit': 20},
    )

    code = cli.main(_args('--folder', '/imgs', '--source', 'cli_test'))

    out = capsys.readouterr().out
    assert code == 0
    assert f'started run {RUN_ID}' in out
    assert 'completed' in out
    assert 'ingested 6' in out
    assert 'stage seconds:' in out


def test_a_run_with_unreadable_entries_exits_one_and_lists_them(
    cli: Any, api: Api, capsys: pytest.CaptureFixture[str]
) -> None:
    api.routes[('POST', '/ingest/bulk')] = (202, {**_run('queued'), 'resumed': False})
    api.routes[('GET', f'/ingest/bulk/{RUN_ID}')] = (
        200,
        _run('completed_with_errors', ingested=1, failed=1),
    )
    api.routes[('GET', f'/ingest/bulk/{RUN_ID}/failures')] = (
        200,
        {
            'items': [{'path': '/imgs/gone.jpg', 'error': 'x', 'error_kind': 'decode_failed'}],
            'total': 1,
            'offset': 0,
            'limit': 20,
        },
    )

    code = cli.main(_args('--manifest', '/list.txt'))

    out = capsys.readouterr().out
    assert code == 1
    assert 'FAILED' in out
    assert '/imgs/gone.jpg [decode_failed]' in out
    assert 'start the same source again to resume' in out


def test_a_refused_source_exits_two_with_the_reason(
    cli: Any, api: Api, capsys: pytest.CaptureFixture[str]
) -> None:
    api.routes[('POST', '/ingest/bulk')] = (
        422,
        {'detail': {'error': 'bulk_ingest_invalid_source', 'message': 'not a directory'}},
    )
    assert cli.main(_args('--folder', '/missing')) == 2
    assert 'not a directory' in capsys.readouterr().err


def test_status_and_cancel_address_a_run_by_id(
    cli: Any, api: Api, capsys: pytest.CaptureFixture[str]
) -> None:
    api.routes[('GET', f'/ingest/bulk/{RUN_ID}')] = (200, _run('running'))
    api.routes[('POST', f'/ingest/bulk/{RUN_ID}/cancel')] = (200, _run('cancelling'))
    assert cli.main(_args('--status', RUN_ID)) == 0
    assert 'running' in capsys.readouterr().out
    assert cli.main(_args('--cancel', RUN_ID)) == 0
    assert 'cancelling' in capsys.readouterr().out
    assert cli.main(_args('--status', 'unknown')) == 2
    assert '404' in capsys.readouterr().err


def test_preview_prints_the_counts_and_the_estimate_and_starts_nothing(
    cli: Any, api: Api, capsys: pytest.CaptureFixture[str]
) -> None:
    api.routes[('POST', '/ingest/bulk/preview')] = (
        200,
        {
            'kind': 'folder',
            'target': '/imgs',
            'files': 6,
            'total_bytes': 600,
            'by_extension': {'.jpg': {'files': 6, 'bytes': 600}},
            'megapixels': {'sampled': 0},
            'unreadable_dirs': 0,
            'symlinks_skipped': 0,
            'missing': 0,
            'outside_roots': 0,
            'unreadable_files': 0,
            'unreadable_samples': [],
            'estimate': {
                'duration_s_min': 1.0,
                'duration_s_max': 2.0,
                'basis': 'measured',
                'opensearch_bytes': 1000,
                'vectors': 12,
            },
            'partial': False,
        },
    )
    api.routes[('GET', '/ingest/bulk/config')] = (
        200,
        {'preview': {'source_document': 'docs/PERFORMANCE.md'}},
    )

    code = cli.main(_args('--folder', '/imgs', '--preview'))

    out = capsys.readouterr().out
    assert code == 0
    assert 'files: 6' in out
    assert '.jpg' in out
    assert 'estimated ingest time' in out
    assert 'docs/PERFORMANCE.md' in out
    assert ('POST', '/ingest/bulk') not in api.calls


def test_preview_cannot_be_combined_with_status(cli: Any) -> None:
    with pytest.raises(SystemExit):
        cli.main(_args('--status', 'x', '--preview'))


@pytest.mark.parametrize(
    ('low', 'high', 'shown'),
    [
        (0.34, 0.34, 'about 0.3 s'),
        (69.0, 69.0, 'about 1m09s'),
        (3600.0, 7200.0, '1h00m to 2h00m'),
    ],
)
def test_the_estimate_is_shown_in_readable_units(
    cli: Any, low: float, high: float, shown: str
) -> None:
    assert cli._fmt_duration(low, high) == shown
