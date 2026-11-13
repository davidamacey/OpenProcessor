"""New coverage (test audit recommendation 5):
`/dashboard`'s stats come over SSE (`GET {API_PREFIX}/pipeline/events`,
`subscribePipelineEvents`, src/lib/sse.ts). A `snapshot` frame carrying
`{stats: {error: "..."}}` (the live-backend G1 failure mode) must render
the literal "Stats unavailable: {error}" text (DatasetStats.svelte)
without ever throwing a page error — resolveStatsUpdate
(src/lib/datasetStats.ts) guards it.
"""

from __future__ import annotations

import json

from fixtures.contract import auto_label_job
from fixtures.wire import labeled_stats
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'ducati',
        'kind': 'item',
        'group': 'moto',
        'hotkey_letter': 'k',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
]

IDLE_JOB = auto_label_job()


def open_pipeline_details(page) -> None:
    """The stats panels sit in the collapsed 'Pipeline details' disclosure."""
    page.get_by_test_id('pipeline-details').get_by_text('Pipeline details', exact=True).click()


def serve_dashboard(stub):
    """Every read the dashboard makes, with empty data."""
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', {'strategies': [], 'flags': {}})
    stub.on('GET', r'/pipeline/auto_label/status', IDLE_JOB)
    # A degraded SSE connection makes DatasetStats fall back to polling
    # these directly — stub them even though this test's assertion is
    # about the SSE-carried error, not these calls.
    stub.on(
        'GET',
        r'/stats/dataset(\?|$)',
        {'total_crops': 3, 'validated': 0, 'test_holdout': 0, 'by_source': {}},
    )
    stub.on('GET', r'/stats/classes(\?|$)', {'classes': []})
    # D2 (visual audit 2026-09-24): the dashboard balance chart reads the
    # served per-class test holdout.
    stub.on('GET', r'/test_holdout/stats(\?|$)', {'total': 0, 'by_class': []})
    # The Data quality card.
    stub.on(
        'GET',
        r'/quality/readiness(\?|$)',
        {'verdict': 'ready', 'checks': [], 'coverage': [], 'computed_at': '2026-11-12T18:20:00Z'},
    )
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 30, 'crops': []})


def test_dashboard_stats_unavailable(stub, page, app_url):
    serve_dashboard(stub)

    snapshot = {'state': {}, 'stats': {'error': 'stats unavailable (503)'}}
    sse_body = f'event: snapshot\ndata: {json.dumps(snapshot)}\n\n'
    stub.on('GET', r'/pipeline/events', (200, sse_body, 'text/event-stream'))

    page.goto(f'{app_url}/p/default/dashboard')
    open_pipeline_details(page)
    page.get_by_text('Stats unavailable:', exact=False).first.wait_for(timeout=ACTION_TIMEOUT_MS)

    expect(
        page.get_by_text('Stats unavailable: stats unavailable (503)', exact=False).first
    ).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'a stats 503 must not throw a page error: {errors[:3]}'


def test_dashboard_labeled_breakdown_shows_the_served_rows_with_imports(stub, page, app_url):
    """The labeled total and rows are the server's: imported labels count (they once did not)."""
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', {'strategies': [], 'flags': {}})
    stub.on('GET', r'/pipeline/auto_label/status', IDLE_JOB)
    stats = {
        'as_of': '2026-11-11T00:00:00Z',
        'total_crops': 502,
        'validated': 285,
        'test_holdout': 25,
        'by_source': [],
        'labeled': labeled_stats(by_human=100, by_import=185),
        'regions': {
            'boxed': 0,
            'coverage_share': 0,
            'confirmed': 0,
            'total_detected': 0,
            'by_detector': 0,
            'by_segmenter': 0,
            'by_human_drew': 0,
            'verified_by_human': 0,
            'verified_by_vlm': 0,
            'validated_by_human': 0,
        },
        'unlabeled': {'pending_detection': 0, 'pending_verification': 0, 'no_label_source': 0},
        'in_progress': {'region_drain_total_unfinished': 0},
        'clusters': {
            'last_run_at': None,
            'cluster_count': 0,
            'residual_count': 0,
            'noise_count': 0,
            'method': None,
        },
    }
    stub.on('GET', r'/stats/dataset(\?|$)', stats)
    stub.on('GET', r'/stats/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/test_holdout/stats(\?|$)', {'total': 0, 'by_class': []})
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 30, 'crops': []})
    stub.on(
        'GET',
        r'/quality/readiness(\?|$)',
        {'verdict': 'ready', 'checks': [], 'coverage': [], 'computed_at': '2026-11-12T18:20:00Z'},
    )
    snapshot = {'state': {}, 'stats': stats}
    sse_body = f'event: snapshot\ndata: {json.dumps(snapshot)}\n\n'
    stub.on('GET', r'/pipeline/events', (200, sse_body, 'text/event-stream'))

    page.goto(f'{app_url}/p/default/dashboard')
    kpis = page.get_by_test_id('dashboard-kpis')
    expect(kpis.locator('[data-kpi="objects"] dd')).to_have_text('502', timeout=ACTION_TIMEOUT_MS)
    expect(kpis.locator('[data-kpi="validated"] dd')).to_have_text('285')
    open_pipeline_details(page)
    panel = page.get_by_role('heading', name='Labeled by source').locator('xpath=ancestor::div[1]')
    expect(panel).to_contain_text('285 total', timeout=ACTION_TIMEOUT_MS)
    expect(panel).to_contain_text('Imported')
    expect(panel).to_contain_text('185')
    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, errors[:3]
