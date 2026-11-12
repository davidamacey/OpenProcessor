"""Embedding spaces page. Payloads are built from the vendored contract (``fill``).

  1. `test_lists_spaces_per_target_and_the_encoders` — a table per target that has spaces, the served
     encoder catalog with the reason an unavailable one cannot be used.
  2. `test_register_follows_the_served_run_to_its_end` — Register posts the chosen encoder with an
     Idempotency-Key; the run's job status and counts are the served ones, then the list re-reads.
  3. `test_promote_refusals_are_served_and_force_waives_only_a_missing_evaluation`.
  4. `test_restart_retire_and_cancel` — the build mode, the target in the query, abandon.
  5. `test_primary_offers_neither_promote_nor_retire`.
  6. `test_page_fits_430`.
  7. `test_a_build_started_elsewhere_is_followed_from_the_list` — the list names the running run; the
     page follows it, offers Cancel build and re-reads the list when the job ends.
  8. `test_the_served_evaluation_is_shown_on_the_row_and_in_the_promote_step`.

Set OP_E2E_SHOT_DIR to also save full-page screenshots (1600 and 430 px).
"""

from __future__ import annotations

from typing import Any

from fixtures.contract import embedding_space, fill
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import shot
from harness.waits import expect_handled
from playwright.sync_api import expect


SHOT_WIDTHS = (1600, 430)
PRIMARY = embedding_space(
    'pe_core_v1_1024', model_key='pe_core', model_version=1, dim=1024, docs_with_vector=4000
)
CANDIDATE = embedding_space(
    'mobileclip2_s2_v1_512',
    model_key='mobileclip2_s2',
    model_version=1,
    dim=512,
    role='candidate',
    status='ready',
    ready_at='2026-11-10T10:00:00Z',
    docs_with_vector=4000,
)
BOX_PRIMARY = embedding_space(
    'pe_core_v1_box',
    target='box',
    model_key='pe_core',
    model_version=1,
    dim=1024,
    docs_with_vector=900,
)
MODELS = fill(
    'EmbeddingModelListResponse',
    models=[
        fill(
            'EmbeddingModelEntry',
            model_key='mobileclip2_s2',
            version=1,
            modalities=['image', 'text'],
            native_dim=512,
            matryoshka_dims=[256],
            licence='apple-amlr',
            available=True,
        ),
        fill(
            'EmbeddingModelEntry',
            model_key='siglip2',
            version=1,
            modalities=['image', 'text'],
            native_dim=768,
            matryoshka_dims=[],
            licence='apache-2.0',
            available=False,
            unavailable_reason='no ready engine for this model',
        ),
    ],
)


def run(**over: Any) -> dict[str, Any]:
    base = {
        'id': 'run-1',
        'space_key': 'mobileclip2_s2_v1_512',
        'mode': 'build',
        'docs_total': 4000,
        'docs_done': 1000,
        'job_id': 'job-1',
        'job_status': 'running',
    }
    base.update(over)
    return fill('EmbeddingReindexRunResponse', **base)


def start(**over: Any) -> dict[str, Any]:
    base = {
        'space_key': 'mobileclip2_s2_v1_512',
        'target': 'item',
        'job_id': 'job-1',
        'run_id': 'run-1',
    }
    base.update(over)
    return fill('EmbeddingSpaceStartResponse', **base)


def serve_spaces(stub: Any, spaces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        'spaces': spaces or [PRIMARY, CANDIDATE, BOX_PRIMARY],
        'evaluations': {},
    }
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/embedding/spaces$',
        lambda _r, _m: (200, fill('EmbeddingSpaceListResponse', spaces=state['spaces'])),
    )
    stub.on('GET', r'/embedding/models$', MODELS)

    def evaluations(request: Any, _m: Any):
        key = request.url.split('/embedding/spaces/')[1].split('/')[0]
        return (200, state['evaluations'].get(key) or evaluation_list(key, []))

    stub.on('GET', r'/embedding/spaces/[^/]+/evaluations(\?|$)', evaluations)
    return state


def evaluation_list(key: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return fill(
        'EmbeddingEvaluationListResponse',
        space_key=key,
        target='item',
        primary_space_key='pe_core_v1_1024',
        evaluated_against_primary=any(r['counts_for_promote'] for r in runs),
        evaluations=runs,
    )


def evaluation_run(verdict: str, against_primary: bool = True) -> dict[str, Any]:
    return fill(
        'EmbeddingEvaluationView',
        id=f'ev-{verdict}',
        baseline_space_key='pe_core_v1_1024' if against_primary else 'older_v1',
        candidate_space_key='mobileclip2_s2_v1_512',
        protocol='label_consistency_v1',
        verdict=verdict,
        metrics={'recall_at_10': {'baseline': 0.7123, 'candidate': 0.78}, 'mrr': 0.41},
        sample_spec={'seed': 7, 'n': 2000},
        counts_for_promote=against_primary,
        created_at='2026-11-11T09:00:00Z',
        created_by='api',
    )


def open_page(page: Any, app_url: str) -> None:
    page.goto(f'{app_url}/p/default/settings/embedding-spaces')
    page.get_by_test_id('model-table').wait_for(timeout=ACTION_TIMEOUT_MS)


def row(page: Any, space: str) -> Any:
    return page.locator(f'[data-testid="space-row"][data-space="{space}"]')


def test_lists_spaces_per_target_and_the_encoders(stub, page, app_url):
    serve_spaces(stub)
    open_page(page, app_url)
    expect(page.locator('[data-testid="space-table"]')).to_have_count(2)
    item = page.locator('[data-testid="space-table"][data-target="item"]')
    expect(item.get_by_test_id('space-row')).to_have_count(2)
    expect(row(page, 'pe_core_v1_1024').get_by_test_id('space-role')).to_have_text('primary')
    expect(row(page, 'mobileclip2_s2_v1_512').get_by_test_id('space-status')).to_have_text('ready')
    expect(page.locator('[data-testid="space-table"][data-target="box"]')).to_have_count(1)
    unavailable = page.locator('[data-testid="model-row"][data-model="siglip2"]')
    expect(unavailable).to_contain_text('no: no ready engine for this model')
    expect(page.locator('[data-testid="model-row"][data-model="mobileclip2_s2"]')).to_contain_text(
        '256'
    )
    shot(page, 'embedding-spaces', widths=SHOT_WIDTHS)


def test_register_follows_the_served_run_to_its_end(stub, page, app_url):
    state = serve_spaces(stub, [PRIMARY])
    posts: list[Any] = []

    def register(request: Any, _m: Any):
        posts.append((request.post_data_json, request.headers.get('idempotency-key')))
        state['spaces'] = [PRIMARY, {**CANDIDATE, 'status': 'building', 'ready_at': None}]
        return (202, start())

    stub.on('POST', r'/embedding/spaces$', register)
    reads = {'n': 0}

    def read_run(_r: Any, _m: Any):
        reads['n'] += 1
        if reads['n'] == 1:
            return (200, run(docs_done=1000))
        if reads['n'] == 2:
            return (200, run(docs_done=2500))
        state['spaces'] = [PRIMARY, CANDIDATE]
        return (
            200,
            run(docs_done=4000, job_status='completed', finished_at='2026-11-12T10:30:00Z'),
        )

    stub.on('GET', r'/embedding/reindex-runs/run-1$', read_run)
    open_page(page, app_url)

    page.get_by_test_id('space-register').click()
    dialog = page.get_by_role('dialog')
    confirm = dialog.get_by_role('button', name='Register and build')
    expect(confirm).to_be_disabled()
    # the unavailable encoder is listed, with why, and cannot be chosen
    option = dialog.get_by_test_id('register-model').locator('option', has_text='siglip2')
    expect(option).to_have_attribute('disabled', '')
    expect(option).to_contain_text('no ready engine for this model')
    dialog.get_by_test_id('register-model').select_option('mobileclip2_s2@1')
    dialog.get_by_test_id('register-dim').select_option('256')
    shot(page, 'embedding-register', widths=SHOT_WIDTHS)
    with expect_handled(page, lambda r: r.method == 'POST'):
        confirm.click()
    body, key = posts[0]
    assert body == {'model_key': 'mobileclip2_s2', 'model_version': 1, 'target': 'item', 'dim': 256}
    assert key and len(key) >= 8

    new_row = row(page, 'mobileclip2_s2_v1_512')
    expect(new_row.get_by_test_id('run-status')).to_have_text('running', timeout=ACTION_TIMEOUT_MS)
    expect(new_row.get_by_test_id('run-counts')).to_contain_text('of 4000 done')
    shot(page, 'embedding-build-progress', widths=SHOT_WIDTHS)
    expect(new_row.get_by_test_id('run-status')).to_have_text(
        'completed', timeout=ACTION_TIMEOUT_MS
    )
    expect(new_row.get_by_test_id('run-counts')).to_contain_text('4000 of 4000 done')
    expect(new_row.get_by_test_id('space-status')).to_have_text('ready')


def test_promote_refusals_are_served_and_force_waives_only_a_missing_evaluation(
    stub, page, app_url
):
    serve_spaces(stub, [PRIMARY, {**CANDIDATE, 'status': 'building', 'ready_at': None}])
    promotes: list[Any] = []

    def promote(request: Any, _m: Any):
        body = request.post_data_json
        promotes.append((request.url, body))
        if len(promotes) == 1:
            return (
                409,
                {
                    'detail': {
                        'error': 'embedding_space_not_ready',
                        'message': 'mobileclip2_s2_v1_512 is building, not ready',
                    }
                },
            )
        if not body['force']:
            return (
                409,
                {
                    'detail': {
                        'error': 'embedding_evaluation_missing',
                        'message': 'no evaluation of mobileclip2_s2_v1_512 against pe_core_v1_1024; run one or pass force',
                    }
                },
            )
        return (
            200,
            fill(
                'EmbeddingPromoteResponse',
                primary={**CANDIDATE, 'role': 'primary'},
                previous={**PRIMARY, 'role': 'optional'},
                config_epoch=2,
            ),
        )

    stub.on('POST', r'/embedding/spaces/mobileclip2_s2_v1_512/promote$', promote)
    open_page(page, app_url)

    row(page, 'mobileclip2_s2_v1_512').get_by_role('button', name='Promote').click()
    dialog = page.get_by_role('dialog')
    dialog.get_by_role('button', name='Promote', exact=True).click()
    refusal = dialog.get_by_test_id('space-refusal')
    expect(refusal).to_contain_text('is building, not ready')
    expect(refusal).to_have_attribute('data-code', 'embedding_space_not_ready')
    expect(dialog.get_by_test_id('promote-force')).to_have_count(
        0
    )  # force never waives "not ready"
    shot(page, 'embedding-promote-not-ready', widths=SHOT_WIDTHS)

    dialog.get_by_role('button', name='Promote', exact=True).click()
    refusal = dialog.get_by_test_id('space-refusal')
    expect(refusal).to_have_attribute('data-code', 'embedding_evaluation_missing')
    expect(refusal).to_contain_text('run one or pass force')
    shot(page, 'embedding-promote-evaluation-missing', widths=SHOT_WIDTHS)
    dialog.get_by_test_id('promote-force').check()
    with expect_handled(page, lambda r: r.method == 'POST'):
        dialog.get_by_role('button', name='Promote without an evaluation').click()
    assert promotes[0][0].endswith('/promote?target=item')
    assert [p[1] for p in promotes] == [{'force': False}, {'force': False}, {'force': True}]
    expect(page.get_by_role('dialog')).to_have_count(0)


def test_restart_retire_and_cancel(stub, page, app_url):
    serve_spaces(stub, [PRIMARY, {**CANDIDATE, 'role': 'optional'}])
    builds: list[Any] = []
    stub.on(
        'POST',
        r'/embedding/spaces/mobileclip2_s2_v1_512/build$',
        lambda r, _m: (builds.append((r.url, r.post_data_json)), (202, start()))[1],
    )
    stub.on('GET', r'/embedding/reindex-runs/run-1$', (200, run()))
    cancels: list[str] = []
    stub.on(
        'POST',
        r'/embedding/reindex-runs/run-1/cancel$',
        lambda r, _m: (
            cancels.append(r.url.split('/curation', 1)[1]),
            (
                200,
                fill(
                    'EmbeddingRunCancelResponse',
                    run_id='run-1',
                    job_status='cancelled',
                    abandoned=True,
                    space_status='failed',
                ),
            ),
        )[1],
    )
    retires: list[str] = []
    stub.on(
        'POST',
        r'/embedding/spaces/mobileclip2_s2_v1_512/retire$',
        lambda r, _m: (
            retires.append(r.url.split('/curation', 1)[1]),
            (202, start(run_id=None)),
        )[1],
    )
    open_page(page, app_url)
    target = row(page, 'mobileclip2_s2_v1_512')
    expect(target.get_by_role('button', name='Cancel build')).to_have_count(0)

    target.get_by_role('button', name='Restart or backfill').click()
    dialog = page.get_by_role('dialog')
    dialog.get_by_test_id('build-mode').select_option('build')
    dialog.get_by_test_id('build-batch').fill('128')
    with expect_handled(page, lambda r: r.method == 'POST'):
        dialog.get_by_role('button', name='Start').click()
    assert builds == [
        (builds[0][0], {'mode': 'build', 'batch_size': 128}),
    ]
    assert builds[0][0].endswith('/build?target=item')
    expect(target.get_by_test_id('run-status')).to_have_text('running', timeout=ACTION_TIMEOUT_MS)

    target.get_by_role('button', name='Cancel build').click()
    page.get_by_test_id('cancel-abandon').check()
    with expect_handled(page, lambda r: r.method == 'POST'):
        page.get_by_role('dialog').get_by_role('button', name='Cancel build').click()
    assert cancels == ['/projects/default/embedding/reindex-runs/run-1/cancel?abandon=true']

    target.get_by_role('button', name='Retire').click()
    with expect_handled(page, lambda r: r.method == 'POST'):
        page.get_by_role('dialog').get_by_role('button', name='Retire', exact=True).click()
    assert retires == [
        '/projects/default/embedding/spaces/mobileclip2_s2_v1_512/retire?target=item'
    ]


def test_primary_offers_neither_promote_nor_retire(stub, page, app_url):
    serve_spaces(stub)
    open_page(page, app_url)
    primary = row(page, 'pe_core_v1_1024')
    expect(primary.get_by_role('button', name='Restart or backfill')).to_be_visible()
    expect(primary.get_by_role('button', name='Promote')).to_have_count(0)
    expect(primary.get_by_role('button', name='Retire')).to_have_count(0)


def test_page_fits_430(stub, page, app_url):
    serve_spaces(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    open_page(page, app_url)
    overflow = page.evaluate(
        'document.documentElement.scrollWidth - document.documentElement.clientWidth'
    )
    assert overflow <= 0, f'the page scrolls sideways by {overflow}px'
    for name in ('pe_core_v1_1024', 'mobileclip2_s2_v1_512'):
        r = row(page, name)
        box = r.locator('td').first.bounding_box()
        assert box is not None and box['x'] >= 0 and box['x'] + box['width'] <= 430
    promote = row(page, 'mobileclip2_s2_v1_512').get_by_role('button', name='Promote')
    promote.scroll_into_view_if_needed()
    expect(promote).to_be_visible()
    shot(page, 'embedding-spaces-430-rows', widths=(430,))


def test_a_build_started_elsewhere_is_followed_from_the_list(stub, page, app_url):
    elsewhere = {
        **CANDIDATE,
        'status': 'building',
        'ready_at': None,
        'run_id': 'run-7',
        'run_mode': 'build',
        'run_status': 'running',
    }
    state = serve_spaces(stub, [PRIMARY, elsewhere])
    reads = {'n': 0}

    def read_run(_r: Any, _m: Any):
        reads['n'] += 1
        if reads['n'] == 1:
            return (200, run(id='run-7', docs_done=1200))
        state['spaces'] = [
            PRIMARY,
            {**CANDIDATE, 'run_id': 'run-7', 'run_mode': 'build', 'run_status': 'completed'},
        ]
        return (
            200,
            run(
                id='run-7',
                docs_done=4000,
                job_status='completed',
                finished_at='2026-11-12T10:30:00Z',
            ),
        )

    stub.on('GET', r'/embedding/reindex-runs/run-7$', read_run)
    open_page(page, app_url)

    new_row = row(page, 'mobileclip2_s2_v1_512')
    expect(new_row.get_by_test_id('run-status')).to_have_text('running', timeout=ACTION_TIMEOUT_MS)
    expect(new_row.get_by_test_id('run-counts')).to_contain_text('1200 of 4000 done')
    expect(new_row.get_by_role('button', name='Cancel build')).to_be_visible()
    shot(page, 'embedding-build-elsewhere', widths=SHOT_WIDTHS)
    expect(new_row.get_by_test_id('space-status')).to_have_text('ready', timeout=ACTION_TIMEOUT_MS)
    expect(new_row.get_by_test_id('run-status')).to_have_text('completed')
    expect(new_row.get_by_role('button', name='Cancel build')).to_have_count(0)
    # a fresh visit shows the finished last run from the list alone, and follows nothing
    page.reload()
    page.get_by_test_id('model-table').wait_for(timeout=ACTION_TIMEOUT_MS)
    fresh = row(page, 'mobileclip2_s2_v1_512')
    expect(fresh.get_by_test_id('last-run-status')).to_have_text('completed')
    expect(fresh.get_by_test_id('run-status')).to_have_count(0)
    assert reads['n'] == 2


def test_the_served_evaluation_is_shown_on_the_row_and_in_the_promote_step(stub, page, app_url):
    state = serve_spaces(stub, [PRIMARY, CANDIDATE])
    open_page(page, app_url)

    # nothing recorded: said as served, no verdict is made up
    row(page, 'mobileclip2_s2_v1_512').get_by_test_id('space-evaluation-open').click()
    dialog = page.get_by_role('dialog')
    expect(dialog.get_by_test_id('evaluation-summary')).to_contain_text('No evaluation of')
    expect(dialog.get_by_test_id('evaluation-run')).to_have_count(0)
    dialog.get_by_role('button', name='Close').click()

    state['evaluations']['mobileclip2_s2_v1_512'] = evaluation_list(
        'mobileclip2_s2_v1_512', [evaluation_run('better'), evaluation_run('tie', False)]
    )
    row(page, 'mobileclip2_s2_v1_512').get_by_test_id('space-evaluation-open').click()
    dialog = page.get_by_role('dialog')
    runs = dialog.get_by_test_id('evaluation-run')
    expect(runs).to_have_count(2)
    expect(runs.first.get_by_test_id('evaluation-verdict')).to_have_text('better')
    expect(runs.first.get_by_test_id('evaluation-counts')).to_be_visible()
    expect(runs.nth(1).get_by_test_id('evaluation-counts')).to_have_count(0)
    expect(runs.first.get_by_test_id('evaluation-metric')).to_have_text(['0.7123', '0.78', '0.41'])
    expect(dialog.get_by_test_id('evaluation-summary')).to_contain_text(
        'One is against the current primary'
    )
    shot(page, 'embedding-evaluation', widths=SHOT_WIDTHS)
    dialog.get_by_role('button', name='Close').click()

    row(page, 'mobileclip2_s2_v1_512').get_by_role('button', name='Promote').click()
    promote = page.get_by_role('dialog')
    expect(promote.get_by_test_id('evaluation-run')).to_have_count(2)
    shot(page, 'embedding-promote-with-evaluation', widths=SHOT_WIDTHS)
