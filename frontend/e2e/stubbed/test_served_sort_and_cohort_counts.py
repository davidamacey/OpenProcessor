"""The Export class table sorts on the server (`GET /stats/classes?sort=`) and the Train
cohorts show the counts `GET /training_cohorts` serves, with no per-cohort query from the page.
Stubbed screenshots at 1600 and 430 px."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import SCREENS
from playwright.sync_api import expect
from test_train_gpus import CLASSES as TRAIN_CLASSES, register_train_mount


FIELDS = [
    ('class_name', 'Class'),
    ('class_id', 'ID'),
    ('count', 'Total'),
    ('validated_count', 'Validated'),
    ('trainable', 'Trainable'),
    ('aug_target', 'Aug target'),
    ('trainable_gap', 'Gap'),
    ('test_count', 'Test'),
]
OPTIONS = [
    {'id': f'{f}:{d}', 'label': f'{label} ({d})'} for f, label in FIELDS for d in ('asc', 'desc')
]
ROWS = [
    ('bmw', 1, 300, 200, 180, 20),
    ('audi', 2, 100, 50, 15, 0),
    ('fiat', 3, 80, 30, 12, 0),
]


def class_row(name, cid, count, validated, trainable, test):
    gap = max(0, 20 - trainable)
    return {
        'class_id': cid,
        'class_name': name,
        'count': count,
        'validated_count': validated,
        'adequacy': 'ok',
        'aug_target': 500,
        'aug_gap': 500 - validated,
        'trainable': trainable,
        'trainable_gap': gap,
        'test_count': test,
        'test_deficient': test < 5,
    }


def serve_export(stub) -> list[str]:
    sorts: list[str] = []

    def classes(request, _match):
        sort = parse_qs(urlparse(request.url).query).get('sort', ['class_id:asc'])[0]
        field, _, direction = sort.partition(':')
        direction = direction or ('asc' if field in ('class_name', 'class_id') else 'desc')
        sorts.append(sort)
        rows = [class_row(*r) for r in ROWS]
        rows.sort(key=lambda r: r[field], reverse=direction == 'desc')
        return {
            'classes': rows,
            'thresholds': {'block_below': 20, 'warn_below': 500, 'min_test': 5},
            'sort_applied': f'{field}:{direction}',
            'sort_options': OPTIONS,
        }

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/stats/classes(\?|$)', classes)
    stub.on(
        'GET',
        r'/stats/dataset(\?|$)',
        {'total_crops': 480, 'validated': 280, 'test_holdout': 20, 'by_source': []},
    )
    stub.on(
        'GET',
        r'/test_holdout/stats(\?|$)',
        {
            'total': 20,
            'by_class': [{'key': 1, 'doc_count': 20, 'deficient': False}],
            'min_test_per_class': 5,
        },
    )
    stub.on('GET', r'/export/status(\?|$)', {'status': 'idle', 'last_run': None})
    stub.on('GET', r'/export/datasets(\?|$)', {'datasets': []})
    return sorts


def first_class(page) -> str:
    return page.locator('tbody tr td:first-child').first.inner_text()


def test_export_header_click_asks_the_server_to_sort(stub, page, app_url):
    sorts = serve_export(stub)
    page.goto(f'{app_url}/p/default/export')
    expect(page.locator('tbody tr').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert sorts[0] == 'trainable_gap'
    assert first_class(page) == 'fiat'
    page.get_by_role('columnheader', name='Class').click()
    expect(page.locator('th[aria-sort="ascending"]')).to_have_text('Class ▲')
    assert sorts[-1] == 'class_name'
    assert first_class(page) == 'audi'
    page.get_by_role('columnheader', name='Class').click()
    assert sorts[-1] == 'class_name:desc'
    page.get_by_role('columnheader', name='Test').click()
    assert sorts[-1] == 'test_count'
    assert first_class(page) == 'bmw'


def test_export_screens(stub, page, app_url):
    serve_export(stub)
    out = SCREENS / 'workflow_g28'
    out.mkdir(parents=True, exist_ok=True)
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 1000})
        page.goto(f'{app_url}/p/default/export')
        expect(page.locator('tbody tr').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
        page.screenshot(path=str(out / f'export-table-{width}.png'), full_page=True)


def cohorts(request, _match):
    class_id = int(parse_qs(urlparse(request.url).query)['class_id'][0])
    rows = [
        ('validated', 'Validated', 'crop', 120 * class_id),
        ('needs_labeling', 'Needs labeling', 'crop', 7 * class_id),
        ('low_confidence', 'Low confidence', 'crop', 3 * class_id),
        ('model_disagreements', 'Model disagreements', 'crop', 0),
    ]
    return {
        'cohorts': [
            {
                'id': cid,
                'label': label,
                'description': f'{label} rows',
                'cutoffs': {},
                'endpoint': '/crops',
                'params': {'class_id': class_id},
                'row_kind': kind,
                'count': count,
            }
            for cid, label, kind, count in rows
        ]
    }


def test_train_cohorts_show_served_counts(stub, page, app_url):
    register_train_mount(stub)
    stub.on('GET', r'/training_cohorts(\?|$)', cohorts)
    assert TRAIN_CLASSES[0]['class_id'] == 1
    out = SCREENS / 'workflow_g28'
    out.mkdir(parents=True, exist_ok=True)
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 1000})
        page.goto(f'{app_url}/p/default/train')
        page.get_by_role('heading', name='ducati').scroll_into_view_if_needed()
        chip = page.locator('button[title="Validated rows"]').first
        expect(chip).to_contain_text('120', timeout=ACTION_TIMEOUT_MS)
        page.screenshot(path=str(out / f'train-cohorts-{width}.png'), full_page=True)
    # An unhandled /crops or /regions request would have failed the stub's fail-closed guard.
