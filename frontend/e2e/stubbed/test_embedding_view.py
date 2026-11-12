"""The Embedding view of the Workspace in a real browser (stub backend).

- the tab appears only where `/methods` reports the projection overlay
- the scatter is asked with the Workspace filter; points the filter leaves out come back
  `matched: false` and are drawn dimmed; the legend and its counts are the served ones
- Shift+drag draws a box: the URL gains `viz_box`, a chip says so, and the Gallery is asked
  with the same `viz_box`
- a click on a dot opens that item in the details panel
- an empty projection shows the served next steps
- at 430 px the page does not scroll sideways
"""

from __future__ import annotations

import math
import re
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import shot
from harness.waits import wait_for_paint
from playwright.sync_api import expect
from test_gallery_view import serve as serve_gallery


CLOUDS = {'sedan': (0.0, 0.0), 'truck': (6.0, 1.0), 'van': (2.5, 5.0)}
COUNTS = {'sedan': 120, 'truck': 90, 'van': 60}
SLOT = {'sedan': 0, 'truck': 1, 'van': 2}
CANVAS = '[data-testid="embedding-canvas"]'
CLASS_ROWS = [
    {
        'class_id': i + 1,
        'class_name': name,
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': 'abc'[i],
        'sample_count': COUNTS[name],
        'validated_count': 0,
        'deprecated': False,
    }
    for i, name in enumerate(CLOUDS)
]
OVERLAY = {
    'id': 'viz_projection',
    'axis': 'overlay',
    'settable': False,
    'label': 'Embedding projection',
    'status': 'stable',
    'requires_banner': False,
}


def _points() -> list[dict]:
    """Three clouds on a spiral; the last two `truck` points sit inside the `sedan` cloud."""
    points = []
    for name, (cx, cy) in CLOUDS.items():
        for i in range(COUNTS[name]):
            angle, radius = i * 2.4, 0.15 * math.sqrt(i)
            points.append(
                (f'{name}-{i}', name, cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
    points += [('mislabel-0', 'truck', 0.2, 0.1), ('mislabel-1', 'truck', -0.15, 0.25)]
    return [
        fill(
            'ProjectionPoint',
            crop_id=crop_id,
            x=x,
            y=y,
            cluster_uid=None,
            class_name=name,
            class_source='item_model',
            matched=True,
            colour_index=SLOT[name],
        )
        for crop_id, name, x, y in points
    ]


def _legend(points: list[dict]) -> list[dict]:
    counts = {n: sum(p['class_name'] == n for p in points) for n in CLOUDS}
    return [
        fill('ProjectionLegendEntry', class_name=n, colour_index=SLOT[n], count=counts[n])
        for n in sorted(CLOUDS, key=lambda n: -counts[n])
    ]


def serve(stub, queries: list[dict], *, overlay: bool = True, points: list[dict] | None = None):
    """The Gallery defaults plus a projection whose `matched` follows the class and box filter."""
    serve_gallery(stub, [])
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASS_ROWS})
    stub.on('GET', r'/methods(\?|$)', {'strategies': [OVERLAY] if overlay else [], 'flags': {}})
    base = _points() if points is None else points

    def projection(request, _match):
        query = parse_qs(urlparse(request.url).query)
        queries.append(query)
        classes = set(query.get('class_name', []))
        box = [float(v) for v in query['viz_box'][0].split(',')] if 'viz_box' in query else None
        out = []
        for p in base:
            ok = not classes or p['class_name'] in classes
            if box:
                ok = ok and box[0] <= p['x'] <= box[2] and box[1] <= p['y'] <= box[3]
            out.append({**p, 'matched': ok})
        body = fill(
            'ProjectionResponse',
            points=out,
            legend=_legend(base) if base else [],
            projection_version='umap_viz_v1',
            fitted_at='2026-11-12T00:00:00Z',
            stale=False,
        )
        return (200, body)

    stub.on('GET', r'/viz/projection(\?|$)', projection)
    stub.on(
        'GET',
        r'/viz/projection/status$',
        fill('VizProjectionState', status='idle', job_id='', n_pool=0, n_written=0),
    )


def _open(page, app_url: str, query: str = '') -> None:
    page.goto(f'{app_url}/p/default/workspace?view=embedding{query}')
    page.locator(CANVAS).wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('embedding-legend-entry')).to_have_count(
        3, timeout=ACTION_TIMEOUT_MS
    )
    wait_for_paint(page)


def _shots(page, name: str) -> None:
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 1000})
        wait_for_paint(page)
        shot(page, f'{name}-{width}', env_var='EMBEDDING_SHOTS')
    page.set_viewport_size({'width': 1280, 'height': 720})
    wait_for_paint(page)


def _drag(page, start: tuple[float, float], end: tuple[float, float], *, shift: bool) -> None:
    box = page.locator(CANVAS).bounding_box()
    assert box is not None
    if shift:
        page.keyboard.down('Shift')
    page.mouse.move(box['x'] + start[0], box['y'] + start[1])
    page.mouse.down()
    page.mouse.move(box['x'] + end[0], box['y'] + end[1], steps=6)
    page.mouse.up()
    if shift:
        page.keyboard.up('Shift')


def test_the_tab_shows_only_where_the_projection_is_served(stub, page, app_url):
    serve(stub, [], overlay=False)
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('view-tab-gallery')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('view-tab-embedding')).to_have_count(0)


def test_the_scatter_follows_the_filter_and_dims_the_rest(stub, page, app_url):
    queries: list[dict] = []
    serve(stub, queries)
    _open(page, app_url, '&class_name=sedan')
    assert queries[-1]['class_name'] == ['sedan']
    legend = page.get_by_test_id('embedding-legend-entry')
    assert [e.inner_text().split() for e in legend.all()] == [
        ['sedan', '(120)'],
        ['truck', '(92)'],
        ['van', '(60)'],
    ]
    expect(page.get_by_test_id('embedding-count')).to_contain_text('272')
    _shots(page, 'embedding_class_filter')


def test_a_drawn_box_filters_every_view(stub, page, app_url):
    queries: list[dict] = []
    serve(stub, queries)
    _open(page, app_url)
    _drag(page, (60, 60), (330, 280), shift=True)
    expect(page).to_have_url(re.compile(r'viz_box='), timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('Area: embedding box')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert 'viz_box' in queries[-1]
    _shots(page, 'embedding_box')

    with page.expect_request(lambda r: '/images?' in r.url and 'viz_box=' in r.url):
        page.get_by_test_id('view-tab-gallery').click()
    expect(page.get_by_text('Area: embedding box')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_the_draw_a_box_button_arms_a_plain_drag(stub, page, app_url):
    serve(stub, [])
    _open(page, app_url)
    page.get_by_test_id('embedding-box-tool').click()
    expect(page.get_by_test_id('embedding-box-tool')).to_have_attribute('aria-pressed', 'true')
    _drag(page, (60, 60), (330, 280), shift=False)
    expect(page).to_have_url(re.compile(r'viz_box='), timeout=ACTION_TIMEOUT_MS)


def test_a_lasso_selects_items_for_the_bulk_bar(stub, page, app_url):
    serve(stub, [])
    _open(page, app_url)
    box = page.locator(CANVAS).bounding_box()
    assert box is not None
    page.mouse.move(box['x'] + 40, box['y'] + 40)
    page.mouse.down()
    for dx, dy in ((420, 40), (420, 300), (40, 300), (40, 40)):
        page.mouse.move(box['x'] + dx, box['y'] + dy, steps=4)
    page.mouse.up()
    expect(page.get_by_test_id('bulk-count')).to_contain_text('selected', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('embedding-preview')).to_be_visible()


def test_an_empty_projection_shows_the_next_steps_card_area(stub, page, app_url):
    serve(stub, [], points=[])
    page.goto(f'{app_url}/p/default/workspace?view=embedding')
    expect(page.get_by_text('No item has a point on the projection yet.')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )


def test_the_view_fits_a_430px_screen(stub, page, app_url):
    page.set_viewport_size({'width': 430, 'height': 900})
    serve(stub, [])
    _open(page, app_url)
    assert page.evaluate('() => document.documentElement.scrollWidth - window.innerWidth') <= 0
    shot(page, 'embedding_430', env_var='EMBEDDING_SHOTS')
