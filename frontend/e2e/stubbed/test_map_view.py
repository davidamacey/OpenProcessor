"""The Map view of the Workspace in a real browser: MapLibre draws the served cells and pins, a
cell click shows the server's summary and first 12 image cards, Shift+drag turns an area into a
filter chip every view obeys, and the empty states say why there is nothing to draw.

JSON comes from the generated contract (`fill`). Basemap tiles are answered from a same-origin
path, so nothing leaves the machine. MapLibre needs WebGL2; the harness launches Chromium with
ANGLE on EGL (`harness/server.py`, D-20). Waits are real (`expect_request` / visible elements).
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill
from harness.constants import ACTION_TIMEOUT_MS, TRANSPARENT_GIF
from harness.screens import shot
from harness.waits import wait_for_paint
from playwright.sync_api import expect


CELL = (11.0, 48.0)
TILE = '12/2178/1420'
BOUNDS = {'west': 10.99, 'south': 47.99, 'east': 11.01, 'north': 48.01}
CANVAS = '[data-testid="map-canvas"] canvas.maplibregl-canvas'
SUMMARY_TEXT = '9 detections in 14 images'


def _meta(app_url: str, **over: Any) -> dict[str, Any]:
    base = {
        'available': True,
        'location_policy': 'store_coarse',
        'n_located': 1920,
        'n_without_location': 88,
        'bounds': fill('GeoBox', west=10.9, south=47.9, east=11.1, north=48.1),
        'tile_url': f'{app_url}/__tiles/{{z}}/{{x}}/{{y}}.gif',
        'tile_attribution': 'e2e tiles',
        'max_zoom': 12,
    }
    return fill('MapMeta', **{**base, **over})


def _cells(pin: str | None = None) -> dict[str, Any]:
    props = fill(
        'MapCellProperties',
        count=1 if pin else 14,
        tile=TILE,
        bounds=fill('GeoBox', **BOUNDS),
        image_id=pin,
        thumbnail_url=f'/curation/images/{pin}/thumbnail' if pin else None,
    )
    feature = fill(
        'MapCellFeature',
        geometry=fill('MapCellGeometry', coordinates=list(CELL)),
        properties=props,
    )
    return fill('MapCellCollection', features=[feature], precision=12)


def _card(i: int) -> dict[str, Any]:
    return fill(
        'ImageCard',
        image_id=f'img-{i}',
        file_name=f'photo-{i}.jpg',
        thumbnail_url=f'/__tiles/0/0/{i}.gif',
    )


def _serve_map(stub, page, app_url: str, meta: dict[str, Any], pin: str | None = None) -> dict:
    seen: dict[str, list] = {'cells': [], 'summary': [], 'images': []}

    def cells(request, _m):
        seen['cells'].append(request.url)
        return (200, _cells(pin), 'application/geo+json')

    def summary(request, _m):
        seen['summary'].append(request.post_data_json)
        return (200, fill('SelectionSummary', images=14, summary_text=SUMMARY_TEXT))

    def images(request, _m):
        seen['images'].append(request.url)
        page_ = fill('ImagePage', items=[_card(i) for i in range(12)], total=14)
        return (200, page_)

    def tile(route, _request):
        route.fulfill(status=200, content_type='image/gif', body=TRANSPARENT_GIF)

    stub.on('GET', r'/map$', (200, meta))
    stub.on('GET', r'/map/cells(\?|$)', cells)
    stub.on('POST', r'/selection/summary$', summary)
    stub.on('GET', r'/images(\?|$)', images)
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 60, 'crops': []})
    page.route('**/__tiles/**', tile)
    return seen


def _shots(page, name: str) -> None:
    """The same state at 1600 and 430 px, after the map has repainted at each size."""
    for width in (1600, 430):
        page.set_viewport_size({'width': width, 'height': 1000})
        wait_for_paint(page)
        shot(page, f'{name}-{width}', env_var='MAP_SHOTS')
    page.set_viewport_size({'width': 1280, 'height': 720})
    wait_for_paint(page)


def _open(page, app_url: str, query: str = '') -> Any:
    with page.expect_request(lambda r: '/map/cells' in r.url) as first:
        page.goto(f'{app_url}/p/default/workspace?view=map{query}')
    page.locator(CANVAS).wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    return first.value


def _centre(page) -> tuple[float, float]:
    box = page.locator(CANVAS).bounding_box()
    assert box is not None
    return box['x'] + box['width'] / 2, box['y'] + box['height'] / 2


def test_served_counts_and_a_cell_click_shows_the_summary_and_twelve_cards(stub, page, app_url):
    seen = _serve_map(stub, page, app_url, _meta(app_url))
    first = _open(page, app_url)
    assert page.get_by_test_id('map-located').inner_text() == '1,920 located'
    assert page.get_by_test_id('map-without-location').inner_text() == '88 without location'
    query = parse_qs(urlparse(first.url).query)
    west, south, east, north = (float(v) for v in query['bbox'][0].split(','))
    assert west < CELL[0] < east and south < CELL[1] < north

    x, y = _centre(page)
    with page.expect_request(lambda r: '/selection/summary' in r.url):
        page.mouse.click(x, y)
    expect(page.get_by_test_id('cell-summary')).to_have_text(
        SUMMARY_TEXT, timeout=ACTION_TIMEOUT_MS
    )
    assert len(page.get_by_test_id('cell-card').all()) == 12
    # The summary and the cards are asked once, for the served cell bounds.
    assert len(seen['summary']) == 1 and len(seen['images']) == 1
    assert seen['summary'][0]['filter']['geo_bbox'] == BOUNDS
    cards = parse_qs(urlparse(seen['images'][0]).query)
    assert [float(v) for v in cards['geo_bbox'][0].split(',')] == [
        BOUNDS['west'],
        BOUNDS['south'],
        BOUNDS['east'],
        BOUNDS['north'],
    ]
    assert cards['limit'] == ['12']
    assert f'focus=cell%3A{TILE.replace("/", "%2F")}' in page.url or f'cell:{TILE}' in page.url
    _shots(page, 'map_cell_open')

    page.get_by_test_id('cell-show').click()
    expect(page).to_have_url(re.compile(r'geo_bbox'), timeout=ACTION_TIMEOUT_MS)
    assert 'view=gallery' in page.url
    expect(page.get_by_test_id('view-tab-gallery')).to_have_attribute(
        'aria-selected', 'true', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_text('Area: box')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_a_single_photo_cell_is_a_pin_that_opens_the_image(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url), pin='img-pin')
    _open(page, app_url)
    x, y = _centre(page)
    page.mouse.click(x, y)
    expect(page).to_have_url(re.compile(r'img-pin'), timeout=ACTION_TIMEOUT_MS)
    assert 'cell' not in parse_qs(urlparse(page.url).query).get('focus', [''])[0]
    expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    _shots(page, 'map_pin')


def test_shift_drag_adds_an_area_chip_and_the_other_views_are_filtered(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    _open(page, app_url)
    x, y = _centre(page)
    page.keyboard.down('Shift')
    page.mouse.move(x - 120, y - 80)
    page.mouse.down()
    page.mouse.move(x, y, steps=4)
    page.mouse.move(x + 120, y + 80, steps=4)
    page.mouse.up()
    page.keyboard.up('Shift')
    expect(page).to_have_url(re.compile(r'geo_bbox'), timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('Area: box')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    _shots(page, 'map_area_chip')
    with page.expect_request(lambda r: '/crops' in r.url and 'geo_bbox' in r.url):
        page.get_by_test_id('view-tab-detections').click()


def test_the_lasso_button_then_a_plain_drag_adds_a_drawn_shape_chip(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    _open(page, app_url)
    x, y = _centre(page)
    page.get_by_test_id('map-lasso').click()
    page.mouse.move(x - 100, y - 60)
    page.mouse.down()
    page.mouse.move(x + 100, y - 60, steps=4)
    page.mouse.move(x + 100, y + 60, steps=4)
    page.mouse.move(x - 100, y + 60, steps=4)
    page.mouse.up()
    expect(page).to_have_url(re.compile(r'geo_polygon'), timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('Area: drawn shape')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_the_no_location_count_filters_the_workspace(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    _open(page, app_url)
    page.get_by_test_id('map-without-location').click()
    expect(page).to_have_url(re.compile(r'has_location=false'), timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('Without a location')).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_a_project_without_a_gps_position_says_so(stub, page, app_url):
    meta = _meta(app_url, available=False, n_located=0, n_without_location=3, bounds=None)
    _serve_map(stub, page, app_url, meta)
    page.goto(f'{app_url}/p/default/workspace?view=map')
    note = page.get_by_test_id('map-unavailable')
    note.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'GPS position' in note.inner_text()
    assert page.get_by_test_id('map-canvas').count() == 0
    shot(page, 'map_no_gps', env_var='MAP_SHOTS')


def test_the_drop_policy_says_locations_are_not_stored(stub, page, app_url):
    meta = _meta(app_url, available=False, location_policy='drop', n_located=0, bounds=None)
    seen = _serve_map(stub, page, app_url, meta)
    page.goto(f'{app_url}/p/default/workspace?view=map')
    note = page.get_by_test_id('map-unavailable')
    note.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'does not store where photos were taken' in note.inner_text()
    assert seen['cells'] == []
    shot(page, 'map_drop', env_var='MAP_SHOTS')


def test_the_map_fits_a_430px_screen(stub, page, app_url):
    page.set_viewport_size({'width': 430, 'height': 900})
    _serve_map(stub, page, app_url, _meta(app_url))
    _open(page, app_url)
    overflow = page.evaluate('() => document.documentElement.scrollWidth - window.innerWidth')
    assert overflow <= 0
    shot(page, 'map_430', env_var='MAP_SHOTS')


def test_without_webgl2_the_card_says_so_and_the_page_keeps_working(stub, page, app_url):
    page.add_init_script(
        """
        const getContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
          return type === 'webgl2' ? null : getContext.call(this, type, ...rest);
        };
        """
    )
    _serve_map(stub, page, app_url, _meta(app_url))
    page.goto(f'{app_url}/p/default/workspace?view=map')
    note = page.get_by_test_id('map-webgl-unsupported')
    note.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert note.inner_text().startswith('Map needs WebGL2')
    assert '1920 photos have GPS' in note.inner_text()
    assert page.get_by_test_id('map-located').inner_text() == '1,920 located'
    assert page.get_by_role('heading', name='Something went wrong').count() == 0
    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, errors[:3]
    shot(page, 'map_no_webgl', env_var='MAP_SHOTS')
