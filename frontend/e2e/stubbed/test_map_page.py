"""`/p/{project}/map` in a real browser: MapLibre draws the served grid cells,
a click on a cell lists its photos, and the empty states say why there is
nothing to draw (no GPS position, or the `drop` location policy).

The page's JSON comes from the generated contract (`fill`). Basemap tiles are
requested from a same-origin path that the test answers, so nothing leaves the
machine. MapLibre needs WebGL2, which headless Chromium only gets with an X server
behind it: run this file as `xvfb-run -a <python> -m pytest ...` (no display is
shown).

Waits are real (`expect_request` / visible elements), never a sleep.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fixtures.contract import fill
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS, TRANSPARENT_GIF
from harness.waits import wait_for_paint


CELL = (11.0, 48.0)
TILE = '12/2178/1420'
IMAGE_IDS = ['img-a', 'img-b']


def _meta(app_url: str, **over: Any) -> dict[str, Any]:
    base = {
        'available': True,
        'location_policy': 'store_coarse',
        'n_located': 2,
        'bounds': fill(
            'MapBounds',
            west=CELL[0] - 0.01,
            south=CELL[1] - 0.01,
            east=CELL[0] + 0.01,
            north=CELL[1] + 0.01,
        ),
        'tile_url': f'{app_url}/__tiles/{{z}}/{{x}}/{{y}}.gif',
        'tile_attribution': 'e2e tiles',
        'max_zoom': 12,
    }
    return fill('MapMeta', **{**base, **over})


def _cells() -> dict[str, Any]:
    feature = fill(
        'MapCellFeature',
        geometry=fill('MapCellGeometry', coordinates=list(CELL)),
        properties=fill('MapCellProperties', count=len(IMAGE_IDS), tile=TILE),
    )
    return fill('MapCellCollection', features=[feature], precision=12)


def _serve_map(stub, page, app_url: str, meta: dict[str, Any]) -> dict[str, list]:
    seen: dict[str, list] = {'cells': [], 'images': [], 'crops': [], 'tiles': []}

    def cells(request, _m):
        seen['cells'].append(request.url)
        return (200, _cells())

    def images(request, _m):
        seen['images'].append(request.url)
        return (
            200,
            fill('MapTileImages', tile=TILE, count=2, image_ids=IMAGE_IDS, truncated=False),
        )

    def crops(request, _m):
        seen['crops'].append(request.url)
        rows = [make_item(crop_id=f'crop-{i}', image_id=i) for i in IMAGE_IDS]
        return (200, {'total': len(rows), 'page': 1, 'page_size': 48, 'crops': rows})

    def tile(route, request):
        seen['tiles'].append(request.url)
        route.fulfill(status=200, content_type='image/gif', body=TRANSPARENT_GIF)

    stub.on('GET', r'/map$', (200, meta))
    stub.on('GET', r'/map/cells(\?|$)', cells)
    stub.on('GET', r'/map/images(\?|$)', images)
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/crops(\?|$)', crops)
    page.route('**/__tiles/**', tile)
    return seen


def _shot(page, name: str) -> None:
    out = os.environ.get('MAP_SHOTS')
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=f'{out}/{name}.png', full_page=True)


def test_cells_render_and_a_click_lists_the_cells_photos(stub, page, app_url):
    seen = _serve_map(stub, page, app_url, _meta(app_url))
    with page.expect_request(lambda r: '/map/cells' in r.url) as first:
        page.goto(f'{app_url}/p/default/map')
    page.get_by_test_id('map-located').wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id('map-located').inner_text() == '2 located photos'
    assert 'coarse' in page.get_by_test_id('map-policy').inner_text()
    # The viewport the map reports is what the server is asked for.
    query = parse_qs(urlparse(first.value.url).query)
    assert query['zoom'] == ['10']
    west, south, east, north = (float(v) for v in query['bbox'][0].split(','))
    assert west < CELL[0] < east
    assert south < CELL[1] < north

    canvas = page.locator('[data-testid="map-canvas"] canvas.maplibregl-canvas')
    canvas.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    assert page.get_by_test_id('map-cell-panel').count() == 0

    # The only cell sits at the centre of the fitted view.
    box = canvas.bounding_box()
    assert box is not None
    with page.expect_request(lambda r: '/map/images' in r.url) as opened:
        page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
    assert parse_qs(urlparse(opened.value.url).query)['tile'] == [TILE]
    page.get_by_test_id('map-cell-panel').wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('matching-total').filter(has_text='2').wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'photos here' in page.get_by_test_id('map-cell-panel').inner_text()
    crops_query = parse_qs(urlparse(seen['crops'][-1]).query)
    assert crops_query['image_id'] == IMAGE_IDS
    assert seen['tiles'], 'the basemap was requested from the served template'
    _shot(page, 'map_cell_open')


def test_a_project_without_a_gps_position_says_so(stub, page, app_url):
    meta = _meta(app_url, available=False, n_located=0, bounds=None)
    _serve_map(stub, page, app_url, meta)
    page.goto(f'{app_url}/p/default/map')
    note = page.get_by_test_id('map-unavailable')
    note.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'GPS position' in note.inner_text()
    assert page.get_by_test_id('map-canvas').count() == 0
    _shot(page, 'map_no_gps')


def test_the_drop_policy_says_locations_are_not_stored(stub, page, app_url):
    meta = _meta(app_url, available=False, location_policy='drop', n_located=0, bounds=None)
    seen = _serve_map(stub, page, app_url, meta)
    page.goto(f'{app_url}/p/default/map')
    note = page.get_by_test_id('map-unavailable')
    note.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'does not store where photos were taken' in note.inner_text()
    assert page.get_by_test_id('map-canvas').count() == 0
    assert seen['cells'] == []
    _shot(page, 'map_drop')


def test_the_map_fits_a_430px_screen(stub, page, app_url):
    page.set_viewport_size({'width': 430, 'height': 900})
    _serve_map(stub, page, app_url, _meta(app_url))
    page.goto(f'{app_url}/p/default/map')
    page.locator('[data-testid="map-canvas"] canvas.maplibregl-canvas').wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    wait_for_paint(page)
    overflow = page.evaluate('() => document.documentElement.scrollWidth - window.innerWidth')
    assert overflow <= 0
    _shot(page, 'map_430')
