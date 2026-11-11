"""`/map` draws a real MapLibre canvas on a project that has located photos.

The page degrades to a "Map needs WebGL2" note when the browser has no WebGL2; that note must
never be what this test sees, or a real render failure would pass as a graceful fallback.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import api_get, page_path


def test_map_renders_a_canvas_when_photos_have_gps(
    guarded_page: Any, live_url: str, live_project: dict[str, Any]
) -> None:
    meta = api_get(live_url, live_project, '/map')
    if not meta.get('available'):
        pytest.skip('the project has no served map (no GPS photos, or a drop location policy)')
    page = guarded_page.page
    page.goto(f'{live_url}{page_path(live_project, "/map")}', wait_until='domcontentloaded')

    canvas = page.locator('[data-testid="map-canvas"] canvas.maplibregl-canvas')
    canvas.wait_for(state='visible')
    box = canvas.bounding_box()
    assert box is not None and box['width'] > 100 and box['height'] > 100, box
    assert page.locator('[data-testid="map-webgl-unsupported"]').count() == 0, (
        'the WebGL2 fallback showed in a browser that should have software WebGL2'
    )
    assert page.evaluate("!!document.createElement('canvas').getContext('webgl2')")
    page.locator('[data-testid="map-located"]').wait_for(state='visible')
    assert str(meta['n_located']) in page.locator('[data-testid="map-located"]').inner_text()
