"""The type system in a real browser (stubbed API): bundled faces, and mono only where it belongs.

- body text resolves to the bundled Inter Variable face, and the browser has loaded it
- `<code>` resolves to the bundled JetBrains Mono Variable face
- on each swept page no element outside `code, kbd, pre, samp, textarea, input, select` resolves a
  monospace family: ids go in `<code>`, numbers stay in the sans face
- an interactive chip keeps 4px from its neighbours (WCAG 2.5.8 spacing exception for a target
  under 24px)

jsdom has no fonts or layout, so this is a browser test. With `OP_E2E_SHOT_DIR` set it also
writes the full-page screenshots of each page at 1600 and 430 px.
"""

from __future__ import annotations

import pytest
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.screens import shot
from harness.waits import wait_for_paint
from playwright.sync_api import expect
from test_narrow_viewport_430 import LONG_SLUG, _stub_common
from test_review_default_queue import item as review_item, queue_body
from test_review_view import serve as serve_review


GALLERY = ('gallery', '/workspace?view=gallery', "[data-testid='gallery-view']")
REVIEW = ('review', '/workspace?view=review', "[data-testid='queue-counter']")
PAGES = [
    ('dashboard', '/dashboard', 'h1'),
    GALLERY,
    ('detections', '/workspace?view=detections', "[data-testid='detections-view']"),
    REVIEW,
    ('export', '/export', 'h1'),
    ('train', '/train', 'h1'),
    ('settings', '/settings', 'h1'),
    ('ingest', '/ingest', 'h1'),
]

FAMILY = """(selector) => {
  const el = document.querySelector(selector);
  return el ? getComputedStyle(el).fontFamily : null;
}"""

# Elements that own text but are not in a code context and still resolve a monospace family.
STRAY_MONO = """() => {
  const codeContext = 'code, kbd, samp, pre, textarea, input, select';
  const stray = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (el.closest(codeContext)) continue;
    const ownsText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (!ownsText) continue;
    const family = getComputedStyle(el).fontFamily;
    if (/mono|courier|consolas|menlo/i.test(family)) {
      stray.push(el.tagName + '.' + String(el.className).slice(0, 60) + ' | '
        + el.textContent.trim().slice(0, 30));
    }
  }
  return stray.slice(0, 8);
}"""

CHIP_GAPS = """() => {
  const chips = [...document.querySelectorAll('button.chip, a.chip, summary.chip')];
  return chips.map((c) => {
    const s = getComputedStyle(c);
    return [s.marginTop, s.marginRight, s.marginBottom, s.marginLeft];
  });
}"""


def _open(stub, page, app_url, path: str, ready: str) -> None:
    _stub_common(stub)
    page.goto(f'{app_url}/p/{LONG_SLUG}{path}')
    page.locator(ready).first.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)


def test_body_text_is_inter_and_the_browser_loaded_it(stub, page, app_url):
    _open(stub, page, app_url, *GALLERY[1:])
    assert page.evaluate(FAMILY, 'body').startswith('"Inter Variable"')
    page.evaluate('document.fonts.ready.then(() => true)')
    assert page.evaluate('document.fonts.check(\'14px "Inter Variable"\')') is True
    loaded = page.evaluate(
        "[...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family)"
    )
    assert any('Inter Variable' in family for family in loaded), loaded


def test_code_is_jetbrains_mono_and_the_browser_loaded_it(stub, page, app_url):
    # The crop id of the Review pane is the one `<code>` a served item brings to the page.
    serve_review(stub)
    stub.on('GET', r'/review/to_label(\?|$)', queue_body([review_item(0)]))
    page.goto(review_url(app_url, 'to_label'))
    code = page.get_by_test_id('review-crop-panel').locator('code').first
    expect(code).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.evaluate(FAMILY, 'code').startswith('"JetBrains Mono Variable"')
    page.evaluate('document.fonts.ready.then(() => true)')
    assert page.evaluate('document.fonts.check(\'13px "JetBrains Mono Variable"\')') is True


@pytest.mark.parametrize(('name', 'path', 'ready'), PAGES, ids=[p[0] for p in PAGES])
def test_no_monospace_outside_code_and_screenshots(stub, page, app_url, name, path, ready):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    _open(stub, page, app_url, path, ready)
    assert page.evaluate(STRAY_MONO) == []
    shot(page, f'{name}', widths=(1600,), height=1000)
    shot(page, f'{name}', widths=(430,), height=900)


def test_interactive_chips_keep_four_pixels_from_a_neighbour(stub, page, app_url):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    _open(stub, page, app_url, *GALLERY[1:])
    page.get_by_test_id('gallery-view').wait_for(timeout=ACTION_TIMEOUT_MS)
    gaps = page.evaluate(CHIP_GAPS)
    assert gaps, 'the gallery stub shows no interactive chip to measure'
    for margins in gaps:
        assert margins == ['2px'] * 4, margins
