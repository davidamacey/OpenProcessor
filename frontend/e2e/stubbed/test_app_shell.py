"""The app shell: the grouped top nav, the breadcrumb, the mobile drawer and the 404 page.

Real layout is needed (the bar's height, what fits at 800px, the drawer below 768px), so this runs
in a browser against the stub. Fail-closed: every request the pages make must be stubbed.
"""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import wait_for_paint
from playwright.sync_api import expect


DESKTOP = {'width': 1600, 'height': 1000}
TABLET = {'width': 800, 'height': 1000}
NARROW = {'width': 430, 'height': 900}


def _stub_classes(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/review/new_class_proposals/summary(\?|$)', {'total': 0, 'proposals': []})
    stub.on('GET', r'/test_holdout/stats(\?|$)', {'total': 0, 'by_class': []})


def _overflow(page) -> int:
    return page.evaluate('document.documentElement.scrollWidth - window.innerWidth')


def test_the_bar_is_48px_on_the_nav_surface_with_a_shadow(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/classes')
    bar = page.locator('header').first
    bar.wait_for(timeout=ACTION_TIMEOUT_MS)
    box = bar.bounding_box()
    assert box and round(box['height']) == 48, box
    style = bar.evaluate(
        'el => ({shadow: getComputedStyle(el).boxShadow, blur: getComputedStyle(el).backdropFilter})'
    )
    assert style['shadow'] != 'none', style
    assert 'blur' in style['blur'], style


def test_a_group_opens_its_pages(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/classes')
    nav = page.get_by_test_id('primary-nav')
    nav.wait_for(timeout=ACTION_TIMEOUT_MS)
    nav.get_by_role('button', name='Data').click()
    items = nav.get_by_role('menuitem')
    expect(items).to_have_text(['Ingest', 'Classes'])
    expect(nav.get_by_role('menuitem', name='Classes')).to_have_attribute('aria-current', 'page')
    page.keyboard.press('Escape')
    expect(nav.get_by_role('menuitem')).to_have_count(0)
    expect(nav.get_by_role('button', name='Data')).to_be_focused()


def test_the_bar_fits_800px_without_overflow(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(TABLET)
    page.goto(f'{app_url}/p/default/classes')
    page.get_by_test_id('primary-nav').wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    assert _overflow(page) <= 1
    bar = page.locator('header').first.bounding_box()
    assert bar and round(bar['height']) == 48, bar


def test_below_768px_the_nav_is_a_drawer(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(NARROW)
    page.goto(f'{app_url}/p/default/classes')
    open_btn = page.get_by_role('button', name='Open navigation')
    open_btn.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('primary-nav')).to_be_hidden()
    assert _overflow(page) <= 1
    open_btn.click()
    drawer = page.get_by_test_id('drawer')
    expect(drawer).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(drawer.get_by_role('link')).to_have_count(10)
    expect(drawer.get_by_role('link', name='Classes')).to_have_attribute('aria-current', 'page')
    page.keyboard.press('Escape')
    expect(drawer).to_be_hidden()


def test_an_unknown_page_in_a_project_is_a_404_with_the_nav(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/no-such-page')
    expect(page.get_by_role('heading', name='Page not found')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('primary-nav')).to_be_visible()
    expect(page.get_by_role('main').get_by_role('link', name='Dashboard')).to_have_attribute(
        'href', '/p/default/dashboard'
    )
