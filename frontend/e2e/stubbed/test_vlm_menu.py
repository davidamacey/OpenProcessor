"""The Workspace's VLM menu: absent without a served VLM, one home for Assist,
the auto-label loop and the Auto-label run, usable at 430 px."""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import shot
from harness.vlm_menu import open_vlm_menu
from playwright.sync_api import expect
from test_assist_loop_panel import serve_loop
from test_assist_run_dialog import COMPLETED, METHODS


PATH = '/p/default/workspace?view=clusters'


def test_menu_is_absent_when_no_vlm_is_active(stub, page, app_url):
    serve_loop(stub, 'completed')
    stub.on('GET', r'/vlm/endpoints/active(\?|$)', {'source': 'off', 'active': {'name': None}})
    page.goto(f'{app_url}{PATH}')
    expect(page.get_by_test_id('view-bar')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state('networkidle', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('vlm-menu')).to_have_count(0)


def test_menu_holds_assist_loop_status_and_the_auto_label_run(stub, page, app_url):
    serve_loop(stub, 'running')
    stub.on('GET', r'/methods(\?|$)', METHODS)
    stub.on('POST', r'/assist/runs$', COMPLETED)
    open_vlm_menu(stub, page, app_url)
    expect(page.get_by_test_id('vlm-menu-loop-status')).to_have_text(
        'running', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('vlm-menu-assist')).to_contain_text('the scope')
    expect(page.get_by_test_id('loop-panel-toggle')).to_be_visible()
    expect(page.get_by_role('button', name='Auto-label now')).to_be_visible()
    page.get_by_test_id('vlm-menu-assist').click()
    expect(page.get_by_role('dialog')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shot(page, 'vlm-menu-assist-dialog', env_var='OP_E2E_SHOT_DIR', widths=(1600, 430))


def test_menu_screens_and_no_overflow_at_430(stub, page, app_url):
    serve_loop(stub, 'completed')
    open_vlm_menu(stub, page, app_url)
    shot(page, 'vlm-menu', widths=(1600, 430))
    page.set_viewport_size({'width': 430, 'height': 900})
    body = page.get_by_test_id('vlm-menu-body')
    expect(body).to_be_visible()
    box = body.bounding_box()
    assert box is not None and box['x'] >= 0 and box['x'] + box['width'] <= 430, box
    assert page.evaluate('document.documentElement.scrollWidth') <= 430
