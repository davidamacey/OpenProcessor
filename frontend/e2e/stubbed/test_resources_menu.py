"""The top-bar Resources menu links the bundled docs and the API's own docs
(same-origin nginx proxies) plus only the monitoring dashboards the backend
serves, and never causes horizontal overflow at 800px."""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS
from test_labeling_flow import register_base

SERVED = {"grafana": "http://grafana.example:3000", "prometheus": None, "opensearch_dashboards": None}


def open_menu(page, app_url):
    page.goto(f"{app_url}/p/default/dashboard")
    page.get_by_test_id("resources-trigger").click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("resources-list").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_resources_menu_hrefs(stub, page, app_url):
    register_base(stub)
    stub.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None, "monitoring_links": SERVED})
    open_menu(page, app_url)
    links = page.get_by_test_id("resource-link")
    hrefs = [a.get_attribute("href") for a in links.all()]
    assert hrefs == ["/cropwright/", "/docs", "/redoc", "/openapi.json", "http://grafana.example:3000"]
    for a in links.all():
        assert a.get_attribute("target") == "_blank"
        assert a.get_attribute("rel") == "noopener noreferrer"


def test_resources_menu_narrow_no_overflow_and_escape(stub, page, app_url):
    register_base(stub)
    page.set_viewport_size({"width": 800, "height": 900})
    open_menu(page, app_url)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    page.keyboard.press("Escape")
    page.get_by_test_id("resources-list").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert page.evaluate("document.activeElement.dataset.testid") == "resources-trigger"
