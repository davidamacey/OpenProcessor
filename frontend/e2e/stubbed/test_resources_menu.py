"""The top-bar Resources menu lists the client-owned Documentation entry and
then exactly the served `resource_links` (docs entries path-relative, service
entries absolute), and never causes horizontal overflow at 800px."""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, DOCS_RESOURCE_LINKS, resource_link
from test_labeling_flow import register_base


def serve(stub, links):
    stub.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None, "resource_links": links})


def open_menu(page, app_url):
    page.goto(f"{app_url}/p/default/dashboard")
    page.get_by_test_id("resources-trigger").click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("resources-list").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_resources_menu_hrefs(stub, page, app_url):
    register_base(stub)
    serve(stub, DOCS_RESOURCE_LINKS + [resource_link("grafana", "Grafana", "http://grafana.example:3000")])
    open_menu(page, app_url)
    links = page.get_by_test_id("resource-link")
    hrefs = [a.get_attribute("href") for a in links.all()]
    # Documentation is the one client-owned entry; docs entries stay relative.
    assert hrefs == ["/OpenProcessor/docs/cropwright/getting-started/introduction", "/docs", "/redoc", "/openapi.json", "http://grafana.example:3000"]
    for a in links.all():
        assert a.get_attribute("target") == "_blank"
        assert a.get_attribute("rel") == "noopener noreferrer"


def test_not_configured_entry_is_a_muted_row_with_the_hint(stub, page, app_url):
    register_base(stub)
    serve(stub, [resource_link("grafana", "Grafana", None, hint="Set OP_GRAFANA_URL")])
    open_menu(page, app_url)
    row = page.get_by_test_id("resource-muted")
    assert row.inner_text().strip() == "Grafana: not configured"
    assert row.get_attribute("title") == "Set OP_GRAFANA_URL"
    assert page.get_by_test_id("resources-list").locator("a").count() == 1  # Documentation only


def test_unreachable_service_is_noted_not_running(stub, page, app_url):
    register_base(stub)
    serve(stub, [
        resource_link("grafana", "Grafana", "http://g:3000", reachable=False),
        resource_link("prometheus", "Prometheus", "http://p:9090", reachable=None),
    ])
    open_menu(page, app_url)
    notes = page.get_by_test_id("resource-not-running")
    assert notes.count() == 1
    assert "not running" in notes.inner_text()


def test_unsafe_served_url_renders_no_anchor(stub, page, app_url):
    register_base(stub)
    serve(stub, [resource_link("grafana", "Grafana", "javascript:alert(1)")])
    open_menu(page, app_url)
    assert "javascript:" not in page.get_by_test_id("resources-list").inner_html()
    assert page.get_by_test_id("resources-list").locator("a").count() == 1


def test_resources_menu_narrow_no_overflow_and_escape(stub, page, app_url):
    register_base(stub)
    page.set_viewport_size({"width": 800, "height": 900})
    open_menu(page, app_url)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    assert page.get_by_test_id("resources-trigger").get_attribute("aria-expanded") == "true"
    page.keyboard.press("Escape")
    page.get_by_test_id("resources-list").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("resources-trigger").get_attribute("aria-expanded") == "false"
    assert page.evaluate("document.activeElement.dataset.testid") == "resources-trigger"
