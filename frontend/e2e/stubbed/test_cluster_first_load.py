"""The first page of a large cluster can take seconds (the backend orders every
member). Until it arrives the view says it is ordering the crops after 1.5 s and
its footer reads 'loading…', not '0 / 0 listed' and 'all loaded'."""

from __future__ import annotations

import json

from conftest import ACTION_TIMEOUT_MS

from playwright.sync_api import expect

from test_cluster_discard import CLASSES, CLUSTERS, crop


def test_first_page_of_a_slow_cluster_says_what_it_is_doing(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})
    held: list = []
    # Registered after the stub's catch-all, so it wins: hold the crops page.
    page.route("**/crops?*cluster_id=*", lambda route: held.append(route))

    page.goto(f"{app_url}/p/default/clusters/1")
    hint = page.get_by_test_id("cluster-slow-load-hint")
    expect(hint).to_contain_text("A large cluster can take a few seconds.", timeout=ACTION_TIMEOUT_MS)
    expect(hint).to_contain_text("Ordering")
    footer = page.get_by_test_id("cluster-status-bar")
    expect(footer).to_contain_text("loading…")
    expect(footer).not_to_contain_text("all loaded")
    expect(footer).not_to_contain_text("0 / 0 listed")

    body = json.dumps({"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})
    for route in held:
        route.fulfill(status=200, content_type="application/json", body=body)
    expect(footer).to_contain_text("3 / 3 listed", timeout=ACTION_TIMEOUT_MS)
    expect(footer).to_contain_text("all loaded")
    expect(hint).to_have_count(0)
