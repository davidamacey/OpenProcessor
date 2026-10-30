"""axe-core on the main pages, against the stub: any violation fails.

axe is injected from `node_modules` (devDependency `axe-core`), so the check is
pinned and offline. Every page is rendered in a state that shows real content
(rows, cards, tables), because an empty page hides most violations. Later
waves add their new page states here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from fixtures.wire import project, projects_response
from test_cluster_discard import CLASSES, CLUSTERS, crop
from test_assist_loop_panel import serve_loop, serve_plan_form, show_finished_loop, show_plan_form
from test_cluster_facets import serve_cluster
from test_cluster_list_filters import serve as serve_filtered_clusters
from test_fresh_run_cosmetics import EMPTY_PROPOSALS
from test_curation_settings import METHODS_TODAY, SETTINGS_EMPTY
from test_open_vocab import serve_open_vocab
from test_prompt_packs import serve_packs
from test_review_crop_viewport import review_item

AXE_JS = Path(__file__).resolve().parents[2] / "node_modules" / "axe-core" / "axe.min.js"
API_PREFIX = "/curation"


def serve_clusters(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})


def serve_projects_page(stub) -> None:
    rows = [
        project(API_PREFIX, "default", is_default=True, deletable=False),
        project(API_PREFIX, "widgets-a", display_name="Widgets A"),
    ]
    stub.on("GET", rf"^{API_PREFIX}/projects$", (200, projects_response(API_PREFIX, rows)))


def serve_review(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        (200, {"items": [review_item(i) for i in range(3)], "total": 3, "page": 1, "page_size": 30}),
    )
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        (200, {"image": {"path": "/img.jpg", "width": 640, "height": 480}, "items": [review_item(0)]}),
    )


def serve_classes(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})


def serve_settings(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})
    stub.on("GET", r"/methods(\?|$)", METHODS_TODAY)
    stub.on("GET", r"/settings(\?|$)", SETTINGS_EMPTY)


def serve_prompt_packs(stub) -> None:
    serve_packs(stub)
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})


def serve_open_vocab_list(stub) -> None:
    serve_open_vocab(stub)
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})


def serve_clusters_with_filters(stub) -> None:
    serve_filtered_clusters(stub, [])


def serve_cluster_with_facets(stub) -> None:
    serve_cluster(stub, [])


# (id, path, stub setup, selector that proves the data loaded[, step that opens a panel])
PAGES = [
    ("projects", "/projects", serve_projects_page, '[data-testid="project-row-widgets-a"]'),
    ("clusters", "/p/default/clusters", serve_clusters, "li button"),
    ("clusters-with-filters", "/p/default/clusters?flag=unvalidated", serve_clusters_with_filters, "li button"),
    ("clusters-loop-plan", "/p/default/clusters", serve_plan_form, '[data-testid="loop-panel-toggle"]', show_plan_form),
    (
        "clusters-loop-finished",
        "/p/default/clusters",
        lambda stub: serve_loop(stub, "completed"),
        '[data-testid="loop-panel-toggle"]',
        show_finished_loop,
    ),
    (
        "clusters-loop-live",
        "/p/default/clusters",
        lambda stub: serve_loop(stub, "running"),
        '[data-testid="loop-panel-toggle"]',
        lambda page: expect(page.get_by_test_id("loop-cancel")).to_be_visible(timeout=ACTION_TIMEOUT_MS),
    ),
    ("cluster-detail", "/p/default/clusters/1", serve_clusters, '[data-testid="class-name"]'),
    ("cluster-detail-facets", "/p/default/clusters/2", serve_cluster_with_facets, '[data-testid="cluster-facets"]'),
    ("review", "/p/default/review", serve_review, "main"),
    ("classes", "/p/default/classes", serve_classes, "main"),
    ("settings", "/p/default/settings", serve_settings, "main"),
    ("settings-prompt-packs", "/p/default/settings/prompt-packs", serve_prompt_packs, "table"),
    ("settings-open-vocab", "/p/default/settings/open-vocab", serve_open_vocab_list, "table"),
]


@pytest.mark.parametrize("path,setup,ready,reveal", [(*p[1:4], p[4] if len(p) > 4 else None) for p in PAGES], ids=[p[0] for p in PAGES])
def test_page_has_no_axe_violations(stub, page, app_url, path, setup, ready, reveal):
    setup(stub)
    page.goto(f"{app_url}{path}")
    expect(page.locator(ready).first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    if reveal is not None:
        reveal(page)
    page.wait_for_load_state("networkidle")
    page.add_script_tag(path=str(AXE_JS))
    result = page.evaluate("axe.run(document, { resultTypes: ['violations'] })")
    found = [
        {"rule": v["id"], "impact": v["impact"], "nodes": [n["target"] for n in v["nodes"][:5]]}
        for v in result["violations"]
    ]
    assert found == [], json.dumps(found, indent=2)
