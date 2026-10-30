"""The cluster page's facet chips: counts are the served ones, and a chip sends
the matching `GET /crops` parameter."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import expect_handled
from playwright.sync_api import expect

from fixtures.wire import make_item
from test_cluster_list_filters import CLASSES, METHODS, clusters_handler

FACETS = {
    "cluster_id": 2,
    "total": 40,
    "detector_classes": [{"value": "widget_x", "count": 31}, {"value": "widget_y", "count": 0}],
    "class_sources": [],
    "validation": {"validated": 4, "unvalidated": 36},
    "review_tabs": {"outliers": 3, "detector_disagreements": 0},
    "detector_confidence": [
        {"min": 0.0, "max": 0.1, "count": 0},
        {"min": 0.9, "max": None, "count": 25},
    ],
    "detector_confidence_missing": 2,
}


def crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=None,
        class_name=None,
        cluster_id=2,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def serve_cluster(stub, crop_queries: list[dict]) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/clusters(\?|$)", clusters_handler([]))
    stub.on("GET", r"/clusters/2/facets$", FACETS)

    def crops(request, _match):
        crop_queries.append(parse_qs(urlparse(request.url).query))
        return (200, {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})

    stub.on("GET", r"/crops(\?|$)", crops)


def test_facet_chips_show_served_counts_and_send_the_served_filters(stub, page, app_url):
    queries: list[dict] = []
    serve_cluster(stub, queries)
    page.goto(f"{app_url}/p/default/clusters/2")
    facets = page.get_by_test_id("cluster-facets")
    expect(facets).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(facets.get_by_test_id("facet-detector-class")).to_have_text(["widget_x 31", "widget_y 0"])
    expect(facets.get_by_test_id("facet-confidence-bin")).to_have_text(["0.0–0.1 0", "0.9+ 25"])
    expect(facets.get_by_test_id("facet-confidence-missing")).to_have_text("none: 2")

    def click_and_wait(locator, param: str) -> dict:
        with expect_handled(
            page, lambda r: "/crops" in r.url and param in r.url, timeout=ACTION_TIMEOUT_MS
        ):
            locator.click()
        return queries[-1]

    q = click_and_wait(facets.get_by_test_id("facet-detector-class").first, "detector_class_name")
    assert q["detector_class_name"] == ["widget_x"]
    expect(facets.get_by_test_id("facet-detector-class").first).to_have_attribute("aria-pressed", "true")

    q = click_and_wait(facets.get_by_test_id("facet-confidence-bin").nth(1), "detector_conf_min")
    assert q["detector_conf_min"] == ["0.9"]
    assert "detector_conf_max" not in q

    q = click_and_wait(facets.get_by_test_id("facet-unvalidated"), "label_validated")
    assert q["label_validated"] == ["false"]

    q = click_and_wait(facets.get_by_test_id("facet-review-tab").first, "review_tab")
    assert q["review_tab"] == ["outliers"]

    with expect_handled(page, lambda r: "/crops" in r.url and "sort=" in r.url, timeout=ACTION_TIMEOUT_MS):
        page.get_by_test_id("facet-sort").select_option("desc")
    assert queries[-1]["sort"] == ["detector_confidence:desc"]
