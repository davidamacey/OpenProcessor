"""OpenProcessor v0.4.0 shared item filter (docs/design/
v040-backend-deltas-ui-plan-2026-10-03.md section 8, addendum V-12): class
is sent by NAME (`class_name`, repeated), never `class_id`, and each route
only gets the params it serves.

- /review: the bar's controls follow the tab's served `filters`; picking a
  class and an origin reaches `GET /review/{tab}` as `class_name` / `origin`
  and persists in the URL.
- /clusters: the same bar scopes the cluster grid (`GET /clusters`) and the
  "Matching items" mode (`GET /crops`, which alone takes `open_vocab_set` /
  `source_prompt`); a URL carrying them opens that mode with removable chips.
- /export: a non-empty filter sends `item_filter` and shows the served
  `total_crops`; an empty one sends no `item_filter` key.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item, review_tab, review_tabs

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget",
        "kind": "item",
        "group": "g",
        "hotkey_letter": "k",
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
    {
        "class_id": 2,
        "class_name": "gadget",
        "kind": "item",
        "group": "g",
        "hotkey_letter": "j",
        "sample_count": 4,
        "validated_count": 1,
        "cluster_size": 4,
        "deprecated": False,
    },
]

CLUSTERS = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 2,
            "validated_count": 0,
            "dominant_class_id": 1,
            "dominant_class_name": "widget",
            "purity": 1.0,
            "is_unlabeled": False,
            "representatives": [],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}


def item(i: int) -> dict:
    return make_item(crop_id=f"crop-{i}", image_id=f"img-{i}", class_id=1, class_name="widget")


def query_of(request) -> dict[str, list[str]]:
    return parse_qs(urlparse(request.url).query)


def _common(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})


def test_review_class_and_origin_reach_the_queue_by_name_and_persist_in_the_url(stub, page, app_url):
    stub.on(
        "GET",
        r"/review/tabs(\?|$)",
        review_tabs(
            review_tab("all", "All", filters=["class_name", "origin", "source"]),
        ),
    )
    review_requests: list = []

    def review_handler(request, _match):
        review_requests.append(request)
        return (200, {"items": [item(0)], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)
    _common(stub)

    page.goto(f"{app_url}/p/default/review?tab=all")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    with page.expect_request(
        lambda r: "/review/all" in r.url and "class_name=gadget" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id("served-filter-class_name").locator("select").select_option("gadget")
    page.get_by_test_id("item-filter-more").locator("summary").click()
    with page.expect_request(
        lambda r: "/review/all" in r.url and "origin=sam3" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ) as sent:
        page.get_by_test_id("served-filter-origin").get_by_role("button", name="Sam3").click()

    last = query_of(sent.value)
    assert last["class_name"] == ["gadget"]
    assert last["origin"] == ["sam3"]
    assert all("class_id" not in query_of(r) for r in review_requests + [sent.value])
    page.wait_for_function(
        "() => location.search.includes('class_name=gadget') && location.search.includes('origin=sam3')",
        timeout=ACTION_TIMEOUT_MS,
    )


def test_review_generic_served_specs_render_by_kind(stub, page, app_url):
    """A served spec the page knows nothing about renders by its kind and is
    sent under its own param (a number with the served bounds)."""
    stub.on(
        "GET",
        r"/review/tabs(\?|$)",
        review_tabs(
            review_tab(
                "all",
                "All",
                filters=["class_name", "min_things"],
                filter_specs=[
                    {
                        "param": "min_things",
                        "kind": "integer",
                        "label": "At least N things",
                        "options": [],
                        "min": 1,
                        "max": 9,
                        "description": "Served help",
                        "default": None,
                        "allows_unset": True,
                    },
                    {
                        "param": "flagged",
                        "kind": "bool",
                        "label": "Flagged",
                        "options": [],
                        "min": None,
                        "max": None,
                        "description": "",
                        "default": None,
                        "allows_unset": True,
                    },
                ],
            ),
        ),
    )
    requests: list = []

    def review_handler(request, _match):
        requests.append(request)
        return (200, {"items": [item(0)], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)
    _common(stub)

    page.goto(f"{app_url}/p/default/review?tab=all")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    field = page.get_by_test_id("served-filter-min_things")
    field.wait_for(timeout=ACTION_TIMEOUT_MS)
    number = field.locator("input")
    assert number.get_attribute("min") == "1" and number.get_attribute("max") == "9"
    assert field.get_attribute("title") == "Served help"
    with page.expect_request(
        lambda r: "/review/all" in r.url and "min_things=3" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        number.fill("3")
        number.blur()
    with page.expect_request(
        lambda r: "/review/all" in r.url and "flagged=true" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id("served-filter-flagged").locator("select").select_option("true")


def test_clusters_grid_and_matching_mode_send_the_filter(stub, page, app_url):
    cluster_requests: list = []
    crop_requests: list = []

    def clusters_handler(request, _match):
        cluster_requests.append(request)
        return (200, CLUSTERS)

    def crops_handler(request, _match):
        crop_requests.append(request)
        return (200, {"total": 1, "page": 1, "page_size": 48, "crops": [item(0)]})

    stub.on("GET", r"/clusters(\?|$)", clusters_handler)
    stub.on("GET", r"/crops(\?|$)", crops_handler)
    _common(stub)

    page.goto(f"{app_url}/p/default/clusters")
    page.get_by_test_id("clusters-filter-bar").wait_for(timeout=ACTION_TIMEOUT_MS)
    # The open-vocabulary pair belongs to the Matching view only.
    assert page.get_by_test_id("served-filter-open_vocab_set").count() == 0

    with page.expect_request(
        lambda r: "/clusters" in r.url and "class_name=gadget" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id("served-filter-class_name").locator("select").select_option("gadget")
    page.get_by_test_id("item-filter-more").locator("summary").click()
    with page.expect_request(
        lambda r: "/clusters" in r.url and "origin=human" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ) as sent:
        page.get_by_test_id("served-filter-origin").get_by_role("button", name="Human").click()
    grid = query_of(sent.value)
    assert grid["class_name"] == ["gadget"] and grid["origin"] == ["human"]
    assert "open_vocab_set" not in grid and "class_id" not in grid

    # Matching items: the same filter, now on GET /crops, plus the set/prompt.
    with page.expect_request(
        lambda r: "/crops" in r.url and "class_name=gadget" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ) as sent:
        page.get_by_test_id("matching-mode-toggle").click()
    page.get_by_test_id("matching-total").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("served-filter-open_vocab_set").count() == 1
    first = query_of(sent.value)
    assert first["class_name"] == ["gadget"] and first["origin"] == ["human"]
    assert "class_id" not in first


def test_open_vocab_link_opens_matching_mode_with_removable_chips(stub, page, app_url):
    crop_requests: list = []

    def crops_handler(request, _match):
        crop_requests.append(request)
        return (200, {"total": 1, "page": 1, "page_size": 48, "crops": [item(0)]})

    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", crops_handler)
    _common(stub)

    page.goto(f"{app_url}/p/default/clusters?open_vocab_set=tags&source_prompt=blue+widget")
    page.get_by_test_id("matching-total").wait_for(timeout=ACTION_TIMEOUT_MS)
    q = query_of(crop_requests[-1])
    assert q["open_vocab_set"] == ["tags"] and q["source_prompt"] == ["blue widget"]

    chips = page.get_by_test_id("item-filter-chip")
    assert chips.count() == 2
    with page.expect_request(
        lambda r: "/crops" in r.url and "open_vocab_set" not in r.url and "source_prompt=blue" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        chips.filter(has_text="tags").click()
    assert page.get_by_test_id("item-filter-chip").count() == 1


def test_export_filter_sends_item_filter_and_shows_the_served_count(stub, page, app_url):
    stats_classes = {
        "classes": [
            {
                "class_id": 1,
                "class_name": "widget",
                "count": 10,
                "validated_count": 5,
                "adequacy": "ok",
                "aug_target": 100,
                "aug_gap": 90,
                "trainable": 5,
                "trainable_gap": 0,
            }
        ],
        "thresholds": {"block_below": 0, "warn_below": 5, "min_test": 5},
    }
    dataset_requests: list = []

    def dataset_handler(request, _match):
        dataset_requests.append(request)
        total = 7 if "class_name=widget" in request.url else 200
        return (200, {"total_crops": total, "validated": 5, "test_holdout": 0, "by_source": []})

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", stats_classes)
    stub.on("GET", r"/stats/dataset(\?|$)", dataset_handler)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": [], "min_test_per_class": 5})
    stub.on("GET", r"/export/status(\?|$)", {"status": "idle", "last_run": None})
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})
    posts: list[dict] = []

    def export_handler(request, _match):
        posts.append(request.post_data_json or {})
        return (
            200,
            {
                "status": "success",
                "export_dir": "/exports/x",
                "version_tag": "",
                "split_counts": {"train": 1, "val": 1, "test": 1},
            },
        )

    stub.on("POST", r"/export/yolo$", export_handler)

    page.goto(f"{app_url}/p/default/export")
    page.get_by_test_id("export-item-filter").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("export-item-filter").locator("summary").first.click()

    # Empty filter: no count, and no item_filter key on the export.
    assert page.get_by_test_id("export-matching-count").count() == 0
    page.get_by_role("button", name="Export", exact=True).click()
    page.get_by_text("Export complete.", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)
    assert posts[0] == {}
    page.get_by_role("button", name="Close").click()

    with page.expect_request(
        lambda r: "/stats/dataset" in r.url and "class_name=widget" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ) as sent:
        page.get_by_test_id("served-filter-class_name").locator("select").select_option("widget")
    count = page.get_by_test_id("export-matching-count")
    count.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_function(
        "() => document.querySelector('[data-testid=export-matching-count]')?.textContent.includes('7')",
        timeout=ACTION_TIMEOUT_MS,
    )
    # stats/dataset is asked without the open-vocabulary pair or class_id.
    last = query_of(sent.value)
    assert last["class_name"] == ["widget"] and "class_id" not in last

    page.get_by_role("button", name=re.compile(r"^(Re-)?[Ee]xport$")).click()
    page.get_by_text("Export complete.", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)
    assert posts[1]["item_filter"] == {"class_names": ["widget"]}
