"""Regression coverage for the live /clusters/[id] header-refresh bug
(artifacts_local/cw-live/train-smoke/): the header's "N validated · N
labeled · N in cluster" chip reads `classesStore` (per-class
`validated_count`/`count`, `GET {API_PREFIX}/classes`), which the server
owns. After a successful class-hotkey label write it stayed on the
pre-write count and only updated on a hard reload — the write handler
never re-fetched `classesStore`. `label_44_after.png` vs
`cluster_44_reload.png` in the smoke evidence shows the same "0 validated"
-> "34 validated" jump this test drives directly: label one crop through
the page, then assert the header's own re-fetch of `GET {API_PREFIX}/classes`
picks up the server's new count with no navigation/reload.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from playwright.sync_api import expect

import json

from fixtures.wire import make_item

CLUSTER_ID = 44

CLUSTERS = {
    "items": [
        {
            "cluster_id": CLUSTER_ID,
            "cluster_kind": "class",
            "size": 2,
            "validated_count": 0,
            "dominant_class_id": CLUSTER_ID,
            "dominant_class_name": "mustang",
            "purity": 1.0,
            "is_unlabeled": False,
            "representatives": [{"crop_id": "crop-0"}],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}


def crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=CLUSTER_ID,
        class_name="mustang",
        cluster_id=CLUSTER_ID,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def _json(request):
    if not request.post_data:
        return None
    try:
        return json.loads(request.post_data)
    except (ValueError, TypeError):
        return None


def test_cluster_header_refreshes_validated_count_after_labeling(stub, page, app_url):
    classes_calls: list[int] = []

    def classes_handler(request, match):
        classes_calls.append(1)
        # Before the label write: 0 validated (matches the live repro).
        # After the write, the server's own count is 34 — the header
        # must show this once classesStore re-fetches, no reload needed.
        validated = 0 if len(classes_calls) == 1 else 34
        return (
            200,
            {
                "classes": [
                    {
                        "class_id": CLUSTER_ID,
                        "class_name": "mustang",
                        "kind": "item",
                        "group": "car",
                        "hotkey_letter": "q",
                        "sample_count": 161,
                        "validated_count": validated,
                        "cluster_size": 161,
                        "deprecated": False,
                    }
                ],
                "thresholds": {
                    "block_below": 0,
                    "warn_below": 5,
                    "min_test_per_class": 5,
                    "aug_target_min": 500,
                    "aug_target_max": 500,
                },
                "reserved_hotkeys": ["g", "n", "d", "z", "x", "u", "a", "m", "/"],
            },
        )

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", classes_handler)
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on(
        "GET",
        r"/crops(\?|$)",
        {"total": 2, "page": 1, "page_size": 60, "crops": [crop(0), crop(1)]},
    )

    def batch_label_handler(request, match):
        ids = (_json(request) or {}).get("crop_ids", [])
        return (200, {"updated": len(ids), "updated_ids": ids, "conflicts": []})

    stub.on("PUT", r"/crops/batch_label$", batch_label_handler)

    page.goto(f"{app_url}/p/default/clusters/{CLUSTER_ID}")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)

    header = page.get_by_title("validated · labeled · cluster total")
    header.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "0" in header.inner_text(), f"expected the pre-label count first: {header.inner_text()!r}"

    page.locator("img").first.click()
    expect(page.get_by_text("1 selected").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    # 'q' is this class's own hotkey_letter (not one of the reserved
    # single-char action keys — g n d z x u a m /), routed by the
    # layout's class-letter keydown listener.
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith("/batch_label"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press("q")

    # The header text only updates once the post-label GET /classes
    # refetch lands, so waiting on it first (a retrying expect) also
    # guarantees classes_calls has already been incremented below.
    expect(header).to_contain_text("34", timeout=ACTION_TIMEOUT_MS)
    assert len(classes_calls) >= 2, (
        "labeling a crop must re-fetch GET /classes so the header picks up the server's new count"
    )
    # K1 (visual audit 2026-09-24): the class registry's counts (161
    # labeled) and the cluster's own size (2) are different scopes; each
    # must say which one it is, so "labeled" can't read as a share of
    # "in cluster".
    text = " ".join(header.inner_text().split())
    assert "class-wide: 34 validated · 161 labeled" in text, text
    assert "2 in this cluster" in text, text

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the label/header-refresh flow: {errors[:3]}"


def test_cluster_header_says_when_the_card_lookup_failed(stub, page, app_url):
    """The cluster card read failing (the crops still load) used to render
    "Cluster #N" with no sign that anything was wrong; the served reason is
    shown instead of a silent unlabeled identity."""
    stub.on(
        "GET",
        r"(?<!/stats)/classes(\?|$)",
        {
            "classes": [],
            "thresholds": {
                "block_below": 0,
                "warn_below": 5,
                "min_test_per_class": 5,
                "aug_target_min": 500,
                "aug_target_max": 500,
            },
            "reserved_hotkeys": [],
        },
    )
    stub.on("GET", r"/clusters(\?|$)", (400, {"detail": "aggregation unavailable"}))
    stub.on(
        "GET",
        r"/crops(\?|$)",
        {"total": 1, "page": 1, "page_size": 60, "crops": [crop(0)]},
    )
    page.goto(f"{app_url}/p/default/clusters/{CLUSTER_ID}")
    expect(page.get_by_test_id("cluster-card-error")).to_contain_text(
        "aggregation unavailable", timeout=ACTION_TIMEOUT_MS
    )
