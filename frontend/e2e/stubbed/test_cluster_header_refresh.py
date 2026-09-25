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
                        "id": CLUSTER_ID,
                        "name": "mustang",
                        "group": "car",
                        "hotkey_letter": "q",
                        "count": 161,
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

    page.goto(f"{app_url}/clusters/{CLUSTER_ID}")
    page.wait_for_selector("img", timeout=15000)
    page.wait_for_timeout(400)

    header = page.get_by_title("validated · labeled · cluster total")
    header.wait_for(timeout=15000)
    assert "0" in header.inner_text(), f"expected the pre-label count first: {header.inner_text()!r}"

    page.locator("img").first.click()
    page.wait_for_timeout(150)
    # 'q' is this class's own hotkey_letter (not one of the reserved
    # single-char action keys — g n d z x u a m /), routed by the
    # layout's class-letter keydown listener.
    page.keyboard.press("q")
    page.wait_for_timeout(600)

    assert len(classes_calls) >= 2, (
        "labeling a crop must re-fetch GET /classes so the header picks up the server's new count"
    )
    assert "34" in header.inner_text(), (
        f"header should show the refreshed server count without a reload: {header.inner_text()!r}"
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
