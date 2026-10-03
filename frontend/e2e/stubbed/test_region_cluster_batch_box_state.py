"""W8 region-cluster bulk triage (docs/design/
w8-multibox-frontend-plan-2026-09-26.md, §7.7): "Triage from a cluster
goes through batch_box_state, never the item-level batch_status, which
would flip every sibling box." Once a curator opens a region cluster
bucket, the same Confirm/No-<region>/Mark-false-positive bulk toolbar
must call POST {API_PREFIX}/regions/batch_box_state with per-box targets,
not POST {API_PREFIX}/regions/batch_status.

No fixed sleeps (parallel e2e, CLAUDE.md): every wait is a real selector
or `page.expect_request`.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import REGION_CLASS

CLASSES = [
    {
        "class_id": 1,
        "class_name": REGION_CLASS,
        "kind": "region",
        "group": "widgets",
        "hotkey_letter": "l",
        "sample_count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
]

STATUSES = {
    "statuses": [],
    "confirm_status": "detected",
    "reject_status": "no_region_visible",
    "false_positive_status": "false_positive",
    "box_states": [
        {
            "value": "accepted",
            "label": "accepted",
            "role": "accepted",
            "human_writable": True,
            "exported": True,
            "dashed": False,
            "dim": False,
            "badge": None,
        },
    ],
}

CLUSTER_CARD = {
    "id": 7,
    "size": 2,
    "box_count": 2,
    "n_subclusters": 0,
    "has_subclusters": False,
    "cluster_kind": None,
    "representative_thumb_urls": [],
}

ROW_1 = {
    "crop_id": "c1",
    "id": "c1",
    "image_path": "/img1.jpg",
    "bbox_norm": [0, 0, 1, 1],
    "region_bbox_norm": [0.1, 0.1, 0.2, 0.2],
    "region_score": 0.9,
    "region_status": "detected",
    "region_verified": True,
    "region_validated": False,
    "region_detector": "tag_detector_v1",
    "region_detector_version": "1",
    "region_detector_chain": [],
    "region_bbox_frame": "source",
    "region_detected_at": None,
    "region_verifier": None,
    "region_verifier_version": None,
    "region_verified_at": None,
    "region_rejection_reason": None,
    "region_visible": True,
    "region_text": None,
    "region_text_source": None,
    "region_text_confidence": None,
    "class_id": 1,
    "class_name": REGION_CLASS,
    "cluster_id": None,
    "updated_at": "2026-09-26T00:00:00Z",
    "region_box_id": "b1",
    "row_key": "c1#b1",
}
ROW_2 = {**ROW_1, "crop_id": "c2", "id": "c2", "region_box_id": "b2", "row_key": "c2#b2"}


def test_cluster_triage_uses_batch_box_state_with_per_box_targets(stub, page, app_url):
    box_state_calls: list[dict] = []
    batch_status_calls: list[dict] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/regions/statuses(\?|$)", STATUSES)
    stub.on(
        "GET",
        r"/regions/clusters(\?|$)",
        {"clusters": [CLUSTER_CARD], "count": 1},
    )
    stub.on("GET", r"(?<!/regions)/clusters(\?|$)", {"clusters": [], "count": 0})

    def regions_handler(request, match):
        return (200, {"items": [ROW_1, ROW_2], "total": 2, "total_rows": 2})

    stub.on("GET", r"/regions(\?|$)", regions_handler)

    def box_state_handler(request, match):
        body = request.post_data_json
        box_state_calls.append(body)
        return (
            200,
            {
                "updated": 2,
                "invalid": [],
                "conflicts": [],
                "items": [
                    {"id": "c1", "class_id": 1},
                    {"id": "c2", "class_id": 1},
                ],
            },
        )

    stub.on("POST", r"/regions/batch_box_state(\?|$)", box_state_handler)

    def batch_status_handler(request, match):
        batch_status_calls.append(request.post_data_json)
        return (200, {"updated": 2, "conflicts": [], "invalid": [], "items": []})

    stub.on("POST", r"/regions/batch_status(\?|$)", batch_status_handler)

    page.goto(f"{app_url}/p/default/clusters?class={REGION_CLASS}")

    cluster_card = page.locator('button:has-text("#7")')
    cluster_card.wait_for(timeout=ACTION_TIMEOUT_MS)
    cluster_card.click()

    select_all = page.get_by_role("button", name="Select all")
    select_all.wait_for(timeout=ACTION_TIMEOUT_MS)
    select_all.click()

    verify_button = page.get_by_role("button", name="Verify")
    verify_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    with page.expect_request(
        lambda r: r.method == "POST" and "/regions/batch_box_state" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        verify_button.click()

    assert not batch_status_calls, (
        "cluster triage must never call the item-level batch_status route "
        f"(would flip every sibling box): {batch_status_calls}"
    )
    assert len(box_state_calls) == 1
    body = box_state_calls[0]
    assert body["state"] == "accepted"
    targets = sorted(body["targets"], key=lambda t: t["crop_id"])
    assert targets == [
        {"crop_id": "c1", "box_id": "b1"},
        {"crop_id": "c2", "box_id": "b2"},
    ]

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the cluster-triage flow: {errors[:3]}"

    stub.assert_fail_closed()
