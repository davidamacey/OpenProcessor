"""OpenProcessor 6c77deb adoption ("export splits by source image,
split-coverage preflight checks, honest holdout freeze, validated
augmentation presets"):

1. `GET {API_PREFIX}/export/status` now serves `image_count`/
   `class_count`/`split_counts`/`class_split_counts` on a successful
   export — the page must render them, with a 0-train/0-val class row
   highlighted using the served numbers only.
2. `POST {API_PREFIX}/test_holdout/freeze`'s request body is `{percent}`
   only — no `seed`. The freeze modal must not offer a Seed field, and
   the actual request fired must carry no `seed` key.
"""

from __future__ import annotations

CLASSES = [
    {
        "id": 1,
        "name": "bmw",
        "group": "car",
        "hotkey_letter": "b",
        "count": 300,
        "validated_count": 200,
        "cluster_size": 300,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "audi",
        "group": "car",
        "hotkey_letter": "a",
        "count": 100,
        "validated_count": 50,
        "cluster_size": 100,
        "deprecated": False,
    },
]

STATS_CLASSES = {
    "classes": [
        {"class_id": 1, "class_name": "bmw", "count": 300, "validated_count": 200, "adequacy": "ok"},
        {"class_id": 2, "class_name": "audi", "count": 100, "validated_count": 50, "adequacy": "warn"},
    ],
    "thresholds": {"block_below": 0, "warn_below": 5, "min_test": 5},
}

STATS_DATASET = {"total_crops": 400, "validated": 250, "test_holdout": 0, "by_source": []}

HOLDOUT_STATS_EMPTY = {"total": 0, "by_class": [], "min_test_per_class": 5}

EXPORT_STATUS_SUCCESS = {
    "status": "success",
    "last_run": "2026-09-24T21:00:00Z",
    "path": "/exports/current",
    "export_dir": "/exports/current",
    "version_tag": "v1",
    "dataset_sha": "abc123",
    "seed": None,
    "group_key": "image_id",
    "image_count": 500,
    "class_count": 2,
    "split_counts": {"train": 400, "val": 80, "test": 20},
    "class_split_counts": [
        {"class_id": 1, "export_id": 0, "class_name": "bmw", "train": 200, "val": 40, "test": 10},
        {"class_id": 2, "export_id": 1, "class_name": "audi", "train": 0, "val": 40, "test": 10},
    ],
}

FREEZE_RESPONSE = {
    "n_frozen": 25,
    "n_classes_covered": 2,
    "test_holdout_sha": "deadbeef",
    "per_class_counts": {"1": 15, "2": 10},
    "selection": "sha1_per_class",
    "percent": 10,
    "min_per_class": 5,
}


def register_export_mount(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", STATS_CLASSES)
    stub.on("GET", r"/stats/dataset(\?|$)", STATS_DATASET)
    stub.on("GET", r"/test_holdout/stats(\?|$)", HOLDOUT_STATS_EMPTY)
    stub.on("GET", r"/export/status(\?|$)", EXPORT_STATUS_SUCCESS)
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})


def test_export_status_renders_served_split_counts_and_highlights_zero_classes(
    stub, page, app_url
):
    register_export_mount(stub)

    page.goto(f"{app_url}/export")
    page.get_by_text("500", exact=False).first.wait_for(timeout=15000)

    assert page.get_by_text("train 400", exact=False).count() > 0
    assert page.get_by_text("val 80", exact=False).count() > 0
    assert page.get_by_text("test 20", exact=False).count() > 0

    summary = page.get_by_text("Per-class split counts", exact=False)
    summary.wait_for(timeout=15000)
    summary.click()

    details = page.locator("details", has_text="Per-class split counts")
    audi_row = details.locator("tr", has_text="audi")
    audi_row.wait_for(timeout=15000)
    assert "bg-red-500/10" in (audi_row.get_attribute("class") or ""), (
        "audi has 0 train instances and must be highlighted"
    )

    bmw_row = details.locator("tr", has_text="bmw")
    assert "bg-red-500/10" not in (bmw_row.get_attribute("class") or "")

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_freeze_modal_has_no_seed_field_and_posts_percent_only(stub, page, app_url):
    register_export_mount(stub)

    freeze_calls: list[dict] = []

    def freeze_handler(request, match):
        freeze_calls.append(request.post_data_json or {})
        return (200, FREEZE_RESPONSE)

    stub.on("POST", r"/test_holdout/freeze(\?|$)", freeze_handler)

    page.goto(f"{app_url}/export")
    freeze_button = page.get_by_role("button", name="Freeze test set", exact=True)
    freeze_button.wait_for(timeout=15000)
    freeze_button.click()

    # No Seed field anywhere in the modal.
    assert page.get_by_text("Seed", exact=True).count() == 0, (
        "the freeze modal must not offer a Seed field — selection is deterministic (sha1 per class)"
    )

    page.once("dialog", lambda d: d.accept())
    page.get_by_role("button", name="Freeze", exact=True).click()

    page.wait_for_function(
        "() => document.body.innerText.includes('sha1_per_class')", timeout=15000
    )

    assert len(freeze_calls) == 1, f"expected exactly one freeze POST: {freeze_calls}"
    assert freeze_calls[0] == {"percent": 10}, (
        f"freeze body must be exactly {{'percent': 10}}, got {freeze_calls[0]}"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
