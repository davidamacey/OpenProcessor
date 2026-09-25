"""OpenProcessor 6c77deb + d5343cb adoption ("export splits by source
image, split-coverage preflight checks, honest holdout freeze, validated
augmentation presets" + "export one image + one label file per source
image, partial-frame policy and counts"):

1. `GET {API_PREFIX}/export/status` now serves `image_count`/
   `class_count`/`split_counts`/`class_split_counts` (6c77deb) and
   `object_count`/`split_object_counts`/`require_fully_labeled_images`/
   partial-frame counts (d5343cb) on a successful export — the page must
   render them, with a 0-train/0-val class row highlighted using the
   served numbers only.
2. `POST {API_PREFIX}/test_holdout/freeze`'s request body is `{percent}`
   only — no `seed`. The freeze modal must not offer a Seed field, and
   the actual request fired must carry no `seed` key.
3. `POST {API_PREFIX}/export/yolo` accepts an opt-in
   `require_fully_labeled_images` — checking "Only images whose every
   object is labeled" must actually send it.
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
    "object_count": 620,
    "class_count": 2,
    "split_counts": {"train": 400, "val": 80, "test": 20},
    "split_object_counts": {"train": 500, "val": 100, "test": 20},
    "require_fully_labeled_images": False,
    "unlabeled_items_on_exported_images": 12,
    "images_with_unlabeled_items": 9,
    "images_dropped_not_fully_labeled": 0,
    "class_split_counts": [
        {"class_id": 1, "export_id": 0, "class_name": "bmw", "train": 200, "val": 40, "test": 10},
        {"class_id": 2, "export_id": 1, "class_name": "audi", "train": 0, "val": 40, "test": 10},
    ],
}

EXPORT_YOLO_RESULT = {
    "status": "success",
    "export_dir": "/exports/new",
    "image_count": 500,
    "object_count": 620,
    "split_counts": {"train": 400, "val": 80, "test": 20},
    "split_object_counts": {"train": 500, "val": 100, "test": 20},
    "require_fully_labeled_images": True,
    "unlabeled_items_on_exported_images": 0,
    "images_with_unlabeled_items": 0,
    "images_dropped_not_fully_labeled": 30,
    "started_at": "2026-09-24T21:00:00Z",
    "finished_at": "2026-09-24T21:00:05Z",
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
    page.get_by_text("620", exact=False).first.wait_for(timeout=15000)

    assert page.get_by_text("objects in", exact=False).count() > 0
    assert page.get_by_text("images: train 400", exact=False).count() > 0
    assert page.get_by_text("objects: train 500", exact=False).count() > 0

    summary = page.get_by_text("Per-class object counts", exact=False)
    summary.wait_for(timeout=15000)
    summary.click()

    details = page.locator("details", has_text="Per-class object counts")
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


def test_require_fully_labeled_images_checkbox_sends_the_flag(stub, page, app_url):
    register_export_mount(stub)

    export_calls: list[dict] = []

    def export_handler(request, match):
        export_calls.append(request.post_data_json or {})
        return (200, EXPORT_YOLO_RESULT)

    stub.on("POST", r"/export/yolo(\?|$)", export_handler)

    page.goto(f"{app_url}/export")
    checkbox = page.get_by_text("Only images whose every object is labeled", exact=False)
    checkbox.wait_for(timeout=15000)
    checkbox.click()

    export_button = page.get_by_role("button", name="Re-export", exact=True)
    if export_button.count() == 0:
        export_button = page.get_by_role("button", name="Export", exact=True)
    export_button.click()

    page.get_by_text("Export complete.", exact=False).wait_for(timeout=15000)

    assert len(export_calls) == 1, f"expected exactly one export POST: {export_calls}"
    assert export_calls[0].get("require_fully_labeled_images") is True, (
        f"checking the box must send require_fully_labeled_images: true, got {export_calls[0]}"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_export_shows_trainable_vs_held_out_and_classes_with_objects(stub, page, app_url):
    """Visual audit 2026-09-24 E1/E2: Validated included frozen test crops
    (so GAP was 5 too small) and "classes 84" counted the registry, not
    the classes that actually have exported objects."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on(
        "GET",
        r"/stats/classes(\?|$)",
        {
            "classes": [
                {"class_id": 1, "class_name": "bmw", "count": 108, "validated_count": 35,
                 "adequacy": "warn", "aug_target": 500, "aug_gap": 465},
                {"class_id": 2, "class_name": "audi", "count": 10, "validated_count": 0,
                 "adequacy": "block", "aug_target": 500, "aug_gap": 500},
            ],
            "thresholds": {"block_below": 20, "warn_below": 500, "min_test": 5},
        },
    )
    stub.on("GET", r"/stats/dataset(\?|$)", STATS_DATASET)
    stub.on(
        "GET",
        r"/test_holdout/stats(\?|$)",
        {"total": 5, "by_class": [{"key": 1, "doc_count": 5, "deficient": False}],
         "min_test_per_class": 5},
    )
    status = dict(EXPORT_STATUS_SUCCESS)
    status["class_split_counts"] = [
        {"class_id": 1, "export_id": 0, "class_name": "bmw", "train": 24, "val": 6, "test": 5},
        {"class_id": 2, "export_id": 1, "class_name": "audi", "train": 0, "val": 0, "test": 0},
    ]
    stub.on("GET", r"/export/status(\?|$)", status)
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})

    page.goto(f"{app_url}/export")
    bmw = page.locator("table tr", has_text="bmw").first
    bmw.wait_for(timeout=15000)
    cells = [c.strip() for c in bmw.locator("td").all_inner_texts()]
    # Class, ID, Total, Validated, Trainable, Aug target, Gap, Test, Adequacy
    assert cells[3:8] == ["35", "30", "500", "+470", "5"], cells

    chip = " ".join(page.locator('[data-testid="export-class-count"]').inner_text().split())
    assert chip == "1 classes with objects (2 in registry)", chip

    page.get_by_text("Per-class object counts", exact=False).click()
    details = page.locator("details", has_text="Per-class object counts")
    assert details.locator("tr", has_text="audi").count() == 0
    assert "1 class with no objects" in details.inner_text()
    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]
