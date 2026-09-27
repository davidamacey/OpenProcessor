"""/train's dataset card shows the current export's own contents (served
GET {API_PREFIX}/export/status counts) both for the `current` symlink and
for an explicit pick of that same directory in the version dropdown, and
falls back to the labelled global pool for a past version.

Live regression (2026-09-24 train smoke): picking the current export
explicitly showed "84 classes (global pool)" instead of its splits, because
the card only rendered the export's counts when nothing was picked.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from playwright.sync_api import expect
from test_train_gpus import register_train_mount

CURRENT = "/exports/20260924T233203Z"
PAST = "/exports/20260924T211523Z"

EXPORT_STATUS = {
    "status": "success",
    "last_run": "2026-09-24T23:32:10Z",
    "export_dir": CURRENT,
    "path": CURRENT,
    "image_count": 174,
    "object_count": 174,
    "class_count": 84,
    "classes_with_objects": 5,
    "split_counts": {"train": 134, "val": 15, "test": 25},
    "split_object_counts": {"train": 134, "val": 15, "test": 25},
    "class_split_counts": [
        {"class_id": 1, "export_id": 0, "class_name": "ducati", "train": 27, "val": 3, "test": 5},
    ],
}

DATASETS = {
    "datasets": [
        {"kind": "yolo", "export_dir": CURRENT, "version_tag": "v2", "image_count": 174, "is_current": True},
        {"kind": "yolo", "export_dir": PAST, "version_tag": "v1", "image_count": 174, "is_current": False},
    ]
}


def test_card_shows_current_export_contents_for_explicit_pick(stub, page, app_url):
    register_train_mount(stub)
    stub.on("GET", r"/export/status(\?|$)", EXPORT_STATUS)
    stub.on("GET", r"/export/datasets(\?|$)", DATASETS)

    page.goto(f"{app_url}/p/default/train")
    select = page.locator("label:has-text('dataset version') select")
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_text("images: train", exact=False).first.wait_for(timeout=10000)

    select.select_option(CURRENT)
    # #36 item 6: the served classes_with_objects (5 of 84), not just the
    # registry size, renders on the dataset card — wait for it as the real
    # proof the card re-rendered for this selection.
    expect(
        page.get_by_text("5 of 84 classes with objects", exact=False).first
    ).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("images: train", exact=False).count() > 0, (
        "an explicit pick of the current export must still show its own split counts"
    )
    assert page.get_by_text("global pool", exact=False).count() == 0

    select.select_option(PAST)
    # Wait for the real "global pool" fallback text to appear before
    # checking the negative below.
    expect(page.get_by_text("global pool", exact=False).first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert page.get_by_text("images: train", exact=False).count() == 0, (
        "a past version must not show the current export's counts"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
