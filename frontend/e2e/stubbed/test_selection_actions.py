"""OpenProcessor v0.4.0 selection writes from /clusters "Matching items":
every action runs the server's dry run first (`dry_run: true`, the dialog
states its `selected`), the real call repeats the same selection with
`dry_run: false` only after Apply, and the served `updated_ids` go to the
undo stack so Z reverts the write (a label via `label/undo_batch`, an
ignore via `batch_unexclude`).
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, expect_handled

from fixtures.wire import make_item
from test_item_filter import CLASSES, CLUSTERS


FILTER = {"class_names": ["widget"]}


def item(i: int) -> dict:
    return make_item(crop_id=f"crop-{i}", image_id=f"img-{i}", class_id=1, class_name="widget")


def _setup(stub, page, app_url) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on(
        "GET",
        r"/crops(\?|$)",
        {"total": 2, "page": 1, "page_size": 48, "crops": [item(0), item(1)]},
    )
    page.goto(f"{app_url}/p/default/clusters?class_name=widget&mode=matching")
    page.get_by_test_id("matching-total").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_ignore_all_matching_dry_runs_first_then_applies_and_z_restores(stub, page, app_url):
    bodies: list[dict] = []

    def exclude_handler(request, _match):
        body = request.post_data_json or {}
        bodies.append(body)
        if body.get("dry_run"):
            return (200, {"dry_run": True, "selected": 2})
        return (200, {"excluded": 2, "updated_ids": ["crop-0", "crop-1"], "errors": 0})

    unexclude_bodies: list[dict] = []

    def unexclude_handler(request, _match):
        unexclude_bodies.append(request.post_data_json or {})
        return (200, {"unexcluded": 2, "updated_ids": ["crop-0", "crop-1"], "errors": 0})

    stub.on("POST", r"/crops/batch_exclude$", exclude_handler)
    stub.on("POST", r"/crops/batch_unexclude$", unexclude_handler)
    _setup(stub, page, app_url)

    with expect_handled(page,
        lambda r: r.method == "POST" and r.url.endswith("/crops/batch_exclude"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id("matching-action-exclude").click()
    page.get_by_test_id("selection-count").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_function(
        "() => document.querySelector('[data-testid=selection-count]')?.textContent.includes('2')",
        timeout=ACTION_TIMEOUT_MS,
    )
    # Only the dry run so far: the real write has not been sent.
    assert bodies == [{"selection": {"filter": FILTER}, "reason": "ignore", "dry_run": True}]

    # Changing the limit re-runs the dry run with it.
    with expect_handled(page,
        lambda r: r.method == "POST" and "limit" in (r.post_data or ""),
        timeout=ACTION_TIMEOUT_MS,
    ):
        limit = page.get_by_test_id("selection-limit")
        limit.fill("5")
        limit.blur()
    assert bodies[-1]["selection"] == {"filter": FILTER, "limit": 5}
    assert bodies[-1]["dry_run"] is True

    with expect_handled(page,
        lambda r: r.method == "POST"
        and r.url.endswith("/crops/batch_exclude")
        and '"dry_run":false' in (r.post_data or "").replace(" ", ""),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_role("button", name="Apply").click()
    assert bodies[-1] == {
        "selection": {"filter": FILTER, "limit": 5},
        "reason": "ignore",
        "dry_run": False,
    }

    # The served ids reached the undo stack: Z un-ignores exactly them.
    page.get_by_role("button", name="Apply").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    with expect_handled(page,
        lambda r: r.method == "POST" and r.url.endswith("/crops/batch_unexclude"),
        timeout=ACTION_TIMEOUT_MS,
    ) as undo:
        page.keyboard.press("z")
    assert undo.value.post_data_json == {"crop_ids": ["crop-0", "crop-1"]}


def test_label_all_matching_needs_a_class_and_z_undoes_through_label_undo(stub, page, app_url):
    bodies: list[dict] = []

    def label_handler(request, _match):
        body = request.post_data_json or {}
        bodies.append(body)
        if body.get("dry_run"):
            return (200, {"dry_run": True, "selected": 2})
        return (200, {"updated": 2, "updated_ids": ["crop-0", "crop-1"], "conflicts": []})

    undo_bodies: list[dict] = []

    def undo_handler(request, _match):
        undo_bodies.append(request.post_data_json or {})
        return (
            200,
            {"items": [], "undone": 2, "nothing_to_undo": [], "conflicts": [], "not_found": []},
        )

    stub.on("PUT", r"/crops/batch_label$", label_handler)
    stub.on("POST", r"/crops/label/undo_batch$", undo_handler)
    _setup(stub, page, app_url)

    page.get_by_test_id("matching-action-label").click()
    page.get_by_test_id("selection-count").wait_for(timeout=ACTION_TIMEOUT_MS)
    # No class yet: nothing was asked of the server and Apply is disabled.
    assert bodies == []
    assert page.get_by_role("button", name="Apply").is_disabled()

    with expect_handled(page,
        lambda r: r.method == "PUT" and r.url.endswith("/crops/batch_label"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_test_id("selection-class").select_option("2")
    assert bodies[-1] == {
        "selection": {"filter": FILTER},
        "class_id": 2,
        "validated": True,
        "dry_run": True,
    }
    page.wait_for_function(
        "() => !document.querySelector('[role=dialog] button.btn-primary')?.disabled",
        timeout=ACTION_TIMEOUT_MS,
    )
    with expect_handled(page,
        lambda r: r.method == "PUT"
        and '"dry_run":false' in (r.post_data or "").replace(" ", ""),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_role("button", name="Apply").click()
    page.get_by_role("button", name="Apply").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)

    with expect_handled(page,
        lambda r: r.method == "POST" and r.url.endswith("/crops/label/undo_batch"),
        timeout=ACTION_TIMEOUT_MS,
    ) as undo:
        page.keyboard.press("z")
    assert undo.value.post_data_json == {"crop_ids": ["crop-0", "crop-1"]}


def test_a_served_refusal_is_shown_verbatim_and_nothing_is_written(stub, page, app_url):
    bodies: list[dict] = []

    def exclude_handler(request, _match):
        bodies.append(request.post_data_json or {})
        return (
            422,
            {"detail": {"error": "selection_too_large", "message": "matches more than 20000 items"}},
        )

    stub.on("POST", r"/crops/batch_exclude$", exclude_handler)
    _setup(stub, page, app_url)

    page.get_by_test_id("matching-action-exclude").click()
    err = page.get_by_test_id("selection-error")
    err.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "matches more than 20000 items" in err.inner_text()
    assert page.get_by_role("button", name="Apply").is_disabled()
    assert all(b.get("dry_run") is True for b in bodies)
