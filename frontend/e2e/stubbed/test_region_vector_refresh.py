"""OpenProcessor v0.4.0: every region write serves `vector_refresh
{embedded, pending}`. After a per-box accept on /review's region tab, a
served `pending` > 0 reads "N boxes have no vector yet"; a zero or absent
one shows nothing; the next item clears it.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint

from fixtures.wire import REGION_CLASS, REGION_TAB_URL_ID, make_box, make_item

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


def tag_item(crop_id: str, state: str = "proposed") -> dict:
    return make_item(
        crop_id=crop_id,
        image_id=f"img-{crop_id}",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url=f"/curation/crops/{crop_id}/thumbnail",
        region_status="pending_verification",
        region_boxes=[make_box("b1", state=state)],
    )


def open_region_review(stub, page, app_url, vector_refresh) -> list[dict]:
    patches: list[dict] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        lambda _r, _m: (
            200,
            {"items": [tag_item("tag-1"), tag_item("tag-2")], "total": 2, "page": 1, "page_size": 30},
        ),
    )

    def patch_box(request, match):
        patches.append(request.post_data_json or {})
        body = {"crop_id": "tag-1", "box_id": "b1", "item": tag_item("tag-1", "accepted")}
        if vector_refresh is not None:
            body["vector_refresh"] = vector_refresh
        return (200, body)

    stub.on("PATCH", r"/crops/([^/]+)/regions/([^/]+)$", patch_box)
    page.goto(f"{app_url}/p/default/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    return patches


def accept_box(page) -> None:
    with page.expect_response(
        lambda r: r.request.method == "PATCH" and "/regions/b1" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press("y")


def test_pending_vectors_are_reported_after_a_box_write_and_cleared_on_the_next_item(
    stub, page, app_url
):
    patches = open_region_review(stub, page, app_url, {"embedded": 1, "pending": 2})
    notice = page.get_by_test_id("vector-refresh-notice")
    assert notice.count() == 0
    accept_box(page)
    notice.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "2 boxes have no vector yet" in " ".join(notice.inner_text().split())
    assert patches and patches[0].get("state") == "accepted"
    # Moving to the next item drops the previous item's notice.
    page.keyboard.press("n")
    page.wait_for_selector('[data-testid="vector-refresh-notice"]', state="detached", timeout=ACTION_TIMEOUT_MS)


def test_nothing_is_shown_when_no_vector_is_pending(stub, page, app_url):
    open_region_review(stub, page, app_url, {"embedded": 3, "pending": 0})
    accept_box(page)
    wait_for_paint(page)
    assert page.get_by_test_id("vector-refresh-notice").count() == 0


def test_nothing_is_shown_when_the_write_serves_no_vector_refresh(stub, page, app_url):
    open_region_review(stub, page, app_url, None)
    accept_box(page)
    wait_for_paint(page)
    assert page.get_by_test_id("vector-refresh-notice").count() == 0
