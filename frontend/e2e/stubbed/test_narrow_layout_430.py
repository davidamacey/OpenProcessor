"""430px layout bugs on /clusters/[id], /train and the /clusters error state
(#312). jsdom has no layout, so each test asserts measured geometry in a real
browser.
"""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import wait_for_paint

from fixtures.wire import make_item
from test_train_gpus import register_train_mount
from test_train_results import FINISHED_STATUS

NARROW = {"width": 430, "height": 900}
CLUSTER_ID = 64

CLASSES = [
    {"class_id": CLUSTER_ID, "class_name": "class_b", "kind": "item", "group": "moto", "hotkey_letter": None,
     "sample_count": 641, "validated_count": 0, "cluster_size": 616, "deprecated": False},
]


def _crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=CLUSTER_ID,
        class_name="class_b",
        class_source="vlm",
        label_source="vlm",
        label_validated=False,
        class_validated=False,
        cluster_id=CLUSTER_ID,
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def test_ignore_reason_menu_stays_inside_the_viewport_at_430(stub, page, app_url):
    page.set_viewport_size(NARROW)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/crops(\?|$)",
            {"total": 3, "page": 1, "page_size": 60, "crops": [_crop(i) for i in range(3)]})
    stub.on("GET", r"/clusters(\?|$)", {"items": [], "total": 0, "total_class_clusters": 0,
                                       "total_candidate_clusters": 0, "cluster_id_offset": 10000})

    page.goto(f"{app_url}/p/default/clusters/{CLUSTER_ID}")
    page.locator('[data-testid="class-name"]').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_role("button", name="Choose ignore reason").click()
    wait_for_paint(page)
    box = page.evaluate(
        """() => {
          const b = document.querySelector('[aria-label="Choose ignore reason"]');
          const m = b.parentElement.querySelector('div.absolute');
          const r = m.getBoundingClientRect();
          return {left: r.left, right: r.right, vw: window.innerWidth};
        }"""
    )
    assert box["left"] >= 0 and box["right"] <= box["vw"], f"ignore menu is clipped: {box}"


def test_past_runs_name_column_does_not_collapse_at_430(stub, page, app_url):
    page.set_viewport_size(NARROW)
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [FINISHED_STATUS], "total": 1})

    page.goto(f"{app_url}/p/default/train")
    page.locator("th", has_text="Name").first.wait_for(state="attached", timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    name_w = page.locator("th", has_text="Name").first.evaluate("el => el.getBoundingClientRect().width")
    assert name_w >= 120, f"Name column collapsed to {name_w}px"
    m = page.evaluate("() => ({sw: document.documentElement.scrollWidth, vw: window.innerWidth})")
    assert m["sw"] <= m["vw"], f"page overflows: {m}"


def test_clusters_error_state_shows_no_counts(stub, page, app_url):
    page.set_viewport_size(NARROW)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", (500, {"detail": "boom"}))

    page.goto(f"{app_url}/p/default/clusters")
    page.get_by_test_id("clusters-error").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.locator('[data-testid="clusters-footer-count"]').count() == 0
