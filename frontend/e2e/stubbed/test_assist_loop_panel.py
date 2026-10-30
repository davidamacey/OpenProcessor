"""The auto-label loop panel on /clusters: it is absent without the backend
routes, shows the served dry run before a start, and follows a live loop."""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_cluster_discard import CLASSES, CLUSTERS, crop

REQUEST = {
    "scope": {"cluster_ids": None, "min_root_size": 40},
    "vlm": None,
    "prompt_pack": None,
    "acknowledge_external": False,
    "seed": 0,
    "gate": {"theta": 0.9, "alpha": 0.05, "max_dissent": 1},
    "recursion": {"max_depth": 3, "branching": 4, "min_node_size": 40},
    "budget": {"max_vlm_crops": 2000, "max_vlm_requests": None, "max_wall_s": None},
    "audit": {
        "vlm": None,
        "acknowledge_external": False,
        "rate": 0.01,
        "min_per_batch": 30,
        "max_crops": 500,
        "floor": 0.85,
        "on_fail": "pause",
    },
    "vlm_concurrency": 16,
    "node_concurrency": 8,
    "dry_run": False,
}

PLAN = {
    "roots": 12,
    "eligible_items": 3400,
    "required_sample_table": [{"n_members": 300, "required_by_dissent": [32, 49]}],
    "estimated_vlm_crops_root_level": 380,
    "estimated_vlm_requests_root_level": 95,
    "fits_budget": True,
}

POOL = {
    "max_crops": 2000,
    "max_requests": 600,
    "crops_spent": 500,
    "crops_reserved": 8,
    "requests_spent": 130,
    "requests_reserved": 2,
}


def loop_state(status: str, *, dry_run: bool = False) -> dict:
    started = not dry_run
    return {
        "loop_id": "loop_abc123" if started else None,
        "status": status,
        "stop_reason": None,
        "pause_reason": None,
        "created_at": "2026-10-29T10:00:00+00:00",
        "started_at": "2026-10-29T10:00:05+00:00" if started else None,
        "finished_at": None,
        "updated_at": None,
        "error": None,
        "request": {**REQUEST, "dry_run": dry_run},
        "pinned": {
            "vlm": {"name": "local_vlm", "revision": 3, "model": "qwen3-vl-4b"},
            "prompt_pack": {"name": "default", "revision": 1},
            "audit_vlm": None,
        },
        "plan": PLAN,
        "partition": None,
        "budget": {"vlm": POOL, "audit": None, "wall_s": 75} if started else None,
        "levels": [
            {
                "level": 0,
                "nodes": 12,
                "passed": 7,
                "split": 5,
                "review_leaf": 0,
                "review_small": 0,
                "stale": 0,
                "unvisited": 0,
                "items_propagated": 2100,
                "items_direct": 224,
                "crops_spent": 400,
            }
        ]
        if started
        else [],
        "totals": {
            "items_propagated": 2100,
            "items_direct": 224,
            "clusters_left_for_review": 1,
            "items_left_for_review": 90,
            "skipped": {"dissent": 0, "drifted": 3, "locked": 12},
        }
        if started
        else None,
        "audit": {
            "status": "off",
            "sampled": 0,
            "agreed": 0,
            "agreement": None,
            "lower_bound": None,
            "crops_spent": 0,
        }
        if started
        else None,
    }


NODE = {
    "path": "r10001",
    "level": 0,
    "parent": None,
    "n": 300,
    "sampled": 33,
    "answered": 33,
    "decision": "pass",
    "reason": None,
    "class_name": "widget_a",
    "agree": 33,
    "bound": 0.913,
    "distribution": [{"class_name": "widget_a", "votes": 33}],
    "new_class_proposals": [],
    "items_propagated": 267,
}


def serve_clusters_page(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})


def serve_loop(stub, status: str) -> None:
    serve_clusters_page(stub)
    stub.on("GET", r"/assist/loops(\?|$)", {"loops": [loop_state(status)]})
    stub.on("GET", r"/assist/loops/loop_abc123$", loop_state(status))
    stub.on("GET", r"/assist/loops/loop_abc123/nodes(\?|$)", {"nodes": [NODE], "next_after": None})
    stub.on("GET", r"/assist/loops/loop_abc123/nodes/r10001/members(\?|$)", {"crop_ids": ["c1", "c2"], "next_after": None})
    stub.on("GET", r"/crops/[^/]+/thumbnail", (200, b"", "image/png"))


def serve_plan_form(stub) -> None:
    serve_clusters_page(stub)
    stub.on("POST", r"/assist/loops$", loop_state("dry_run", dry_run=True))


def open_panel(page) -> None:
    toggle = page.get_by_test_id("loop-panel-toggle")
    expect(toggle).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()


def show_finished_loop(page) -> None:
    open_panel(page)
    expect(page.get_by_test_id("loop-run")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("loop-node-toggle").first.click()
    expect(page.get_by_test_id("loop-node-detail")).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def show_plan_form(page) -> None:
    open_panel(page)
    expect(page.get_by_test_id("loop-plan-roots")).to_have_text("12", timeout=ACTION_TIMEOUT_MS)


def test_panel_is_collapsed_until_opened_and_shows_the_served_plan(stub, page, app_url):
    serve_plan_form(stub)
    page.goto(f"{app_url}/p/default/clusters")
    toggle = page.get_by_test_id("loop-panel-toggle")
    expect(toggle).to_have_attribute("aria-expanded", "false", timeout=ACTION_TIMEOUT_MS)
    show_plan_form(page)
    expect(page.get_by_test_id("loop-plan-items")).to_contain_text("3,400")
    expect(page.get_by_test_id("loop-start")).to_be_visible()


def test_finished_loop_shows_totals_and_node_members(stub, page, app_url):
    serve_loop(stub, "completed")
    page.goto(f"{app_url}/p/default/clusters")
    show_finished_loop(page)
    expect(page.get_by_test_id("loop-propagated")).to_contain_text("2,100")
    expect(page.get_by_test_id("loop-members-count")).to_contain_text("2 member ids loaded")
