"""Assist runs on /clusters/[id] (stubbed backend): the dialog posts exactly
the server's request shape, shows the dry run's counts, follows the run to
its proposal, applies it only on the explicit click, and renders the
server's `scope_changed` refusal. A sub-cluster tab asks the server for
`cluster_subid` instead of filtering the loaded page.

Screenshots (1600 and 430 wide) go to artifacts_local/assist-ui/ for a human
to look at; they are not assertions.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect

from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import wait_for_paint

SHOTS = Path(__file__).resolve().parents[2] / "artifacts_local" / "assist-ui"

CLASSES = [
    {"class_id": 1, "class_name": "widget_a", "kind": "item", "group": "widgets", "hotkey_letter": "a",
     "sample_count": 40, "validated_count": 12, "cluster_size": 44, "deprecated": False},
    {"class_id": 2, "class_name": "widget_b", "kind": "item", "group": "widgets", "hotkey_letter": "b",
     "sample_count": 20, "validated_count": 5, "cluster_size": 22, "deprecated": False},
]

CLUSTER_ID = 10000
CLUSTERS = {
    "items": [
        {
            "cluster_id": CLUSTER_ID,
            "cluster_kind": "candidate",
            "size": 412,
            "validated_count": 0,
            "dominant_class_id": None,
            "dominant_class_name": None,
            "purity": 0.8,
            "is_unlabeled": True,
            "representatives": [{"crop_id": "crop-0"}],
            "n_subclusters": 2,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 0,
    "total_candidate_clusters": 1,
    "cluster_id_offset": 10000,
}

METHODS = {
    "strategies": [
        {"id": "local_vlm", "axis": "vlm", "label": "Local VLM", "status": "stable",
         "endpoint_status_label": "Ready", "per_run_ack_required": False},
        {"id": "widgets", "axis": "prompt_pack", "label": "Widget pack", "status": "stable"},
    ]
}

RESOLVED = {
    "scope_count": 412,
    "scope_digest": "sha256:abc",
    "sampled": 41,
    "sample_seed": 0,
    "strata": [
        {"key": "10000a", "size": 120, "sampled": 12},
        {"key": "10000b", "size": 292, "sampled": 29},
    ],
    "skipped": {"holdout": 3, "excluded": 1, "class_locked": 27},
    "vlm": {"name": "env", "revision": 3, "model": "vlm-model"},
    "prompt_pack": {"name": "widgets", "revision": 7},
    "estimated_vlm_calls": 6,
}

PROPOSAL = {
    "class_name": "widget_a",
    "in_registry": True,
    "agreement": 0.83,
    "answered": 41,
    "min_agreement": 0.6,
    "passes_gate": True,
    "distribution": [{"class_name": "widget_a", "votes": 34}, {"class_name": "widget_b", "votes": 7}],
    "new_class_proposals": [{"name": "widget_c", "votes": 2}],
    "attribute_summary": [
        {"key": "finish", "answered": 38, "values": [{"value": "matte", "count": 27}, {"value": "gloss", "count": 11}]}
    ],
}


def crop(i: int, subid: str) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=None,
        class_name=None,
        cluster_id=CLUSTER_ID,
        cluster_subid=subid,
        label_validated=False,
        class_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        assist=(
            {
                "run_id": "asr_old",
                "run_created_at": "2026-10-15T00:00:00Z",
                "class_name": "widget_a",
                "class_confidence": "high",
                "attributes": [{"key": "finish", "value": "matte", "confidence": "high"}],
                "boxes": [],
            }
            if i == 0
            else None
        ),
    )


def state(status: str, run_id: str | None, **over) -> dict:
    out = {
        "run_id": run_id,
        "status": status,
        "created_at": "2026-10-15T10:00:00Z",
        "finished_at": None,
        "error": None,
        "request": {"target": {"kind": "cluster", "cluster_id": CLUSTER_ID, "cluster_subid": None},
                    "actions": ["classify"], "dry_run": status == "dry_run"},
        "resolved": RESOLVED,
        "progress": {"processed": 0, "total": 41},
        "result": None,
        "scope_now": None,
    }
    out.update(over)
    return out


COMPLETED = state(
    "completed", "asr_1",
    progress={"processed": 41, "total": 41},
    result={"classify": {"predicted": 41, "updated": 14, "failed": 0, "unreadable": 0, "locked_suggest_only": 27},
            "attributes": None, "verify": None, "segment": None, "proposal": PROPOSAL},
    scope_now={"count": 412, "overlap": 412, "jaccard": 1.0, "changed": False},
)


def register(stub, *, apply_refusal=None, listed=None):
    seen = {"crops_queries": [], "polls": 0}
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/methods(\?|$)", METHODS)

    def crops(request, _m):
        q = parse_qs(urlsplit(request.url).query)
        seen["crops_queries"].append(q)
        subid = q.get("cluster_subid", [None])[0]
        subs = [subid] if subid else ["10000a", "10000b", "10000a", "10000b"]
        items = [crop(i, s) for i, s in enumerate(subs)]
        return {"total": len(items), "page": 1, "page_size": 60, "crops": items}

    stub.on("GET", r"/crops(\?|$)", crops)
    stub.on("GET", r"/assist/runs(\?|$)", {"runs": listed if listed is not None else [COMPLETED]})

    def create(request, _m):
        body = request.post_data_json or {}
        return (200, state("dry_run" if body.get("dry_run", True) else "queued", None if body.get("dry_run", True) else "asr_1"))

    stub.on("POST", r"/assist/runs$", create)

    def poll(_request, _m):
        seen["polls"] += 1
        return state("running", "asr_1", progress={"processed": 12, "total": 41}) if seen["polls"] == 1 else COMPLETED

    stub.on("GET", r"/assist/runs/asr_1$", poll)

    def apply(request, _m):
        body = request.post_data_json or {}
        if apply_refusal and not body.get("allow_changed_scope"):
            return (409, {"detail": {"error": "scope_changed", "message": apply_refusal}})
        dry = body.get("dry_run", True)
        return {
            "dry_run": dry, "class_name": body["class_name"], "selected": 380, "skipped_locked": 32,
            "updated": 0 if dry else 380, "updated_ids": [], "conflicts": [],
            "scope_now": {"count": 412, "overlap": 412, "jaccard": 1.0, "changed": False},
        }

    stub.on("POST", r"/assist/runs/asr_1/apply$", apply)
    return seen


def shots(page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for width, height in ((1600, 1000), (430, 1100)):
        page.set_viewport_size({"width": width, "height": height})
        wait_for_paint(page)
        page.screenshot(path=str(SHOTS / f"{name}-{width}.png"), full_page=True)
    page.set_viewport_size({"width": 1280, "height": 720})


def open_cluster(stub, page, app_url, **kw):
    seen = register(stub, **kw)
    page.goto(f"{app_url}/p/default/clusters/{CLUSTER_ID}")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    return seen


def test_sub_cluster_tab_asks_the_server_for_cluster_subid(stub, page, app_url):
    seen = open_cluster(stub, page, app_url)
    page.get_by_role("button", name="#10000b").click()
    page.wait_for_function("document.querySelectorAll('img').length === 1", timeout=ACTION_TIMEOUT_MS)
    assert any(q.get("cluster_subid") == ["10000b"] for q in seen["crops_queries"]), seen["crops_queries"]
    # The other tab stays offered although the page now holds only one sub-cluster.
    expect(page.get_by_role("button", name="#10000a")).to_be_visible()


def test_assist_dialog_dry_run_then_run_then_apply(stub, page, app_url):
    open_cluster(stub, page, app_url)
    panel = page.get_by_test_id("cluster-assist-proposal")
    expect(panel).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("proposal-summary-class")).to_have_text("widget_a")
    shots(page, "cluster-proposal-folded")
    panel.locator("summary").click()
    expect(panel.get_by_test_id("proposal-class")).to_be_visible()
    shots(page, "cluster-proposal")
    panel.locator("summary").click()

    page.locator("img").nth(0).click()
    page.locator("img").nth(1).click(modifiers=["Control"])
    expect(page.get_by_text("2 selected").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("assist-open").click()
    dialog = page.get_by_test_id("assist-dialog")
    expect(dialog).to_be_visible()
    expect(page.get_by_test_id("assist-choice-selected")).to_contain_text("(2)")

    # The dry run is the server's: its counts are on screen before anything starts.
    expect(page.get_by_test_id("assist-plan-sampled")).to_have_text("41", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("assist-plan-calls")).to_have_text("6")
    assert all(c[2].get("dry_run") is True for c in stub.calls if c[1].endswith("/assist/runs")), stub.calls

    page.get_by_test_id("assist-choice-representative").locator("input").check()
    page.get_by_test_id("assist-size").fill("40")
    page.get_by_test_id("assist-action-attributes").locator("input").check()
    page.get_by_test_id("assist-pack").select_option("widgets")
    expect(page.get_by_test_id("assist-plan-sampled")).to_be_visible()
    shots(page, "dialog-configure")

    # The footer stays reachable on a phone-width viewport: the body scrolls, the buttons do not.
    page.set_viewport_size({"width": 430, "height": 700})
    wait_for_paint(page)
    start_box = page.get_by_test_id("assist-start").bounding_box()
    assert start_box and start_box["y"] + start_box["height"] <= 700, start_box
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.set_viewport_size({"width": 1280, "height": 720})

    with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/assist/runs")
                              and r.request.post_data_json.get("dry_run") is False, timeout=ACTION_TIMEOUT_MS):
        page.get_by_test_id("assist-start").click()
    start = [c for c in stub.calls if c[1].endswith("/assist/runs") and c[2].get("dry_run") is False][-1][2]
    assert start["target"] == {"kind": "cluster", "cluster_id": CLUSTER_ID, "cluster_subid": None}
    assert start["sample"] == {"mode": "representative", "size": 40, "seed": 0, "stratify_by": "auto"}
    assert start["actions"] == ["classify", "attributes"]
    assert start["prompt_pack"] == "widgets"

    expect(page.get_by_test_id("assist-status")).to_contain_text("Completed", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("assist-result")).to_contain_text("41 answered")
    card = page.get_by_test_id("assist-dialog").get_by_test_id("assist-proposal-card")
    expect(card.get_by_test_id("proposal-class")).to_have_text("widget_a")
    apply_btn = card.get_by_test_id("proposal-apply")
    expect(apply_btn).to_contain_text("380", timeout=ACTION_TIMEOUT_MS)
    shots(page, "dialog-proposal")

    # Only the dry run has gone out so far.
    applies = [c for c in stub.calls if c[1].endswith("/apply")]
    assert applies and all(c[2]["dry_run"] is True for c in applies)
    with page.expect_response(lambda r: r.url.endswith("/apply") and r.request.post_data_json.get("dry_run") is False,
                              timeout=ACTION_TIMEOUT_MS):
        apply_btn.click()
    last = [c for c in stub.calls if c[1].endswith("/apply")][-1][2]
    assert last["class_name"] == "widget_a" and last["label_source"] == "human"
    expect(card.get_by_test_id("proposal-applied")).to_contain_text("380")


def test_stale_proposal_shows_the_servers_scope_changed_refusal(stub, page, app_url):
    stale = {**COMPLETED, "scope_now": {"count": 300, "overlap": 186, "jaccard": 0.62, "changed": True}}
    open_cluster(stub, page, app_url, listed=[stale],
                 apply_refusal="only 62% of the scope is what the run saw (a re-refine moved items)")
    page.get_by_test_id("cluster-assist-proposal").locator("summary").click()
    banner = page.get_by_test_id("proposal-stale")
    expect(banner).to_contain_text("186", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("proposal-error")).to_contain_text("only 62% of the scope")
    expect(page.get_by_test_id("proposal-apply")).to_have_count(0)
    shots(page, "proposal-stale")
    page.get_by_test_id("proposal-anyway").click()
    expect(page.get_by_test_id("proposal-apply")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert [c for c in stub.calls if c[1].endswith("/apply")][-1][2]["allow_changed_scope"] is True


def test_dialog_shows_a_served_refusal_and_blocks_start(stub, page, app_url):
    open_cluster(stub, page, app_url)
    stub.on("POST", r"/assist/runs$", (409, {"detail": {"error": "vlm_not_configured", "message": "This project has no VLM; activate one first."}}))
    page.get_by_test_id("assist-open").click()
    expect(page.get_by_test_id("assist-plan-error")).to_contain_text("activate one first", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("assist-start")).to_be_disabled()
    shots(page, "dialog-refusal")


def test_attribute_chips_show_on_the_card_as_suggestions(stub, page, app_url):
    open_cluster(stub, page, app_url)
    chips = page.get_by_test_id("card-assist").first
    expect(chips).to_contain_text("VLM suggests", timeout=ACTION_TIMEOUT_MS)
    expect(chips.get_by_test_id("assist-chip")).to_contain_text("matte")
