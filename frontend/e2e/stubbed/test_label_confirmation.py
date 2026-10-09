"""Label confirmation (#119, docs/design/label_confirmation_plan.md section 7):
a VLM label is a suggestion until a human validates it, and the detector's own
class is kept beside it.

  1. VLM scope panel on /settings: load, gated knobs, save with the revision,
     409 conflict.
  2. The `detector_disagreements` review tab: served-only (absent unless
     `GET /review/tabs` serves it), label and description as served, and the
     item shows the detector class and the "VLM suggestion" wording.
  3. /audit: draw a sample, the served report, an insufficient-sample class,
     the confusion matrix, the queue, a refused draw.
  4. /export warns that only validated crops are exported.
  5. /dashboard says a label is not ground truth.
"""

from __future__ import annotations

import json

from conftest import ACTION_TIMEOUT_MS
from fixtures.wire import make_item, review_tab, review_tabs
from playwright.sync_api import expect

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget_a",
        "kind": "item",
        "group": "g",
        "hotkey_letter": "k",
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
]

CLASS_SOURCES = {
    "class_sources": [
        {"id": "vlm", "label": "Labeled by the VLM", "role": "vlm", "short_label": "VLM"},
        {"id": "human", "label": "Human", "role": "human", "short_label": "H"},
    ]
}

POLICY = {
    "scope": "all",
    "conf_max": 0.8,
    "per_cluster": 5,
    "max_crops_per_day": 0,
    "sample_frac": 1.0,
    "revision": 3,
}


# -- 1. VLM scope panel ------------------------------------------------------


def _stub_settings_page(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})


def _stub_settings(stub):
    state = {"policy": dict(POLICY)}
    _stub_settings_page(stub)
    stub.on("GET", r"/vlm/policy$", lambda _r, _m: dict(state["policy"]))

    def put(request, _m):
        body = json.loads(request.post_data)
        if body["expected_revision"] != state["policy"]["revision"]:
            return (
                409,
                {"detail": {"error": "revision_conflict", "message": "vlm policy changed: 3 != 9"}},
            )
        new = {k: v for k, v in body.items() if k != "expected_revision"}
        new["revision"] = state["policy"]["revision"] + 1
        state["policy"] = new
        return new

    stub.on("PUT", r"/vlm/policy$", put)
    return state


def test_vlm_scope_panel_saves_the_policy_with_its_revision(stub, page, app_url):
    state = _stub_settings(stub)
    page.goto(f"{app_url}/p/default/settings")
    panel = page.get_by_test_id("vlm-scope-panel")
    panel.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("vlm-scope-all")).to_be_checked(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("vlm-knob-per_cluster")).to_have_count(0)

    page.get_by_test_id("vlm-scope-representatives").check()
    per_cluster = page.get_by_test_id("vlm-knob-per_cluster")
    expect(per_cluster).to_have_value("5")
    expect(page.get_by_test_id("vlm-knob-conf_max")).to_have_count(0)
    per_cluster.fill("8")
    page.get_by_test_id("vlm-knob-max_crops_per_day").fill("500")

    page.get_by_test_id("vlm-scope-save").click()
    expect(page.get_by_test_id("vlm-scope-saved")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    puts = [c for c in stub.calls if c[0] == "PUT" and c[1].endswith("/vlm/policy")]
    assert len(puts) == 1
    assert puts[0][2]["scope"] == "representatives"
    assert puts[0][2]["per_cluster"] == 8
    assert puts[0][2]["max_crops_per_day"] == 500
    assert puts[0][2]["expected_revision"] == 3
    assert state["policy"]["revision"] == 4


def test_vlm_scope_panel_offers_reload_on_a_stale_revision(stub, page, app_url):
    state = _stub_settings(stub)
    page.goto(f"{app_url}/p/default/settings")
    page.get_by_test_id("vlm-scope-panel").wait_for(timeout=ACTION_TIMEOUT_MS)
    state["policy"]["revision"] = 9  # someone else saved first
    page.get_by_test_id("vlm-scope-off").check()
    expect(page.get_by_test_id("vlm-scope-knobs")).to_have_count(0)
    page.get_by_test_id("vlm-scope-save").click()
    err = page.get_by_test_id("vlm-scope-save-error")
    expect(err).to_contain_text("vlm policy changed: 3 != 9", timeout=ACTION_TIMEOUT_MS)
    expect(err.get_by_role("button", name="Reload")).to_be_visible()
    expect(err.get_by_role("button", name="Keep my edits")).to_be_visible()


def test_vlm_scope_panel_shows_the_served_error(stub, page, app_url):
    _stub_settings_page(stub)
    page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_test_id("vlm-scope-load-error")).to_be_visible(timeout=ACTION_TIMEOUT_MS)


# -- 2. detector_disagreements review tab -----------------------------------


def _disagreement_item():
    return make_item(
        crop_id="crop-d1",
        image_id="img-d1",
        class_id=1,
        class_name="widget_a",
        class_source="vlm",
        label_source="vlm",
        class_validated=False,
        label_validated=False,
        detector_class_name="widget_h",
        detector_class_id=8,
        detector_confidence=0.72,
    )


def _stub_review(stub, tabs):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/class_sources(\?|$)", CLASS_SOURCES)
    stub.on("GET", r"/review/tabs(\?|$)", tabs)
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        lambda _r, _m: {"items": [_disagreement_item()], "total": 1, "page": 1, "page_size": 30},
    )


def test_detector_disagreements_tab_is_served_only(stub, page, app_url):
    _stub_review(stub, review_tabs(review_tab("all", "All")))
    page.goto(f"{app_url}/p/default/review?tab=all")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role("tab", name="Detector disagreements")).to_have_count(0)
    expect(page.get_by_text("Detector Disagreements")).to_have_count(0)


def test_detector_disagreements_tab_shows_served_label_and_item_wording(stub, page, app_url):
    _stub_review(
        stub,
        review_tabs(
            review_tab("all", "All"),
            review_tab(
                "detector_disagreements",
                "Detector disagreements",
                description="VLM labels that differ from the detector's class, hardest first.",
            ),
        ),
    )
    seen = []
    stub.on(
        "GET",
        r"/review/detector_disagreements(\?|$)",
        lambda req, _m: (
            seen.append(req.url)
            or {"items": [_disagreement_item()], "total": 1, "page": 1, "page_size": 30}
        ),
    )
    page.goto(f"{app_url}/p/default/review?tab=detector_disagreements")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    tab = page.get_by_text("Detector disagreements", exact=True).first
    expect(tab).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert seen, "the tab must call GET /review/detector_disagreements"
    # The item panel: detector class beside the VLM label, VLM wording.
    expect(page.get_by_test_id("detector-class").first).to_contain_text(
        "widget_h 72%", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("confirmation-chip").first).to_have_text("VLM suggestion")


def test_deep_link_to_an_unserved_detector_tab_falls_back_to_all(stub, page, app_url):
    _stub_review(stub, review_tabs(review_tab("all", "All")))
    page.goto(f"{app_url}/p/default/review?tab=detector_disagreements")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('There is no "detector_disagreements" review tab')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )


# -- 3. audit ----------------------------------------------------------------

REPORT = {
    "audited": 52,
    "pending": 31,
    "min_per_class": 30,
    "detector": [
        {"name": "widget_a", "n": 40, "correct": 38, "precision": 0.95, "ci_low": 0.835, "ci_high": 0.986, "insufficient_sample": False},
        {"name": "widget_b", "n": 12, "correct": 6, "precision": 0.5, "ci_low": 0.254, "ci_high": 0.746, "insufficient_sample": True},
    ],
    "vlm": [
        {"name": "widget_a", "n": 30, "correct": 27, "precision": 0.9, "ci_low": 0.744, "ci_high": 0.965, "insufficient_sample": False},
    ],
    "confusion": {
        "widget_a": {"widget_a": 38, "widget_b": 2},
        "widget_b": {"widget_b": 6, "widget_a": 6},
    },
    "outcomes": {"agree": 44, "detector_wrong": 5, "vlm_wrong": 2, "both_wrong": 1},
}


def _stub_audit(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/class_sources(\?|$)", CLASS_SOURCES)
    stub.on("GET", r"/audit/report(\?|$)", REPORT)
    stub.on(
        "GET",
        r"/audit/queue(\?|$)",
        {"items": [_disagreement_item()], "total": 31, "page": 1, "page_size": 30},
    )


def test_audit_page_renders_the_served_report_and_queue(stub, page, app_url):
    _stub_audit(stub)
    page.goto(f"{app_url}/p/default/audit")
    expect(page.get_by_test_id("audit-audited")).to_have_text("52", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("audit-pending")).to_have_text("31")
    rows = page.get_by_test_id("audit-detector").get_by_test_id("audit-class-row")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text("95.0%")
    expect(rows.nth(1).get_by_test_id("insufficient-sample")).to_be_visible()
    expect(page.get_by_test_id("confusion-cell")).to_have_count(4)
    expect(page.get_by_test_id("audit-queue-item")).to_have_count(1)
    open_link = page.get_by_test_id("audit-queue-open").first
    assert "/review?tab=all&crop_id=crop-d1" in open_link.get_attribute("href")


def test_audit_draw_sends_only_the_typed_values_and_shows_the_strata(stub, page, app_url):
    _stub_audit(stub)
    stub.on(
        "POST",
        r"/audit/start$",
        {
            "batch_id": "b1",
            "min_per_class": 30,
            "requested": 120,
            "sampled": 38,
            "strata": [
                {"detector_class": "widget_a", "available": 500, "sampled": 30, "short_of_floor": False},
                {"detector_class": "widget_b", "available": 90, "sampled": 8, "short_of_floor": True},
            ],
        },
    )
    page.goto(f"{app_url}/p/default/audit")
    page.get_by_test_id("audit-start").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("audit-sample-size").fill("120")
    page.get_by_test_id("audit-start-button").click()
    expect(page.get_by_test_id("audit-started")).to_contain_text(
        "Drew 38 of the 120 requested crops", timeout=ACTION_TIMEOUT_MS
    )
    posts = [c for c in stub.calls if c[0] == "POST" and c[1].endswith("/audit/start")]
    assert posts[0][2] == {"sample_size": 120}
    strata = page.get_by_test_id("audit-stratum")
    expect(strata).to_have_count(2)
    assert strata.nth(1).get_attribute("data-short") == "true"


def test_audit_draw_refusal_is_shown_as_served(stub, page, app_url):
    _stub_audit(stub)
    stub.on(
        "POST",
        r"/audit/start$",
        (409, {"detail": {"error": "audit_no_candidates", "message": "no crop is eligible for an audit"}}),
    )
    page.goto(f"{app_url}/p/default/audit")
    page.get_by_test_id("audit-start").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("audit-start-button").click()
    expect(page.get_by_test_id("audit-start-error")).to_contain_text(
        "no crop is eligible for an audit", timeout=ACTION_TIMEOUT_MS
    )


def test_audit_is_in_the_primary_nav(stub, page, app_url):
    _stub_audit(stub)
    page.goto(f"{app_url}/p/default/audit")
    nav = page.get_by_test_id("primary-nav")
    expect(nav.get_by_role("link", name="Audit")).to_have_attribute(
        "aria-current", "page", timeout=ACTION_TIMEOUT_MS
    )


# -- 4. export ---------------------------------------------------------------


def test_export_warns_that_only_validated_crops_are_exported(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on(
        "GET",
        r"/stats/dataset(\?|$)",
        {"total_crops": 14023, "validated": 120, "test_holdout": 0, "by_source": []},
    )
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": [], "min_test_per_class": 5})
    stub.on("GET", r"/export/status(\?|$)", {"status": "idle", "last_run": None})
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": [], "count": 0})
    page.goto(f"{app_url}/p/default/export")
    notice = page.get_by_test_id("validated-ratio")
    expect(notice).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(notice).to_have_attribute("data-level", "partial")
    expect(page.get_by_test_id("validated-ratio-counts")).to_have_text("120 of 14,023")
    expect(notice).to_contain_text("Only validated crops are exported")


# -- 5. dashboard ------------------------------------------------------------


def test_dashboard_says_a_label_is_not_ground_truth(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/pipeline/auto_label/status", {
        "job_id": "job-0", "status": "idle", "stage": "", "processed": 0, "total": 0,
        "started_at": 0, "finished_at": 0, "error": None, "result": {}, "args": {},
        "eta_seconds": None, "elapsed_seconds": 0,
    })
    stub.on("GET", r"/stats/dataset(\?|$)", {"total_crops": 3, "validated": 0, "test_holdout": 0, "by_source": []})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})
    stats = {
        "as_of": "2026-10-09T00:00:00Z",
        "total_crops": 14023,
        "validated": 120,
        "validated_by_import": 0,
        "test_holdout": 0,
        "by_source": [{"key": "vlm", "doc_count": 11103}],
        "labeled": {"by_human": 120, "by_vlm": 11103, "by_classifier": 0, "by_import": 0, "other": 0},
        "regions": {"boxed": 0, "confirmed": 0, "total_detected": 0, "by_detector": 0, "by_segmenter": 0,
                    "by_human_drew": 0, "verified_by_human": 0, "verified_by_vlm": 0, "validated_by_human": 0},
        "unlabeled": {"pending_detection": 0, "pending_verification": 0, "no_label_source": 2920,
                      "vlm_no_class": 0, "by_proposal": 0},
        "in_progress": {"region_drain_total_unfinished": 0},
        "clusters": {"cluster_count": 0, "residual_count": 0, "noise_count": 0, "last_run_at": None, "method": None},
        "embedding": {"embedded": 0, "not_embedded": 0, "by_state": {}},
    }
    snapshot = {"state": {}, "stats": stats}
    stub.on(
        "GET",
        r"/pipeline/events",
        (200, f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n", "text/event-stream"),
    )
    page.goto(f"{app_url}/p/default/dashboard")
    note = page.get_by_test_id("labeled-not-ground-truth")
    expect(note).to_contain_text("not ground truth until a human validates it", timeout=ACTION_TIMEOUT_MS)
    expect(note).to_contain_text("120 of 14,023 crops are validated")
    expect(page.get_by_text("VLM suggestions", exact=True)).to_be_visible()
