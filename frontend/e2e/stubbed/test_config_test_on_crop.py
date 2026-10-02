"""OpenProcessor W5 (test-on-crop for prompt packs and region profiles),
any_domain_plan.md §5, §7.5-7.7; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
§5.2-5.4. Payloads follow the vendored contract (PackTestResponse,
RegionTestResponse), in the neutral widget/tag domain. The fail-closed stub
means any request the page makes that a test did not expect fails the test.

  1. `test_pack_test_sends_the_draft_and_renders_the_contract_response` —
     body asserted (draft, call, crop ids, nothing unchosen); the pack and VLM
     refs, prompt, raw reply, parsed answer and preview render; a skipped crop
     shows its served reason.
  2. `test_profile_test_draws_candidates_and_greys_a_dropped_one` — the legs
     table, a dropped candidate greyed, a mask polygon drawn over the source
     image and in the crop frame, the "Selection (not verified)" heading.
  3. `test_profile_test_verify_block_and_vlm_verdicts_heading`.
  4. `test_crop_not_found_shows_the_served_message_and_the_id`.
"""

from __future__ import annotations

from typing import Any

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_prompt_packs import BODY as PACK_BODY
from test_prompt_packs import CLEAN
from test_prompt_packs import open_editor as open_pack_editor
from test_prompt_packs import serve_packs
from test_region_profiles import open_editor as open_profile_editor
from test_region_profiles import serve_profiles


def candidate(index: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "bbox_correct": None,
        "bbox_norm": [0.2, 0.2, 0.6, 0.5],
        "bbox_in_parent": [0.1, 0.1, 0.9, 0.9],
        "box_id": None,
        "candidate_index": index,
        "cluster_distance": None,
        "cluster_id": None,
        "cluster_subid": None,
        "confidence": None,
        "detected_at": None,
        "detector": "tag_detector_v1",
        "detector_version": None,
        "drop_reason": None,
        "locked": False,
        "mask_iou": None,
        "mask_polygon": None,
        "mask_polygon_in_parent": None,
        "rejection_reason": None,
        "score": 0.91,
        "selected": True,
        "source": None,
        "state": "detected",
        "text": None,
        "text_choice": None,
        "text_confidence": None,
        "text_disagreement": None,
        "text_engine_version": None,
        "text_ocr": None,
        "text_raw": None,
        "text_source": None,
        "text_vlm": None,
        "text_vlm_invalid": None,
        "thumbnail_url": None,
    }
    base.update(over)
    return base


LEGS = [
    {
        "leg": "detector",
        "status": "ok",
        "reason": None,
        "elapsed_ms": 41,
        "candidates": [
            candidate(0),
            candidate(
                1,
                selected=False,
                drop_reason="below_min_score",
                score=0.12,
                bbox_norm=[0.7, 0.1, 0.9, 0.3],
                bbox_in_parent=[0.6, 0.6, 0.8, 0.8],
            ),
        ],
    },
    {
        "leg": "segmenter",
        "status": "ok",
        "reason": None,
        "elapsed_ms": 220,
        "candidates": [
            candidate(
                0,
                detector="tag_segmenter",
                mask_iou=0.83,
                mask_polygon=[[0.2, 0.2], [0.6, 0.2], [0.4, 0.5]],
                mask_polygon_in_parent=[[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            )
        ],
    },
]

VLM = {"name": "env", "revision": None, "draft": False, "endpoint": "env@abc123", "model": "example/vision-model"}


def region_response(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "crop_id": "c_123",
        "item_eligible": True,
        "legs": LEGS,
        "preview_basis": "selection_accepted",
        "preview_item": {"crop_id": "c_123", "image_id": "img_1", "bbox_norm": [0.1, 0.1, 0.5, 0.5]},
        "profile": {"name": "widget_tag", "revision": 2, "draft": False},
        "validation": CLEAN,
        "verify": None,
    }
    base.update(over)
    return base


def test_pack_test_sends_the_draft_and_renders_the_contract_response(stub, page, app_url):
    serve_packs(stub)
    stub.on("POST", r"/prompt_packs/validate$", CLEAN)
    bodies: list[Any] = []

    def run_test(request: Any, _m: Any):
        bodies.append(request.post_data_json)
        return (
            200,
            {
                "call": "classify",
                "pack": {"name": None, "revision": None, "draft": True},
                "vlm": VLM,
                "prompt": {"system": "You classify widgets.", "user_text": "Pick one of: widget, gadget"},
                "raw_reply": '[{"img": 1, "class": "widget", "confidence": 0.91}]',
                "reasoning": "It has a handle.",
                "latency_ms": 812.4,
                "parse_ok": True,
                "parse_error": None,
                "validation": CLEAN,
                "results": [
                    {
                        "crop_id": "c_123",
                        "box_id": None,
                        "parsed": {"class_name": "widget", "confidence": 0.91},
                        "skipped": None,
                        "preview_item": {"crop_id": "c_123", "bbox_norm": [0.1, 0.1, 0.5, 0.5]},
                    },
                    {
                        "crop_id": "c_456",
                        "box_id": "b_1",
                        "parsed": None,
                        "skipped": "no stored box to verify",
                        "preview_item": None,
                    },
                ],
            },
        )

    stub.on("POST", r"/prompt_packs/test$", run_test)

    open_pack_editor(page, app_url)
    page.locator('[data-field="class_system"] textarea').fill("You sort widgets.")
    panel = page.get_by_test_id("pack-test-panel")
    panel.get_by_test_id("test-crop-ids").fill("c_123, c_456")
    with page.expect_request(lambda r: r.url.endswith("/prompt_packs/test")):
        panel.get_by_test_id("test-run").click()
    # Only what the operator chose: no use_region_box, no VLM selection.
    assert bodies == [
        {
            "draft": {**PACK_BODY, "class_system": "You sort widgets."},
            "call": "classify",
            "crop_ids": ["c_123", "c_456"],
        }
    ], bodies

    expect(panel.get_by_test_id("test-raw-reply")).to_contain_text('"class": "widget"', timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("test-pack-ref")).to_have_text("draft")
    expect(panel.get_by_test_id("test-vlm-ref")).to_contain_text("env@abc123")
    expect(panel.get_by_test_id("test-latency")).to_have_text("812.4 ms")
    expect(panel.get_by_test_id("test-prompt-user")).to_have_text("Pick one of: widget, gadget")
    items = panel.get_by_test_id("test-result-item")
    expect(items).to_have_count(2)
    expect(items.nth(0).get_by_test_id("test-parsed")).to_contain_text('"class_name": "widget"')
    expect(items.nth(0).get_by_test_id("test-preview-item")).to_be_visible()
    expect(items.nth(1).get_by_test_id("test-skipped")).to_have_text("no stored box to verify")
    expect(items.nth(1).get_by_test_id("test-parsed")).to_have_count(0)


def test_profile_test_draws_candidates_and_greys_a_dropped_one(stub, page, app_url):
    serve_profiles(stub)
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    bodies: list[Any] = []

    def run_test(request: Any, _m: Any):
        bodies.append(request.post_data_json)
        return (200, region_response())

    stub.on("POST", r"/region_profiles/test$", run_test)

    open_profile_editor(page, app_url)
    panel = page.get_by_test_id("profile-test-panel")
    panel.get_by_test_id("profile-test-crop-id").fill("c_123")
    with page.expect_request(lambda r: r.url.endswith("/region_profiles/test")):
        panel.get_by_test_id("test-run").click()
    assert len(bodies) == 1
    assert bodies[0]["crop_id"] == "c_123"
    assert "draft" in bodies[0] and "profile_name" not in bodies[0], bodies
    for unchosen in ("verify", "segmenter_text_prompt", "vlm_name", "acknowledge_external"):
        assert unchosen not in bodies[0], bodies

    expect(panel.get_by_test_id("test-profile-ref")).to_have_text("widget_tag@2", timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("test-leg")).to_have_count(2)
    rows = panel.get_by_test_id("test-candidate")
    expect(rows).to_have_count(3)
    dropped = panel.locator('[data-testid="test-candidate"][data-selected="false"]')
    expect(dropped).to_have_count(1)
    expect(dropped).to_contain_text("Below min score")
    assert "opacity-50" in (dropped.get_attribute("class") or "")
    kept = panel.locator('[data-testid="test-candidate"][data-selected="true"]').first
    assert "opacity-50" not in (kept.get_attribute("class") or "")

    # Candidates over the source image: a dimmed dropped box, a mask polygon.
    expect(panel.get_by_test_id("overlay-extra-polygon")).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("overlay-extra-box")).to_have_count(3)
    expect(panel.locator('[data-testid="overlay-extra-box"][data-dimmed="true"]')).to_have_count(1)
    # And in the crop's own frame, from the server-projected geometry.
    expect(panel.get_by_test_id("crop-frame-polygon")).to_have_count(1)
    expect(panel.get_by_test_id("crop-frame-box")).to_have_count(3)
    expect(panel).to_contain_text("Selection (not verified)")
    expect(panel.get_by_test_id("test-verify-block")).to_have_count(0)


def test_profile_test_verify_block_and_vlm_verdicts_heading(stub, page, app_url):
    serve_profiles(stub)
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    bodies: list[Any] = []

    def run_test(request: Any, _m: Any):
        bodies.append(request.post_data_json)
        return (
            200,
            region_response(
                preview_basis="vlm_verdicts",
                verify={
                    "latency_ms": 530.0,
                    "pack": {"name": "widget_tag", "revision": 3, "draft": False},
                    "parse_ok": True,
                    "parse_error": None,
                    "prompt": {"system": "Verify the tag box.", "user_text": "Is this a tag?"},
                    "raw_reply": '[{"box": 1, "ok": true}]',
                    "reasoning": None,
                    "vlm": VLM,
                },
            ),
        )

    stub.on("POST", r"/region_profiles/test$", run_test)

    open_profile_editor(page, app_url)
    panel = page.get_by_test_id("profile-test-panel")
    panel.get_by_test_id("profile-test-crop-id").fill("c_123")
    panel.get_by_test_id("test-segmenter-prompt").fill("a price tag")
    panel.get_by_test_id("test-verify").check()
    with page.expect_request(lambda r: r.url.endswith("/region_profiles/test")):
        panel.get_by_test_id("test-run").click()
    assert bodies[0]["verify"] is True
    assert bodies[0]["segmenter_text_prompt"] == "a price tag"

    block = panel.get_by_test_id("test-verify-block")
    expect(block).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(block).to_contain_text("widget_tag@3")
    expect(block).to_contain_text("env@abc123")
    expect(block.get_by_test_id("test-prompt-user")).to_have_text("Is this a tag?")
    expect(block.get_by_test_id("test-raw-reply")).to_contain_text('"ok": true')
    expect(panel).to_contain_text("VLM verdicts")
    expect(panel).not_to_contain_text("Selection (not verified)")


def test_crop_not_found_shows_the_served_message_and_the_id(stub, page, app_url):
    serve_profiles(stub)
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    stub.on(
        "POST",
        r"/region_profiles/test$",
        (404, {"detail": {"error": "crop_not_found", "message": "No crop with id c_404.", "crop_ids": ["c_404"]}}),
    )

    open_profile_editor(page, app_url)
    panel = page.get_by_test_id("profile-test-panel")
    panel.get_by_test_id("profile-test-crop-id").fill("c_404")
    panel.get_by_test_id("test-run").click()
    expect(panel.get_by_test_id("test-error")).to_contain_text("No crop with id c_404.", timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("test-missing-ids")).to_contain_text("c_404")
    expect(panel.get_by_test_id("profile-test-crop-id")).to_have_attribute("aria-invalid", "true")
    expect(panel.get_by_test_id("test-result")).to_have_count(0)
