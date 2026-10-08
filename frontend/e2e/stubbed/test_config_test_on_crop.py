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

from conftest import ACTION_TIMEOUT_MS, expect_handled
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
    with expect_handled(page, lambda r: r.url.endswith("/prompt_packs/test")):
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
    expect(panel.get_by_test_id("test-latency")).to_have_text("812 ms")
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
    with expect_handled(page, lambda r: r.url.endswith("/region_profiles/test")):
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
    with expect_handled(page, lambda r: r.url.endswith("/region_profiles/test")):
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


# --- Layout: the source-image overlay stays in a bounded column ------------

import struct
import zlib
from pathlib import Path

SHOT_DIR = Path(__file__).resolve().parents[2] / "artifacts_local" / "track-C-fix"


def real_png(width: int, height: int) -> bytes:
    """A real, non-trivial PNG (diagonal gradient + grid), not the 1px GIF."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            grid = 40 if (x % 100 == 0 or y % 100 == 0) else 0
            rows += bytes(((x * 255 // width + grid) % 256, (y * 255 // height) % 256, 140))
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def serve_real_image(stub, width: int, height: int) -> None:
    png = real_png(width, height)
    stub.on("GET", r"/crops/[^/]+/image(/|$|\?)", (200, png, "image/png"))
    stub.on("GET", r"(thumbnail|region_thumbnail)(/|$|\?)", (200, real_png(256, 256), "image/png"))
    item = {
        "crop_id": "c_123",
        "image_id": "img_1",
        "class_id": 1,
        "class_name": "widget",
        "bbox_norm": [0.1, 0.1, 0.5, 0.5],
    }
    stub.on(
        "GET",
        r"/crops/[^/]+/context$",
        (
            200,
            {
                "image": {
                    "image_id": "img_1",
                    "image_path": "/fixtures/img.png",
                    "width": width,
                    "height": height,
                    "source": None,
                    "indexed_at": None,
                },
                "items": [item],
            },
        ),
    )


def assert_no_overlap(page, container_selector: str) -> None:
    """The overlay column stays bounded and no direct child of the result
    container overlaps the next one (the original defect: a huge block over
    the following rows)."""
    boxes = page.evaluate(
        """(sel) => {
          const root = document.querySelector(sel);
          return [...root.children].map((c) => {
            const r = c.getBoundingClientRect();
            return {tag: c.tagName, id: c.dataset.testid || '', top: r.top, bottom: r.bottom,
                    left: r.left, right: r.right, h: r.height, w: r.width};
          });
        }""",
        container_selector,
    )
    for a, b in zip(boxes, boxes[1:]):
        assert a["bottom"] <= b["top"] + 1, ("overlap", a, b)
    assert_overlay_contained(page)


def assert_overlay_contained(page) -> None:
    """The image and its box layer stay inside the preview column; the
    column is bounded (the defect: the image grew to the card's width and
    spilled over the rows below while its wrapper stayed small)."""
    r = page.evaluate(
        """() => {
          const box = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, left: r.left, right: r.right, h: r.height}; };
          const col = document.querySelector('[data-testid="test-preview-item"]');
          return {col: box(col), img: box(col.querySelector('img')), layer: box(col.querySelector('[data-testid="overlay-layer"]'))};
        }"""
    )
    col = r["col"]
    assert col["h"] <= 340, r
    for k in ("img", "layer"):
        assert r[k]["top"] >= col["top"] - 1 and r[k]["bottom"] <= col["bottom"] + 1, (k, r)
        assert r[k]["left"] >= col["left"] - 1 and r[k]["right"] <= col["right"] + 1, (k, r)


def test_pack_test_overlay_is_bounded_and_does_not_overlap(stub, page, app_url):
    serve_packs(stub)
    stub.on("POST", r"/prompt_packs/validate$", CLEAN)
    serve_real_image(stub, 1600, 1000)
    stub.on(
        "POST",
        r"/prompt_packs/test$",
        (
            200,
            {
                "call": "classify",
                "pack": {"name": None, "revision": None, "draft": True},
                "vlm": VLM,
                "prompt": {"system": "s", "user_text": "u"},
                "raw_reply": "[]",
                "reasoning": None,
                "latency_ms": 1.0,
                "parse_ok": True,
                "parse_error": None,
                "validation": CLEAN,
                "results": [
                    {
                        "crop_id": "c_123",
                        "box_id": None,
                        "parsed": {"class_name": "widget"},
                        "skipped": None,
                        "preview_item": {"crop_id": "c_123", "image_id": "img_1", "bbox_norm": [0.1, 0.1, 0.5, 0.5]},
                    }
                ],
            },
        ),
    )
    open_pack_editor(page, app_url)
    panel = page.get_by_test_id("pack-test-panel")
    panel.get_by_test_id("test-crop-ids").fill("c_123")
    panel.get_by_test_id("test-run").click()
    expect(panel.get_by_test_id("overlay-layer")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_function(
        "() => { const i = document.querySelector('[data-testid=\"test-preview-item\"] img'); return i && i.complete && i.naturalWidth > 0; }"
    )
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    for w in (1600, 800):
        page.set_viewport_size({"width": w, "height": 1400})
        page.wait_for_timeout(200)
        assert_no_overlap(page, '[data-testid="test-result-item"]')
        panel.screenshot(path=str(SHOT_DIR / f"pack-test-{w}.png"))


def test_profile_test_overlay_is_bounded_and_does_not_overlap(stub, page, app_url):
    serve_profiles(stub)
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    serve_real_image(stub, 1600, 1000)
    stub.on("POST", r"/region_profiles/test$", (200, region_response()))
    open_profile_editor(page, app_url)
    panel = page.get_by_test_id("profile-test-panel")
    panel.get_by_test_id("profile-test-crop-id").fill("c_123")
    panel.get_by_test_id("test-run").click()
    expect(panel.get_by_test_id("crop-frame-shapes")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_function(
        "() => { const i = document.querySelector('[data-testid=\"test-preview-item\"] img'); return i && i.complete && i.naturalWidth > 0; }"
    )
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    for w in (1600, 800):
        page.set_viewport_size({"width": w, "height": 1600})
        page.wait_for_timeout(200)
        # The preview and crop-frame columns sit side by side (stacked
        # narrow): neither may overlap the other.
        rects = page.evaluate(
            """() => ['test-preview-item', 'crop-frame-shapes'].map((id) => {
              const r = document.querySelector(`[data-testid="${id}"]`).getBoundingClientRect();
              return {id, top: r.top, bottom: r.bottom, left: r.left, right: r.right, h: r.height};
            })"""
        )
        a, b = rects
        separate = a["right"] <= b["left"] + 1 or b["right"] <= a["left"] + 1 or a["bottom"] <= b["top"] + 1 or b["bottom"] <= a["top"] + 1
        assert separate, rects
        assert a["h"] <= 340, rects
        assert_overlay_contained(page)
        panel.screenshot(path=str(SHOT_DIR / f"profile-test-{w}.png"))


# --- The per-run VLM picker on the test panels (W9 x W5) --------------------

from test_vlm_run_selection import BASE_STRATEGIES, VLM_STRATEGIES, WARNING


def serve_vlm_axis(stub) -> None:
    stub.on("GET", r"/methods(\?|$)", {"strategies": [*BASE_STRATEGIES, *VLM_STRATEGIES], "flags": {}})


def test_pack_test_sends_the_picked_vlm_and_the_acknowledgement(stub, page, app_url):
    serve_packs(stub)
    serve_vlm_axis(stub)
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
                "prompt": {"system": "s", "user_text": "u"},
                "raw_reply": "[]",
                "reasoning": None,
                "latency_ms": 1.0,
                "parse_ok": True,
                "parse_error": None,
                "validation": CLEAN,
                "results": [],
            },
        )

    stub.on("POST", r"/prompt_packs/test$", run_test)
    open_pack_editor(page, app_url)
    panel = page.get_by_test_id("pack-test-panel")
    panel.get_by_test_id("test-crop-ids").fill("c_123")
    picker = panel.get_by_test_id("vlm-run-picker")
    expect(picker).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert picker.locator("option").all_inner_texts()[0] == "Active endpoint"

    # Left on the active endpoint: no vlm field.
    with expect_handled(page, lambda r: r.url.endswith("/prompt_packs/test")):
        panel.get_by_test_id("test-run").click()
    assert "vlm_name" not in bodies[0] and "acknowledge_external" not in bodies[0], bodies

    # An external endpoint: the served warning, then the acknowledgement.
    panel.get_by_test_id("vlm-run-select").select_option("cloud_vlm")
    expect(panel.get_by_test_id("vlm-run-ack")).to_contain_text(WARNING)
    panel.get_by_test_id("vlm-run-ack-checkbox").check()
    with expect_handled(page, lambda r: r.url.endswith("/prompt_packs/test")):
        panel.get_by_test_id("test-run").click()
    assert bodies[1]["vlm_name"] == "cloud_vlm", bodies
    assert bodies[1]["vlm_revision"] is None
    assert bodies[1]["acknowledge_external"] is True


def test_profile_test_vlm_picker_follows_verify_and_sends_the_pick(stub, page, app_url):
    serve_profiles(stub)
    serve_vlm_axis(stub)
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    bodies: list[Any] = []

    def run_test(request: Any, _m: Any):
        bodies.append(request.post_data_json)
        return (200, region_response())

    stub.on("POST", r"/region_profiles/test$", run_test)
    open_profile_editor(page, app_url)
    panel = page.get_by_test_id("profile-test-panel")
    expect(panel.get_by_test_id("vlm-run-picker")).to_have_count(0)
    panel.get_by_test_id("profile-test-crop-id").fill("c_123")
    panel.get_by_test_id("test-verify").check()
    expect(panel.get_by_test_id("vlm-run-picker")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    panel.get_by_test_id("vlm-run-select").select_option("local_vlm")
    with expect_handled(page, lambda r: r.url.endswith("/region_profiles/test")):
        panel.get_by_test_id("test-run").click()
    assert bodies[0]["verify"] is True
    assert bodies[0]["vlm_name"] == "local_vlm" and bodies[0]["vlm_revision"] is None, bodies
    assert "acknowledge_external" not in bodies[0]

    panel.get_by_test_id("vlm-run-select").select_option("cloud_vlm")
    panel.get_by_test_id("vlm-run-ack-checkbox").check()
    with expect_handled(page, lambda r: r.url.endswith("/region_profiles/test")):
        panel.get_by_test_id("test-run").click()
    assert bodies[1]["vlm_name"] == "cloud_vlm" and bodies[1]["acknowledge_external"] is True, bodies

    # Verify off: the picker goes and the selection is not sent.
    panel.get_by_test_id("test-verify").uncheck()
    expect(panel.get_by_test_id("vlm-run-picker")).to_have_count(0)
    with expect_handled(page, lambda r: r.url.endswith("/region_profiles/test")):
        panel.get_by_test_id("test-run").click()
    assert "vlm_name" not in bodies[2] and "verify" not in bodies[2], bodies
