"""/models — 2026-09-25 follow-up to OpenProcessor #36 item 5.

GET {API_PREFIX}/models/status now serves `unloadable` on every entry and
widens `is_region_protected` to cover the ingest primary/secondary and
OCR det/rec models, not just the region detector; the segmenter (`sam3`)
is now `kind: "external"` with a `not_configured` status possible and
null inference/exec/latency fields.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

MODELS = {
    "models": [
        {
            "name": "yolov11_small_trt_end2end",
            "friendly_name": "Primary Item Proposer",
            "role": "Proposes item boxes when images are ingested.",
            "kind": "triton",
            "model_type": "TensorRT detection",
            "status": "ready",
            "version": "1",
            "inference_count": 9344,
            "exec_count": 2040,
            "inference_failed": 0,
            "avg_latency_ms": 2.19,
            "last_error": None,
            "endpoint": "http://triton-server:8000",
            "is_region_protected": True,
            "requires_force_to_unload": False,
            "job_id": None,
            "promoted_at": None,
            "unloadable": True,
        },
        {
            "name": "sam3",
            "friendly_name": "Segmenter",
            "role": "Refines or re-detects the region of interest on crops the primary detector missed.",
            "kind": "external",
            "model_type": "Promptable segmentation",
            "status": "not_configured",
            "version": None,
            "inference_count": None,
            "exec_count": None,
            "inference_failed": None,
            "avg_latency_ms": None,
            "last_error": None,
            "endpoint": None,
            "unloadable": False,
        },
        {
            "name": "pe_image_encoder",
            "friendly_name": "PE-Core-L14-336 Image Encoder",
            "role": "Generates 1024-d unit-norm embeddings for semantic search.",
            "kind": "triton",
            "model_type": "ONNX Runtime encoder",
            "status": "ready",
            "version": "1",
            "inference_count": 14825,
            "exec_count": 3383,
            "inference_failed": 0,
            "avg_latency_ms": 60.4,
            "last_error": None,
            "endpoint": "http://triton-server:8000",
            "is_region_protected": False,
            "requires_force_to_unload": False,
            "job_id": None,
            "promoted_at": None,
            "unloadable": True,
        },
    ]
}


def test_not_configured_status_and_null_counts_render_as_dashes(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", MODELS)

    page.goto(f"{app_url}/p/default/models")
    page.get_by_text("Segmenter", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)

    card = page.locator("li", has_text="Segmenter")
    assert "not configured" in card.inner_text().lower(), card.inner_text()
    # Null inference_count/avg_latency_ms render as "—", never "0" / blank.
    assert card.inner_text().count("—") >= 2, card.inner_text()

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_region_protected_model_shows_protected_chip_not_an_unload_button(
    stub, page, app_url
):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", MODELS)

    page.goto(f"{app_url}/p/default/models")
    page.get_by_text("Primary Item Proposer", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)

    card = page.locator("li", has_text="Primary Item Proposer")
    assert "protected: in use by the pipeline" in card.inner_text().lower(), (
        card.inner_text()
    )
    assert card.get_by_role("button", name="Unload").count() == 0
    assert card.get_by_role("button", name="Force unload").count() == 0


def test_external_unloadable_false_model_shows_neither_button_nor_chip(
    stub, page, app_url
):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", MODELS)

    page.goto(f"{app_url}/p/default/models")
    page.get_by_text("Segmenter", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)

    card = page.locator("li", has_text="Segmenter")
    assert "protected" not in card.inner_text().lower(), card.inner_text()
    assert card.get_by_role("button", name="Unload").count() == 0


def test_ordinary_unloadable_model_still_offers_a_plain_unload_button(
    stub, page, app_url
):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", MODELS)

    page.goto(f"{app_url}/p/default/models")
    page.get_by_text("PE-Core-L14-336 Image Encoder", exact=False).wait_for(
        timeout=ACTION_TIMEOUT_MS
    )

    card = page.locator("li", has_text="PE-Core-L14-336 Image Encoder")
    assert card.get_by_role("button", name="Unload").count() == 1


def test_unload_refused_detector_in_use_shows_reason_and_offers_force(
    stub, page, app_url
):
    """OpenProcessor #75/#121: the ingest detector is a 409 detector_in_use
    without force; the served reason is shown and a confirmed retry sends
    force=true."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", MODELS)
    reason = "'pe_core' is this project's ingest detector; deleting it makes ingest fail"
    seen: list[str] = []

    def delete(request, _match):
        seen.append(request.url)
        if "force=true" in request.url:
            return (
                200,
                {"triton_name": "pe_core", "triton_unloaded": True,
                 "directory_removed": True, "forced": True, "warning": None},
            )
        return (409, {"detail": {"error": "detector_in_use", "message": reason}})

    stub.on("DELETE", r"/models/[^/?]+(\?|$)", delete)
    messages: list[str] = []

    def on_dialog(d):
        messages.append(d.message)
        d.accept()

    page.on("dialog", on_dialog)
    page.goto(f"{app_url}/p/default/models")
    card = page.locator("li", has_text="PE-Core-L14-336 Image Encoder")
    card.wait_for(timeout=ACTION_TIMEOUT_MS)
    card.get_by_role("button", name="Unload").click()
    deadline = ACTION_TIMEOUT_MS
    while len(seen) < 2 and deadline > 0:
        page.wait_for_timeout(100)
        deadline -= 100
    assert len(seen) == 2, seen
    assert "force=true" not in seen[0]
    assert "force=true" in seen[1]
    assert any(reason in m for m in messages), messages
