"""Ingest-profile list-and-activate on `/settings/ingest-policy` (#440).

  1. `test_lists_profiles_and_marks_the_active_one`
  2. `test_activate_posts_expected_active_and_refreshes_the_active_card`
  3. `test_a_refusal_shows_the_served_issue_messages` -- one 422 per served
     refusal code; the override appears only when the report allows it.
"""

from __future__ import annotations

import pytest
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_ingest_policy import ACTIVE_PROFILE, POLICY, PREVIEW, _ingest_config

LIST = {
    "profiles": [
        {
            "name": "yolo-primary",
            "source": "stored",
            "read_only": False,
            "revision": 3,
            "etag": "a",
            "primary_model": "yolo11_small_trt",
            "active": True,
            "active_revision": 3,
        },
        {
            "name": "yolo-large",
            "source": "stored",
            "read_only": False,
            "revision": 2,
            "etag": "b",
            "primary_model": "yolo11_large_trt",
            "secondary_model": None,
            "active": False,
        },
    ],
    "templates": [],
    "active": {"name": "yolo-primary", "revision": 3},
    "config_revision": 9,
    "stale": False,
}

OK_REPORT = {"ok": True, "errors": [], "warnings": [], "force_allowed": False}


def _stub(stub, activate_handler, active_handler=None):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/ingest/config(\?|$)", _ingest_config(POLICY))
    stub.on("GET", r"/ingest/policy(\?|$)", POLICY)
    stub.on("POST", r"/ingest/policy/preview(\?|$)", PREVIEW)
    stub.on("GET", r"/ingest_profiles(\?|$)", LIST)
    stub.on("GET", r"/ingest_profiles/active(\?|$)", active_handler or ACTIVE_PROFILE)
    stub.on("POST", r"/ingest_profiles/[^/]+/activate(\?|$)", activate_handler)


def _refusal(code: str, message: str, force_allowed: bool = False):
    return (
        422,
        {
            "detail": {
                "error": "validation_failed",
                "message": "the profile cannot be activated",
                "report": {
                    "ok": False,
                    "force_allowed": force_allowed,
                    "warnings": [],
                    "errors": [
                        {
                            "code": code,
                            "id": code,
                            "severity": "error",
                            "field": "primary_model",
                            "message": message,
                            "detail": {},
                            "bypassable": force_allowed,
                        }
                    ],
                },
            }
        },
    )


def _open(page, app_url):
    page.goto(f"{app_url}/p/default/settings/ingest-policy")
    page.get_by_test_id("ingest-profile-picker").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_lists_profiles_and_marks_the_active_one(stub, page, app_url):
    _stub(stub, lambda _r, _m: (500, {"detail": "unexpected"}))
    _open(page, app_url)
    rows = page.get_by_test_id("ingest-profile-row")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text("yolo-primary")
    expect(rows.nth(0)).to_contain_text("yolo11_small_trt")
    expect(rows.nth(0).get_by_test_id("ingest-profile-active-chip")).to_be_visible()
    expect(rows.nth(1).get_by_test_id("ingest-profile-activate")).to_be_visible()


def test_activate_posts_expected_active_and_refreshes_the_active_card(stub, page, app_url):
    state = {"active": ACTIVE_PROFILE}

    def activate(_req, _m):
        state["active"] = {
            **ACTIVE_PROFILE,
            "active": {"name": "yolo-large", "revision": 2},
            "previous": {"name": "yolo-primary", "revision": 3},
        }
        return {**state["active"], "validation": OK_REPORT}

    _stub(stub, activate, active_handler=lambda _r, _m: state["active"])
    _open(page, app_url)
    expect(page.get_by_test_id("active-ingest-profile-name")).to_contain_text("yolo-primary")
    page.get_by_test_id("ingest-profile-activate").click()
    expect(page.get_by_test_id("active-ingest-profile-name")).to_contain_text(
        "yolo-large", timeout=ACTION_TIMEOUT_MS
    )
    posts = [c for c in stub.calls if c[0] == "POST" and c[1].endswith("/yolo-large/activate")]
    assert len(posts) == 1
    assert posts[0][2] == {
        "revision": 2,
        "expected_active": {"name": "yolo-primary", "revision": 3},
        "force": False,
    }


@pytest.mark.parametrize(
    "code,message",
    [
        ("detector_model_not_ready", "yolo11_large_trt is not loaded in Triton"),
        ("detector_model_unusable", "yolo11_large_trt has an output shape ingest cannot read"),
        ("triton_unreachable", "Triton did not answer, so the model cannot be checked"),
    ],
)
def test_a_refusal_shows_the_served_issue_messages(stub, page, app_url, code, message):
    _stub(stub, lambda _r, _m: _refusal(code, message))
    _open(page, app_url)
    page.get_by_test_id("ingest-profile-activate").click()
    box = page.get_by_test_id("ingest-profile-refusal")
    expect(box).to_contain_text(message, timeout=ACTION_TIMEOUT_MS)
    expect(box).to_contain_text("the profile cannot be activated")
    expect(box).to_contain_text(code)
    expect(page.get_by_test_id("ingest-profile-force")).to_have_count(0)
    expect(page.get_by_test_id("active-ingest-profile-name")).to_contain_text("yolo-primary")
