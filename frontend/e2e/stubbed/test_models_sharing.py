"""/models cross-project model sharing (OpenProcessor projects P2, §5.5)
against a stateful stub: `GET {prefix}/models/status?include_other_projects=true`
reflects every `PUT {prefix}/models/{name}/sharing`.

- the owner-only toggle round trip: confirm dialog, the PUT body carries
  the served sharing revision, the list reloads with the served state;
- a 409 `revision_conflict` shows the served message, reloads, and the
  retry carries the fresh served revision;
- another project's shared model shows its project chip and the served
  class mapping (count + unmapped names), with no toggle.

Waits are real (`expect_request` / visible elements), never a sleep.
"""

from __future__ import annotations

import re
from typing import Any

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import default_project, project, projects_response

API_PREFIX = "/curation"
PREFIX = f"{API_PREFIX}/projects/default"


def base(name: str, **over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "friendly_name": f"{name} (promoted)",
        "role": "Promoted via /curation/train/promote",
        "kind": "triton",
        "model_type": "Promoted checkpoint",
        "status": "ready",
        "version": "1",
        "inference_count": 0,
        "exec_count": 0,
        "inference_failed": 0,
        "avg_latency_ms": None,
        "last_error": None,
        "endpoint": "http://triton-server:8000",
        "is_region_protected": False,
        "requires_force_to_unload": False,
        "job_id": "2026-09-27T01-00-00_yolo26n",
        "promoted_at": "2026-09-27T02:00:00Z",
        "unloadable": True,
        "optional": False,
        "project": None,
        "owned": False,
        "sharing_revision": None,
        "shared": False,
        "class_mapping": None,
    }
    out.update(over)
    return out


class Models:
    def __init__(self) -> None:
        self.own = base(
            "widget_det",
            project="default",
            owned=True,
            shared=False,
            sharing_revision=3,
            class_mapping={"mapped_count": 4, "unmapped": []},
        )
        self.foreign = base(
            "beta__crate_det",
            friendly_name="beta__crate_det (shared by beta)",
            project="beta",
            owned=False,
            shared=True,
            unloadable=False,
            class_mapping={"mapped_count": 2, "unmapped": ["van", "bus"]},
        )
        self.puts: list[tuple[str, Any]] = []
        self.status_queries: list[str] = []
        self.conflict_next = False
        self.in_use_projects: list[str] = []

    def install(self, stub: Any) -> None:
        base_re = re.escape(PREFIX)
        stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
        stub.on(
            "GET",
            rf"^{re.escape(API_PREFIX)}/projects$",
            projects_response(
                API_PREFIX,
                [default_project(API_PREFIX), project(API_PREFIX, "beta", display_name="Beta lot")],
            ),
        )
        stub.on("GET", rf"^{base_re}/models/status$", self.status)
        stub.on("PUT", rf"^{base_re}/models/([^/]+)/sharing$", self.put)

    def status(self, request: Any, _m: Any) -> Any:
        self.status_queries.append(request.url.split("?", 1)[-1])
        return (200, {"models": [dict(self.own), dict(self.foreign)]})

    def put(self, request: Any, m: Any) -> Any:
        body = request.post_data_json
        self.puts.append((m.group(1), body))
        forced = "force=true" in request.url
        if self.in_use_projects and not body["shared"] and not forced:
            return (409, {"detail": {
                "error": "in_use",
                "message": f"'widget_det' is still used by {len(self.in_use_projects)} other project(s)",
                "projects": self.in_use_projects,
            }})
        if self.conflict_next:
            self.conflict_next = False
            self.own["sharing_revision"] += 2  # someone else toggled twice
            return (409, {"detail": {
                "error": "revision_conflict",
                "message": f"'widget_det' sharing revision is {self.own['sharing_revision']}, "
                f"not {body['expected_revision']}",
                "current_revision": self.own["sharing_revision"],
            }})
        self.own["shared"] = body["shared"]
        self.own["sharing_revision"] += 1
        return (200, {
            "name": "widget_det",
            "project": "default",
            "shared": body["shared"],
            "revision": self.own["sharing_revision"],
            "used_by": [{"project": p, "profile": "tag"} for p in self.in_use_projects]
            if forced
            else [],
        })


def open_models(page: Any, app_url: str) -> None:
    page.goto(f"{app_url}/p/default/models")
    page.get_by_test_id("model-sharing-widget_det").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_share_toggle_round_trip_sends_the_served_revision(stub, page, app_url):
    reg = Models()
    reg.install(stub)
    open_models(page, app_url)
    assert "include_other_projects=true" in reg.status_queries[0]

    state = page.get_by_test_id("model-sharing-widget_det").get_by_test_id("model-sharing-state")
    assert state.inner_text() == "Not shared"
    page.get_by_test_id("model-share-toggle-widget_det").click()
    page.get_by_test_id("share-model-dialog").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert reg.puts == []  # confirm-gated

    with page.expect_request(lambda r: r.method == "PUT" and r.url.endswith("/sharing")):
        page.get_by_test_id("share-model-confirm").click()
    page.get_by_test_id("share-model-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"model-sharing-widget_det\"]')"
        "?.textContent.includes('Shared with other projects')",
        timeout=ACTION_TIMEOUT_MS,
    )
    assert reg.puts == [("widget_det", {"shared": True, "expected_revision": 3})]

    # And back: unsharing names the server-side in-use check, never "safe".
    page.get_by_test_id("model-share-toggle-widget_det").click()
    text = page.get_by_test_id("share-model-text").inner_text()
    assert "active detection profile" in text, text
    assert "may be using it" not in text, text
    assert "safe" not in text.lower(), text
    with page.expect_request(lambda r: r.method == "PUT" and r.url.endswith("/sharing")):
        page.get_by_test_id("share-model-confirm").click()
    page.get_by_test_id("share-model-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert reg.puts[-1] == ("widget_det", {"shared": False, "expected_revision": 4})


def test_revision_conflict_reloads_and_retries_with_the_fresh_revision(stub, page, app_url):
    reg = Models()
    reg.conflict_next = True
    reg.install(stub)
    open_models(page, app_url)

    page.get_by_test_id("model-share-toggle-widget_det").click()
    with page.expect_request(
        lambda r: r.method == "GET" and "/models/status" in r.url
    ):
        page.get_by_test_id("share-model-confirm").click()
    err = page.get_by_test_id("share-model-error")
    err.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "'widget_det' sharing revision is 5, not 3" in err.inner_text()
    page.get_by_test_id("share-model-reloaded").wait_for(timeout=ACTION_TIMEOUT_MS)

    page.get_by_test_id("share-model-confirm").click()
    page.get_by_test_id("share-model-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert [b for _n, b in reg.puts] == [
        {"shared": True, "expected_revision": 3},
        {"shared": True, "expected_revision": 5},
    ]


def test_another_projects_shared_model_shows_its_chip_and_mapping(stub, page, app_url):
    Models().install(stub)
    open_models(page, app_url)

    card = page.get_by_test_id("model-sharing-beta__crate_det")
    assert card.get_by_test_id("model-project-chip").inner_text() == "from Beta lot"
    assert card.get_by_test_id("model-mapped-count").inner_text() == "2 classes map"
    card.get_by_text("2 not in this project").click()
    assert card.get_by_test_id("model-unmapped-names").inner_text() == "van, bus"
    assert page.get_by_test_id("model-share-toggle-beta__crate_det").count() == 0


def test_unshare_in_use_shows_served_projects_and_confirm_gates_force(stub, page, app_url):
    reg = Models()
    reg.own["shared"] = True
    reg.in_use_projects = ["beta"]
    reg.install(stub)
    open_models(page, app_url)

    page.get_by_test_id("model-share-toggle-widget_det").click()
    page.get_by_test_id("share-model-confirm").click()
    page.get_by_test_id("share-model-in-use").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "beta" in page.get_by_test_id("share-model-in-use").inner_text()
    assert "still used by 1 other project(s)" in page.get_by_test_id("share-model-error").inner_text()

    page.get_by_test_id("share-model-force").click()
    assert "beta" in page.get_by_test_id("share-model-force-warning").inner_text()
    assert len(reg.puts) == 1  # arming the override sends nothing
    with page.expect_request(
        lambda r: r.method == "PUT" and r.url.endswith("/sharing?force=true")
    ):
        page.get_by_test_id("share-model-force-confirm").click()
    page.get_by_test_id("share-model-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert len(reg.puts) == 2
