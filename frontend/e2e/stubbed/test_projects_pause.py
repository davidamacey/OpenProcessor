"""Per-project pipeline pause (OpenProcessor projects P2, §5.1) against a
stateful stub: `GET|POST {prefix}/pause` and `POST {prefix}/resume`,
each addressed through the row's own served `prefix`.

- /projects: the served `paused` on each row drives the chip; Pause is
  confirm-gated, POSTs to that row's own prefix, and the "paused" chip
  appears; Resume clears it.
- the switcher shows a "paused" chip for a paused active project.

Waits are real (`expect_request` / visible elements), never a sleep.
"""

from __future__ import annotations

import re
from typing import Any

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import default_project, project, projects_response

API_PREFIX = "/curation"


class Pause:
    def __init__(self, **paused: bool) -> None:
        self.paused: dict[str, bool] = dict(paused)
        self.reads: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def listing(self, request: Any, _m: Any) -> Any:
        rows = [
            default_project(API_PREFIX),
            project(API_PREFIX, "alpha", display_name="Alpha"),
            project(
                API_PREFIX,
                "wip",
                status="building",
                writable=False,
                selectable=False,
            ),
        ]
        for r in rows:
            r["paused"] = self.paused.get(r["slug"], False)
        return (200, projects_response(API_PREFIX, rows))

    def install(self, stub: Any) -> None:
        stub.on("GET", rf"^{re.escape(API_PREFIX)}/projects$", self.listing)
        stub.on("GET", r"/projects/([^/]+)/pause$", self.read)
        stub.on("POST", r"/projects/([^/]+)/(pause|resume)$", self.write)

    def read(self, request: Any, m: Any) -> Any:
        slug = m.group(1)
        self.reads.append(slug)
        on = self.paused.get(slug, False)
        return (
            200,
            {
                "project": slug,
                "paused": on,
                "paused_by": ["project"] if on else [],
                "reason": None,
            },
        )

    def write(self, request: Any, m: Any) -> Any:
        slug, verb = m.group(1), m.group(2)
        self.writes.append((slug, verb))
        self.paused[slug] = verb == "pause"
        return (
            200,
            {
                "project": slug,
                "paused": self.paused[slug],
                "paused_by": ["project"] if self.paused[slug] else [],
                "reason": None,
            },
        )


def test_pause_then_resume_a_project_from_the_projects_page(stub, page, app_url):
    reg = Pause()
    reg.install(stub)
    page.goto(f"{app_url}/projects")
    page.get_by_test_id("project-pause-alpha").wait_for(timeout=ACTION_TIMEOUT_MS)
    # Only selectable rows are read; the building project never is.
    assert "wip" not in reg.reads
    assert page.get_by_test_id("project-paused-alpha").count() == 0

    page.get_by_test_id("project-pause-alpha").click()
    page.get_by_test_id("pause-project-dialog").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert reg.writes == []  # confirm-gated
    with page.expect_request(
        lambda r: r.method == "POST" and r.url.endswith(f"{API_PREFIX}/projects/alpha/pause")
    ):
        page.get_by_test_id("pause-project-confirm").click()
    page.get_by_test_id("project-paused-alpha").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-paused-default").count() == 0

    page.get_by_test_id("project-resume-alpha").click()
    with page.expect_request(
        lambda r: r.method == "POST" and r.url.endswith(f"{API_PREFIX}/projects/alpha/resume")
    ):
        page.get_by_test_id("pause-project-confirm").click()
    page.get_by_test_id("project-paused-alpha").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    assert reg.writes == [("alpha", "pause"), ("alpha", "resume")]


def test_switcher_shows_paused_for_a_paused_active_project(stub, page, app_url):
    Pause(alpha=True).install(stub)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", {"models": []})
    page.goto(f"{app_url}/p/alpha/models")
    page.get_by_test_id("project-switcher-paused").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-switcher-current").inner_text() == "Alpha"
