"""`/projects` management against a stateful stub of the P3 lifecycle API
(OpenProcessor `cutover/projects-lifecycle`): create, edit (with a 409
`revision_conflict` → reload → save), archive and the archived toggle,
and the guarded delete (dry-run blocking, then a confirmed delete). Every
write's request body is asserted, and every refusal renders the served
message verbatim.

Waits are real (`expect_request` / visible elements), never a sleep.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import project, projects_response

API_PREFIX = "/curation"


class Projects:
    """The served project registry: GET reflects every write."""

    def __init__(self, capacity_status: str = "ok") -> None:
        self.rows: dict[str, dict[str, Any]] = {
            "default": project(API_PREFIX, "default", is_default=True, deletable=False),
            "alpha": project(API_PREFIX, "alpha", display_name="Alpha", revision=4),
        }
        self.capacity_status = capacity_status
        self.writes: list[tuple[str, str, Any]] = []
        self.conflict_next_patch = False
        self.dry_run_blocking: list[dict[str, str]] = []

    def listed(self, include_archived: bool) -> list[dict[str, Any]]:
        return [
            p for p in self.rows.values()
            if p["status"] != "deleted" and (include_archived or p["status"] != "archived")
        ]

    def install(self, stub: Any) -> None:
        base = re.escape(API_PREFIX)
        stub.on("GET", rf"^{base}/projects$", self.get_list)
        stub.on("POST", rf"^{base}/projects$", self.create)
        stub.on("PATCH", rf"^{base}/projects/([^/]+)$", self.patch)
        stub.on("POST", rf"^{base}/projects/([^/]+)/(archive|unarchive)$", self.archive)
        stub.on("DELETE", rf"^{base}/projects/([^/]+)$", self.delete)

    def get_list(self, request: Any, _m: Any) -> Any:
        qs = parse_qs(urlparse(request.url).query)
        include = qs.get("include_archived", ["false"])[0] == "true"
        res = projects_response(API_PREFIX, self.listed(include), self.capacity_status)
        res["include_archived"] = include
        return (200, res)

    def create(self, request: Any, _m: Any) -> Any:
        body = request.post_data_json
        self.writes.append(("POST", "/projects", body))
        if body["slug"] in self.rows:
            return (409, {"detail": {"error": "slug_taken", "message": f"a project named '{body['slug']}' already exists"}})
        row = project(API_PREFIX, body["slug"], display_name=body["display_name"], description=body.get("description", ""))
        self.rows[row["slug"]] = row
        warnings = (
            [{"code": "shard_budget_high", "message": "Served warning: near the recommended shard budget."}]
            if self.capacity_status == "warn"
            else []
        )
        return (201, {"project": row, "warnings": warnings})

    def patch(self, request: Any, m: Any) -> Any:
        slug = m.group(1)
        body = request.post_data_json
        self.writes.append(("PATCH", slug, body))
        row = self.rows[slug]
        if self.conflict_next_patch:
            self.conflict_next_patch = False
            row["revision"] += 2  # someone else saved twice meanwhile
            return (409, {"detail": {
                "error": "revision_conflict",
                "message": f"expected revision {body['expected_revision']}, current is {row['revision']}",
                "current_revision": row["revision"],
            }})
        for k in ("display_name", "description"):
            if body.get(k) is not None:
                row[k] = body[k]
        row["revision"] += 1
        return (200, {"project": row, "warnings": []})

    def archive(self, request: Any, m: Any) -> Any:
        slug, verb = m.group(1), m.group(2)
        self.writes.append(("POST", f"{slug}/{verb}", request.post_data_json))
        row = self.rows[slug]
        archived = verb == "archive"
        row.update(
            status="archived" if archived else "active",
            writable=not archived,
            archivable=not archived,
            unarchivable=archived,
            revision=row["revision"] + 1,
        )
        return (200, {"project": row, "warnings": []})

    def delete(self, request: Any, m: Any) -> Any:
        slug = m.group(1)
        qs = parse_qs(urlparse(request.url).query)
        if qs.get("dry_run") == ["true"]:
            self.writes.append(("DELETE", f"{slug}?dry_run", None))
            return (200, {
                "indexes": [{"name": f"op_{slug}_items", "docs": 20, "store_bytes": None}],
                "dirs": [{"path": f"/data/{slug}", "bytes": 2048}],
                "promoted_models": [],
                "mlflow_experiment": slug,
                "running_jobs": [],
                "referenced_by": [],
                "blocking": [b["code"] for b in self.dry_run_blocking],
                "blocking_detail": self.dry_run_blocking,
            })
        confirm = qs.get("confirm", [""])[0]
        self.writes.append(("DELETE", f"{slug}?confirm={confirm}", None))
        if confirm != slug:
            return (422, {"detail": {"error": "confirm_mismatch", "message": f"confirm must equal the slug '{slug}'"}})
        row = self.rows[slug]
        row.update(status="deleting", writable=False, selectable=False, archivable=False, unarchivable=False)
        return (202, {"project": row, "warnings": []})


def open_projects(page: Any, app_url: str) -> None:
    page.goto(f"{app_url}/projects")
    page.get_by_test_id("project-row-default").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_create_sends_the_form_and_toasts_served_warnings(stub, page, app_url):
    reg = Projects(capacity_status="warn")
    reg.install(stub)
    open_projects(page, app_url)

    # Served capacity renders as-is; default can never be deleted.
    assert "Served capacity message (warn)." in page.get_by_test_id("projects-capacity").inner_text()
    assert page.get_by_test_id("project-delete-default").count() == 0
    assert page.get_by_test_id("project-delete-alpha").count() == 1

    page.get_by_test_id("projects-create").click()
    page.get_by_test_id("create-project-slug").fill("gamma")
    page.get_by_test_id("create-project-name").fill("Gamma")
    page.get_by_test_id("create-project-description").fill("third one")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/curation/projects")):
        page.get_by_test_id("create-project-submit").click()
    page.get_by_test_id("project-row-gamma").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_text("Served warning: near the recommended shard budget.").first.wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    assert reg.writes == [
        ("POST", "/projects", {"slug": "gamma", "display_name": "Gamma", "description": "third one"})
    ]


def test_create_renders_a_served_refusal_verbatim(stub, page, app_url):
    reg = Projects()
    reg.install(stub)
    open_projects(page, app_url)
    page.get_by_test_id("projects-create").click()
    page.get_by_test_id("create-project-slug").fill("alpha")
    page.get_by_test_id("create-project-name").fill("Another alpha")
    page.get_by_test_id("create-project-submit").click()
    err = page.get_by_test_id("create-project-error")
    err.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert err.inner_text() == "a project named 'alpha' already exists"


def test_blocked_capacity_disables_create(stub, page, app_url):
    Projects(capacity_status="blocked").install(stub)
    open_projects(page, app_url)
    assert page.get_by_test_id("projects-create").is_disabled()
    assert "No room for another project" in page.get_by_test_id("projects-capacity").inner_text()


def test_edit_revision_conflict_offers_a_reload_then_saves(stub, page, app_url):
    reg = Projects()
    reg.conflict_next_patch = True
    reg.install(stub)
    open_projects(page, app_url)

    page.get_by_test_id("project-edit-alpha").click()
    page.get_by_test_id("edit-project-name").fill("Alpha renamed")
    page.get_by_test_id("edit-project-submit").click()
    err = page.get_by_test_id("edit-project-error")
    err.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "expected revision 4, current is 6" in err.inner_text()

    with page.expect_request(lambda r: r.method == "GET" and "/curation/projects" in r.url):
        page.get_by_test_id("edit-project-reload").click()
    page.get_by_test_id("edit-project-reload").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("edit-project-submit").click()
    page.get_by_test_id("edit-project-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    page.get_by_text("Alpha renamed").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert reg.writes == [
        ("PATCH", "alpha", {"display_name": "Alpha renamed", "expected_revision": 4}),
        ("PATCH", "alpha", {"display_name": "Alpha renamed", "expected_revision": 6}),
    ]


def test_archive_then_show_archived_then_unarchive(stub, page, app_url):
    reg = Projects()
    reg.install(stub)
    open_projects(page, app_url)

    page.get_by_test_id("project-archive-alpha").click()
    # The served list hides archived projects unless asked.
    page.get_by_test_id("project-row-alpha").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    with page.expect_request(lambda r: "include_archived=true" in r.url):
        page.get_by_test_id("projects-include-archived").check()
    page.get_by_test_id("project-unarchive-alpha").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-status-alpha").inner_text() == "Archived"
    assert page.get_by_test_id("project-archive-alpha").count() == 0

    page.get_by_test_id("project-unarchive-alpha").click()
    page.get_by_test_id("project-archive-alpha").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert reg.writes == [
        ("POST", "alpha/archive", {"expected_revision": 4}),
        ("POST", "alpha/unarchive", {"expected_revision": 5}),
    ]


def test_delete_blocked_by_the_dry_run_offers_no_confirm(stub, page, app_url):
    reg = Projects()
    reg.dry_run_blocking = [{"code": "project_busy", "message": "1 job(s) still running"}]
    reg.install(stub)
    open_projects(page, app_url)

    page.get_by_test_id("project-delete-alpha").click()
    blocking = page.get_by_test_id("delete-project-blocking")
    blocking.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "1 job(s) still running" in blocking.inner_text()
    assert page.get_by_test_id("delete-project-confirm").count() == 0
    assert page.get_by_test_id("delete-project-submit").is_disabled()
    assert reg.writes == [("DELETE", "alpha?dry_run", None)]


def test_delete_confirms_with_the_typed_slug(stub, page, app_url):
    reg = Projects()
    reg.install(stub)
    open_projects(page, app_url)

    page.get_by_test_id("project-delete-alpha").click()
    page.get_by_test_id("delete-project-confirm").fill("alpah")
    page.get_by_test_id("delete-project-submit").click()
    err = page.get_by_test_id("delete-project-error")
    err.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert err.inner_text() == "confirm must equal the slug 'alpha'"

    page.get_by_test_id("delete-project-confirm").fill("alpha")
    page.get_by_test_id("delete-project-submit").click()
    page.get_by_test_id("delete-project-dialog").wait_for(state="detached", timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("project-status-alpha").filter(has_text="Deleting").wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    assert reg.writes == [
        ("DELETE", "alpha?dry_run", None),
        ("DELETE", "alpha?confirm=alpah", None),
        ("DELETE", "alpha?confirm=alpha", None),
    ]
    assert json.dumps(reg.rows["alpha"]["status"]) == '"deleting"'
