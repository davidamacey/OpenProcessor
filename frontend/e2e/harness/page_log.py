"""Per-test page diagnostics: the default action timeout and, on a failed
test, a dump of the console, requests, stub state and DOM."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

from harness.constants import ACTION_TIMEOUT_MS, ROOT

FAILURE_DIR = ROOT / "artifacts_local" / "e2e-failures"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Stash each phase's report on the item so fixtures can see, at
    teardown, whether the test failed."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


class _PageLog:
    """Everything a failed test needs to explain itself: every console
    message and pageerror, and every request with its outcome and timing
    (relative to the page's creation)."""

    def __init__(self, page: Any) -> None:
        self.t0 = time.monotonic()
        self.console: list[str] = []
        self.requests: dict[Any, dict[str, Any]] = {}
        page.on("console", lambda m: self.console.append(f"{self._t()} {m.type}: {m.text}"))
        page.on("pageerror", lambda e: self.console.append(f"{self._t()} pageerror: {e}"))
        page.on("request", self._on_request)
        page.on("response", lambda r: self._done(r.request, f"{r.status}"))
        page.on("requestfailed", lambda r: self._done(r, f"FAILED {r.failure}"))
        page.on("requestfinished", lambda r: self._done(r, None, finished=True))

    def _t(self) -> str:
        return f"+{time.monotonic() - self.t0:7.3f}s"

    def _on_request(self, request: Any) -> None:
        self.requests[request] = {
            "start": self._t(),
            "line": f"{request.method} {request.url}",
            "status": "PENDING (never answered)",
            "finished": None,
        }

    def _done(self, request: Any, status: str | None, finished: bool = False) -> None:
        entry = self.requests.get(request)
        if entry is None:
            return
        if status is not None:
            entry["status"] = status
        if finished:
            entry["finished"] = self._t()

    def dump(self, page: Any, out: Path, stub: Any) -> None:
        out.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{e['start']} -> {e['finished'] or '(not finished)':>11} {e['status']:<28} {e['line']}"
            for e in self.requests.values()
        ]
        (out / "requests.txt").write_text("\n".join(lines) + "\n")
        (out / "console.txt").write_text("\n".join(self.console) + "\n")
        extra = [f"url: {page.url}"]
        if stub is not None:
            assets = getattr(stub, "app_assets", None)
            extra += [
                f"stub.unhandled: {stub.unhandled}",
                f"stub.handler_errors: {stub.handler_errors}",
                f"app asset fetch errors: {assets.errors if assets else []}",
                f"stub.handled ({len(stub.handled)}): {stub.handled}",
            ]
        (out / "summary.txt").write_text("\n".join(extra) + "\n")
        try:
            page.screenshot(path=str(out / "screenshot.png"), full_page=True, timeout=10000)
        except Exception as exc:  # noqa: BLE001 — diagnostics must never mask the real failure
            (out / "screenshot-error.txt").write_text(repr(exc))
        try:
            (out / "body.txt").write_text(page.evaluate("() => document.body?.innerText ?? ''"))
            (out / "dom.html").write_text(page.content())
        except Exception as exc:  # noqa: BLE001
            (out / "dom-error.txt").write_text(repr(exc))


@pytest.fixture
def page(page: Any, request: Any) -> Any:
    """Override pytest-playwright's own `page` fixture to apply the shared
    ACTION_TIMEOUT_MS budget as the page's default — every *implicit*
    timeout (a `.click()`/`.fill()`/etc. call with no explicit `timeout=`)
    gets the same generous, documented budget as the explicit
    `timeout=ACTION_TIMEOUT_MS` calls, instead of Playwright's own 30s
    default living as a second, undocumented number.

    On a failed test it also writes a screenshot, the console, every
    request with its status and timing, the stub's unhandled list and the
    DOM to `artifacts_local/e2e-failures/<test id>/`, so an intermittent
    failure explains itself instead of only reporting a timeout."""
    page.set_default_timeout(ACTION_TIMEOUT_MS)
    rate = float(os.environ.get("E2E_CPU_THROTTLE", "1"))
    if rate > 1:
        # Stress knob: a slow CI runner on demand (4 = the page runs 4x slower).
        cdp = page.context.new_cdp_session(page)
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": rate})
    log = _PageLog(page)
    yield page
    rep = getattr(request.node, "rep_call", None) or getattr(request.node, "rep_setup", None)
    failed = rep is not None and rep.failed
    if failed or getattr(request.node, "_e2e_teardown_failed", False):
        name = re.sub(r"[^\w.-]+", "_", request.node.nodeid)
        stub = getattr(request.node, "_e2e_stub", None)
        log.dump(page, FAILURE_DIR / name, stub)
