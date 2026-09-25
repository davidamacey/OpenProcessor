"""Fixtures for the "live read-only" tier — real browser, real deployment.

Unlike `e2e/stubbed/` (every `{API_PREFIX}` request answered by an
in-browser stub, never touches a real backend), this tier drives the
actual built app served by the deployment's nginx container
(`http://localhost:5184` by default, proxying `/curation` to a live
OpenProcessor backend) and checks that frontend and backend actually
agree with each other. See CLAUDE.md's "Live read-only tier" section
under "End-to-end tests (e2e/)".

Two hard safety properties, both enforced here, not by test discipline:

  1. **The whole tier is skipped unless `CROPWRIGHT_LIVE_URL` is set.**
     `npm run test:e2e` (CI, and the default local command) never points
     at `e2e/live` at all, but a human could still run
     `e2e/.venv/bin/pytest e2e/live` directly — the session-scoped
     autouse `_require_live_url` fixture below skips every test in that
     case, before any browser is ever launched.
  2. **Read-only, structurally.** The `guarded_page` fixture every test
     in this tier uses routes every `**/curation/**` request through a
     handler that lets GET/HEAD through untouched and `route.abort()`s
     anything else — a POST/PUT/PATCH/DELETE never reaches the real
     backend. Aborting instead of silently 200-ing is deliberate: a
     write attempt should break the UI flow that tried it (so a bug that
     makes the frontend try to write during a "read-only" flow is loud),
     not be swallowed. Every aborted attempt is recorded, and the
     fixture's teardown asserts the list is empty — a test that
     triggers a write, even one the guard blocked, fails. This is the
     property `test_write_guard_smoke.py` (see its own module docstring
     for the disposable-proof runbook) exists to demonstrate; it isn't
     part of the permanent suite.

Chromium launch args (`--disable-gpu`) are inherited from
`e2e/conftest.py`'s `browser_type_launch_args` fixture — this directory's
conftest is layered under that one, so no need to redeclare it here.
"""

from __future__ import annotations

import datetime
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

API_PREFIX = "/curation"

# e2e/live/conftest.py -> e2e/live -> e2e -> repo root.
ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_ROOT = ROOT / "artifacts_local" / "cw-live" / "live-tier"

# Requests to `**/curation/**` matching this method set pass through
# untouched; everything else is aborted and recorded. HEAD is included
# even though the app never issues it today — it's exactly as read-only
# as GET, and excluding it would be an arbitrary asymmetry.
_SAFE_METHODS = {"GET", "HEAD"}


def _env_live_url() -> str | None:
    url = os.environ.get("CROPWRIGHT_LIVE_URL")
    return url.rstrip("/") if url else None


@pytest.fixture(scope="session", autouse=True)
def _require_live_url() -> None:
    """Skip the entire tier, before any fixture that could touch a
    browser or network runs, unless CROPWRIGHT_LIVE_URL is set.

    Session-scoped + autouse: pytest instantiates session-scoped
    fixtures before function-scoped ones (`page`, `browser`, …)
    regardless of argument order, so `pytest.skip()` here fires before
    Playwright ever launches a browser.
    """
    if _env_live_url() is None:
        pytest.skip(
            "live tier skipped: set CROPWRIGHT_LIVE_URL (e.g. "
            "http://localhost:5184) to run it against a real deployment — "
            "see CLAUDE.md's 'Live read-only tier' section. Never set in "
            "CI or by `npm run test:e2e`."
        )


@pytest.fixture(scope="session")
def live_url(_require_live_url: None) -> str:
    """The deployment base URL, preflighted against `{API_PREFIX}/health`.

    Skips (doesn't error) on anything short of a clean 200 — an
    unreachable or unhealthy backend isn't this suite's bug to report.
    """
    url = _env_live_url()
    assert url is not None  # guaranteed by _require_live_url not skipping
    health_url = f"{url}{API_PREFIX}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=10) as resp:
            status = resp.status
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"live tier skipped: preflight GET {health_url} failed: {exc}")
        return ""  # unreachable, keeps type-checkers happy
    if status != 200:
        pytest.skip(f"live tier skipped: preflight GET {health_url} returned {status}, not 200")
    return url


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict[str, Any]) -> dict[str, Any]:
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
    }


@pytest.fixture(scope="session")
def screenshot_run_dir() -> Path:
    """One timestamped directory per test-session run
    (`artifacts_local/cw-live/live-tier/<run-timestamp>/`), holding every
    route's full-page screenshots at both the desktop (1600x1000) and
    narrow (800x1000) viewports — captured unconditionally by
    `test_route_sweep.py`, not only on a failure.

    These are NOT a substitute for a human looking at them: per CLAUDE.md's
    "Live read-only tier" section, someone must actually open a sample of
    the saved PNGs after each run — the sweep only proves a route mounted
    without erroring, never that it looks right.
    """
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    d = SCREENSHOT_ROOT / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def api_get(live_url: str, path: str) -> Any:
    """A plain GET against the live backend, outside the browser — used
    by data-agreement tests to fetch the "expected" side of a comparison.
    Never anything but GET: this helper doesn't even have a way to send
    a body or a non-GET method, matching this tier's read-only contract.
    """
    import json

    req = urllib.request.Request(f"{live_url}{API_PREFIX}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


class GuardedPage:
    """Thin wrapper around a Playwright `Page` carrying the write-guard's
    recorded state, so tests can assert on it without reaching into
    fixture internals."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.write_attempts: list[tuple[str, str]] = []
        self.page_errors: list[str] = []
        self.bad_responses: list[tuple[str, int]] = []


@pytest.fixture
def guarded_page(live_url: str, page: Any) -> Any:
    """A `Page` wired with this tier's two structural guarantees:

      * every `**/curation/**` request is GET/HEAD-only (anything else
        is aborted, recorded, and fails the test at teardown);
      * any `pageerror` fails the test at teardown.

    Also sets explicit navigation/action timeouts (never "load" waits or
    fixed sleeps — callers use `wait_until="domcontentloaded"` plus a
    `wait_for_selector` for a concrete element).
    """
    page.set_default_timeout(15_000)
    page.set_default_navigation_timeout(20_000)

    gp = GuardedPage(page)

    def _guard_curation(route: Any, request: Any) -> None:
        if request.method not in _SAFE_METHODS:
            gp.write_attempts.append((request.method, request.url))
            route.abort("failed")
            return
        route.continue_()

    page.route(f"**{API_PREFIX}/**", _guard_curation)
    page.on("pageerror", lambda exc: gp.page_errors.append(str(exc)))

    # Exposed for tests that want it directly (`guarded_page.page`,
    # `.write_attempts`, `.page_errors`) as well as every place that just
    # wants a Playwright Page (`guarded_page.page.goto(...)`).
    yield gp

    assert gp.write_attempts == [], (
        "live tier is read-only: a non-GET/HEAD request to `**/curation/**` "
        f"was attempted (and aborted before reaching the backend), but must "
        f"never be attempted at all: {gp.write_attempts}"
    )
    assert gp.page_errors == [], f"page error(s) during test: {gp.page_errors}"


def route_url(live_url: str, path: str) -> str:
    return f"{live_url}{path}"


ALLOWED_4XX_5XX = {
    # (method, path-regex): documented reason a non-2xx response here is
    # expected UI behavior, not a bug — keep this list short and
    # explicit; a route sweep failure should default to "real bug", not
    # "add it to the allow-list".
}


def is_allowlisted_bad_response(method: str, path: str, status: int) -> bool:
    for (m, pattern), _reason in ALLOWED_4XX_5XX.items():
        if m == method and re.search(pattern, path):
            return True
    return False


def int_field(payload: dict, *keys: str) -> int:
    """Walk a nested dict by key path and return an int, e.g.
    `int_field(stats, "clusters", "cluster_count")`."""
    node: Any = payload
    for k in keys:
        node = node[k]
    return int(node)


def wait_for_stable_text(page: Any, selector: str, settle_ms: int = 800, timeout_ms: int = 20_000) -> None:
    """Wait until `selector`'s text content hasn't changed for
    `settle_ms`. Some values (e.g. the `/review` queue counter, when a
    URL-seeded `?region_status=` filter arrives before `GET
    {API_PREFIX}/review/tabs`'s `filter_specs` has loaded) render an
    intermediate unfiltered value before the real one lands — see
    CLAUDE.md's "Served per-tab filters" section. Polling for text
    stability (rather than hardcoding "wait for exactly N") makes tests
    robust to that transition and to a legitimate later value.
    """
    page.wait_for_function(
        """
        (args) => {
          const [selector, settleMs] = args;
          const el = document.querySelector(selector);
          if (!el) return false;
          const now = Date.now();
          const text = el.textContent;
          window.__stableTextSince = window.__stableTextSince || {};
          const rec = window.__stableTextSince[selector];
          if (!rec || rec.text !== text) {
            window.__stableTextSince[selector] = { text, since: now };
            return false;
          }
          return now - rec.since >= settleMs;
        }
        """,
        arg=[selector, settle_ms],
        timeout=timeout_ms,
        polling=100,
    )


def agrees_with_retry(live_url: str, path: str, keys: tuple, displayed: int) -> tuple[bool, int, int]:
    """A live deployment can be mutated by another actor mid-test (a
    concurrent labeling smoke test is explicitly expected — see
    CLAUDE.md). Rather than a flaky hard-equal assert, fetch the
    expected value, compare; on mismatch, re-fetch once (the value may
    have moved between our first API read and the UI read) and accept
    either. Returns (ok, first_expected, second_expected).
    """
    first = int_field(api_get(live_url, path), *keys)
    if displayed == first:
        return True, first, first
    second = int_field(api_get(live_url, path), *keys)
    return displayed == second, first, second
