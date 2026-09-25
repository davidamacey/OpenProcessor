"""Shared fixtures for the stubbed-backend Playwright/pytest e2e suite.

See docs/design/test-audit-2026-09-24.md recommendation 5 (P1-1) and
CLAUDE.md's "Development" section.

Fail-closed contract (the whole point of this file — see the R2 finding
in the audit doc about the old scripts/playwright_*.py stubs failing
open):

  * Every request under the configured API prefix (default ``/curation``)
    is routed through ``Stub``. A path/method a test did not register via
    ``stub.on(...)`` gets a ``501`` and is recorded in ``stub.unhandled`` —
    never a silent ``200 {}``.
  * Any request to the retired ``/curation/`` prefix is also intercepted,
    recorded in ``stub.op_hits``, and answered with ``501`` — it must
    never reach a real network.
  * At teardown, the ``stub`` fixture asserts ``unhandled == []`` and
    ``op_hits == []`` and that at least one request was actually
    handled — a prefix/route mismatch fails the test instead of passing
    on an empty page.

The app under test is a real production build: ``npm run build`` once
per session, then ``vite preview`` serves the static output. The stubs
intercept in the browser, so a static build is enough — no dev server,
no SSR.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from fixtures.wire import REGION_PROFILE, REGION_TAB_LABEL

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict[str, Any]) -> dict[str, Any]:
    """Force software rendering off entirely.

    On this machine, headless chromium's default GPU-process init
    (`--use-gl=disabled`'s ANGLE/EGL path) crash-loops the GPU process
    repeatedly, pinning the renderer at ~100% CPU for a couple of minutes
    before `Browser.new_page()` itself raises `TargetClosedError` —
    `--disable-gpu` on top of pytest-playwright's own `--headless=new`
    sidesteps the crash loop entirely (verified: `new_page()` returns in
    under a second with it, hangs for minutes without it).
    """
    return {**browser_type_launch_args, "args": ["--disable-gpu"]}
API_PREFIX = "/curation"

# 1x1 transparent GIF — grids only need the <img> to resolve, not real pixels.
TRANSPARENT_GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)

HandlerResult = Any  # dict/list (-> 200 json) | tuple[int, Any] | tuple[int, Any, str]
Handler = Callable[[Any, "re.Match[str]"], HandlerResult]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"vite preview exited early (code {proc.returncode}):\n{out}")
        try:
            urllib.request.urlopen(base_url, timeout=1)
            return
        except urllib.error.URLError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"vite preview did not come up within {timeout}s at {base_url}")


@pytest.fixture(scope="session")
def app_url() -> Any:
    """Build the app once, serve it via `vite preview`, tear down at session end.

    Machine is heavily loaded (see CLAUDE.md) — build gets a generous timeout
    and preview startup is polled, never a fixed sleep.

    Set `E2E_APP_URL` to point at an already-running `vite preview` (or any
    static server of a build you built yourself) and skip both the build
    and the server lifecycle — a fast inner loop for local iteration; not
    used in CI.
    """
    import os

    preset = os.environ.get("E2E_APP_URL")
    if preset:
        yield preset.rstrip("/")
        return

    build = subprocess.run(
        ["npm", "run", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if build.returncode != 0:
        raise RuntimeError(f"npm run build failed:\n{build.stdout}\n{build.stderr}")

    port = _free_port()
    proc = subprocess.Popen(
        ["npx", "vite", "preview", "--port", str(port), "--strictPort"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://localhost:{port}"
    try:
        _wait_for_server(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


class Stub:
    """Fail-closed router for `{api_prefix}/**` and the retired `/curation/**`.

    Tests register handlers with `.on(method, path_regex, handler_or_body)`.
    `path_regex` is matched (via `re.search`) against the URL path with the
    query string stripped. The most-recently-registered matching handler
    wins, so a test can override a default (e.g. the built-in image/health
    handlers) by registering a more specific or later pattern.
    """

    def __init__(self, page: Any, api_prefix: str = API_PREFIX) -> None:
        self.page = page
        self.api_prefix = api_prefix
        self._handlers: list[tuple[str, re.Pattern[str], Handler | HandlerResult]] = []
        self.handled: list[tuple[str, str]] = []
        self.unhandled: list[tuple[str, str]] = []
        self.op_hits: list[tuple[str, str]] = []
        # Every mutating (non-GET) call, in order: (method, path, parsed_json_body_or_None).
        self.calls: list[tuple[str, str, Any]] = []
        # A handler (default or test-registered) raising is itself a bug in
        # the stub, not the app under test — recorded and asserted on so it
        # fails the test loudly instead of the route just hanging.
        self.handler_errors: list[str] = []

        # Defaults every test needs; tests may override with a later `.on(...)`
        # (dispatch checks the most-recently-registered handler first).
        # The four below are fetched by the root layout on EVERY route, not
        # just the page under test — every test would otherwise have to
        # know and stub them individually.
        # The served region profile gates every region feature; the
        # default deployment has one (the neutral widget/tag domain).
        # e2e/stubbed/test_no_region_profile.py overrides it with None.
        self.on("GET", r"/health$", {"status": "ok", "region_profile": REGION_PROFILE})
        self.on("GET", r"(thumbnail|region_thumbnail|/source)(/|$|\?)", self._image)
        # K6 (docs/design/k6-frontend-overlay-plan-2026-09-24.md):
        # `/review`'s source panel (SourceImageOverlay.svelte) fetches
        # `/crops/{id}/context` and then the now-unannotated
        # `/crops/{id}/image` on every visit — same "every test needs
        # this" rationale as the four defaults above. A test that cares
        # about the overlay's actual boxes registers its own, more
        # specific `.on(...)` for `/context` (later registration wins).
        self.on("GET", r"/crops/[^/]+/context$", self._context)
        self.on("GET", r"/crops/[^/]+/image(/|$|\?)", self._image)
        self.on("GET", r"(/events|/stream)(/|$|\?)", (204, "", "text/plain"))
        self.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
        # S1 fix (visual audit 2026-09-24): `/review` now reads the
        # deployment's pinned curation defaults (`curationSettingsStore`)
        # so StrategyBar can flag a zero-coverage pinned sort — fired on
        # every `/review` mount, same "every test needs this default"
        # rationale as `/methods` above. `getCurationSettings` (api.ts)
        # requests the UNPREFIXED `/settings` path, not `/curation/settings`
        # — matches `test_curation_settings.py`'s own stub pattern
        # (`r"/settings(\?|$)"`), which is deliberately looser than most
        # patterns here to also match that bare path under `{api_prefix}`.
        # `test_curation_settings.py` overrides this per-test.
        self.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None})
        self.on("GET", r"/class_sources(\?|$)", {"class_sources": []})
        self.on(
            "GET",
            r"/regions/statuses(\?|$)",
            {
                "statuses": [],
                "confirm_status": "detected",
                "reject_status": "no_region_visible",
                "false_positive_status": "false_positive",
            },
        )
        # W0 naming-sweep finding m9: the deployment-configured detector/
        # segmenter/verifier vocabulary. Shapes lifted from the vendored
        # OpenAPI description (contracts/openprocessor/openapi/curation.json,
        # `/curation/regions/vocabulary`) — `{detectors, region_sources,
        # chain_actors}`, each a list of `{id, label, role, filterable?}`.
        # Defaults are the neutral fixture domain's detector ids (a tag
        # detector and segmenter, audit §4.4) so the region gallery's
        # detector filter and provenance chips render sensibly without
        # every test having to stub the endpoint itself.
        self.on(
            "GET",
            r"/regions/vocabulary(\?|$)",
            {
                "detectors": [
                    {
                        "id": "tag_detector_v1",
                        "label": "Tag detector",
                        "role": "detector",
                        "filterable": True,
                    },
                    {
                        "id": "tag_segmenter",
                        "label": "Tag segmenter",
                        "role": "segmenter",
                        "filterable": True,
                    },
                    {"id": "human", "label": "Human", "role": "human", "filterable": True},
                ],
                "region_sources": [
                    {"id": "detector", "label": "Detector", "role": "detector"},
                    {"id": "segmenter", "label": "Segmenter", "role": "segmenter"},
                ],
                "chain_actors": [
                    {"id": "tag_verifier", "label": "Tag verifier", "role": "verifier"},
                ],
                # openprocessor fix #29 / 840beb8 adoption: labeled
                # `region_rejection_reason` vocabulary. Empty by default —
                # unlike detectors/chain_actors above, no shared default
                # data is needed for most tests (the rejection-styled
                # candidate badge/Reason row is exercised by dedicated
                # tests that stub their own entries), and an empty list
                # keeps the pre-existing wording ("rejected candidate ·
                # confirm to accept") unchanged for every test that
                # doesn't care about kind-based styling.
                "rejection_reasons": [],
                "region_profile": REGION_PROFILE,
            },
        )
        # W0 naming-sweep finding m9: every review tab's served
        # id/label/description (contracts/openprocessor/openapi/curation.json,
        # `/curation/review/tabs`). The region tab's label is served (the
        # region profile's display name), so tests find that tab by
        # REGION_TAB_LABEL; every other tab falls back to its static label.
        self.on(
            "GET",
            r"/review/tabs(\?|$)",
            {"tabs": [{"id": "regions", "label": REGION_TAB_LABEL}]},
        )
        self.on("GET", r"/bakeoff/runs(\?|$)", {"runs": []})
        # The root layout's ingestAvailability probe fires on every route
        # (same pattern as bakeoff/runs above) — every existing test needs
        # this default so the /ingest nav link's probe doesn't 501.
        self.on("GET", r"/ingest/status(\?|$)", {"total": 0, "by_source": [], "by_day": []})
        # BA-2 (OpenProcessor #36, c676d2b): once the probe above confirms
        # the ingest router is mounted, /ingest's own page fetches
        # `GET /ingest/config` on mount — every existing test needs this
        # default too, same reasoning as /ingest/status above.
        self.on(
            "GET",
            r"/ingest/config(\?|$)",
            {
                "upload": {
                    "enabled": True,
                    "max_images_per_request": 128,
                    "max_bytes_per_request": 268435456,
                    "accepted_extensions": [".jpg", ".jpeg", ".png"],
                    "persists_bytes": True,
                },
                "batch": {"enabled": True, "max_items": 256, "source_roots": []},
                "region_drain": {"poll_interval_s": 10, "stable_polls": 3},
            },
        )

        page.route(f"**{api_prefix}/**", self._dispatch)
        page.route("**/curation/**", self._dispatch_kb)

    def on(self, method: str, path_regex: str, handler_or_body: Handler | HandlerResult) -> None:
        self._handlers.append((method.upper(), re.compile(path_regex), handler_or_body))

    @staticmethod
    def _image(_request: Any, _match: "re.Match[str]") -> HandlerResult:
        return (200, TRANSPARENT_GIF, "image/gif")

    @staticmethod
    def _context(_request: Any, _match: "re.Match[str]") -> HandlerResult:
        # Minimal-but-valid `{API_PREFIX}/crops/{id}/context` default: no
        # items, so SourceImageOverlay renders the (stubbed) image with
        # zero boxes — a test asserting real box geometry registers its
        # own `.on("GET", r"/crops/[^/]+/context$", ...)` instead.
        return (
            200,
            {
                "image": {
                    "image_id": "stub-image",
                    "image_path": "/fixtures/stub-image.jpg",
                    "width": None,
                    "height": None,
                    "source": None,
                    "indexed_at": None,
                },
                "items": [],
            },
        )

    def _dispatch(self, route: Any, request: Any) -> None:
        # Every branch below MUST end in a `route.fulfill`/`route.abort` —
        # an exception escaping this method leaves Playwright's route
        # handler thread having never resolved the route, which hangs the
        # browser's request (and therefore the test) forever instead of
        # failing loudly. The try/except is the harness's own fail-closed
        # guarantee, not app behavior under test.
        try:
            path = re.sub(r"^https?://[^/]+", "", request.url).split("?")[0]
            method = request.method
            body: Any = None
            if request.post_data:
                try:
                    body = json.loads(request.post_data)
                except (ValueError, TypeError):
                    body = request.post_data
            if method != "GET":
                self.calls.append((method, path, body))

            for m, pattern, handler in reversed(self._handlers):
                if m != method:
                    continue
                match = pattern.search(path)
                if not match:
                    continue
                self.handled.append((method, path))
                result = handler(request, match) if callable(handler) else handler
                self._fulfill(route, result)
                return

            # Fail closed: no registered handler matched. Record it and
            # answer 501, never a silent 200 — this is the whole point of
            # P1-1.
            self.unhandled.append((method, path))
            route.fulfill(
                status=501,
                content_type="application/json",
                body=json.dumps({"detail": f"e2e stub: unstubbed {method} {path}"}),
            )
        except Exception as exc:  # noqa: BLE001 — must never leave the route unresolved
            self.handler_errors.append(f"{request.method} {request.url}: {exc!r}")
            route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps({"detail": f"e2e stub handler raised: {exc!r}"}),
            )

    def _dispatch_kb(self, route: Any, request: Any) -> None:
        path = re.sub(r"^https?://[^/]+", "", request.url).split("?")[0]
        self.op_hits.append((request.method, path))
        route.fulfill(
            status=501,
            content_type="application/json",
            body=json.dumps({"detail": f"e2e stub: retired /curation/ prefix hit at {path}"}),
        )

    @staticmethod
    def _fulfill(route: Any, result: HandlerResult) -> None:
        status = 200
        content_type = "application/json"
        body: Any = result
        if isinstance(result, tuple):
            if len(result) == 3:
                status, body, content_type_override = result
                if content_type_override:
                    content_type = content_type_override
            elif len(result) == 2:
                status, body = result
        if content_type == "application/json":
            route.fulfill(status=status, content_type=content_type, body=json.dumps(body))
        else:
            route.fulfill(status=status, content_type=content_type, body=body)

    def assert_fail_closed(self) -> None:
        assert self.handler_errors == [], f"a stub handler raised: {self.handler_errors}"
        assert self.unhandled == [], f"unhandled request(s) hit the stub: {self.unhandled}"
        assert self.op_hits == [], f"retired /curation/ prefix was hit: {self.op_hits}"
        assert self.handled, (
            "no stubbed request was ever hit — an API-prefix change or route "
            "mismatch would otherwise pass silently (see R2 in "
            "docs/design/test-audit-2026-09-24.md)"
        )


@pytest.fixture
def stub(page: Any) -> Any:
    s = Stub(page)
    # Every console message (not just `error`-typed ones) — tests match on
    # console.warn output too (e.g. the malformed-annotation-profile
    # warning), same as the original scripts/playwright_*.py behavior.
    console_errors: list[str] = []
    page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}"))
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
    s.console_errors = console_errors  # type: ignore[attr-defined]
    yield s
    s.assert_fail_closed()
