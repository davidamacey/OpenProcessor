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
  * At teardown, the ``stub`` fixture asserts ``unhandled == []`` and
    that at least one request was actually handled — a prefix/route
    mismatch fails the test instead of passing on an empty page.

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
from urllib.parse import urlparse

from fixtures.wire import (
    REGION_PROFILE,
    REGION_TAB_LABEL,
    projects_response,
    review_tab,
    review_tabs,
)

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

# Single, documented action-timeout budget for the whole stubbed suite.
#
# Every explicit `.wait_for(timeout=...)`/`.click(timeout=...)` call used to
# hardcode a bare `15000` literal, independently, in ~40 files. On a heavily
# loaded host (concurrent docker builds, GPU engine compiles pegging most
# cores — see CLAUDE.md) that fixed 15s budget was occasionally too tight
# for hydration + the stub's request round trip to finish, not a real app
# defect: a controlled repro (pinning `vite preview` and pytest to a single
# CPU core contended by an equal-priority stress load) reproduced both a
# ~10x slowdown (13 tests: ~26s -> ~261s) and, under heavier contention, the
# preview server failing to even finish starting inside its own startup
# window. 45s gives real headroom for that kind of contention while still
# failing fast (well under pytest's own per-test wall clock) on an actual
# regression. Import this into a test file instead of writing a new literal.
ACTION_TIMEOUT_MS = 45000

# 1x1 transparent GIF — grids only need the <img> to resolve, not real pixels.
TRANSPARENT_GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)

HandlerResult = Any  # dict/list (-> 200 json) | tuple[int, Any] | tuple[int, Any, str]
Handler = Callable[[Any, "re.Match[str]"], HandlerResult]


class _Handled:
    """Context manager whose `.value` is the matched *request*, resolved only
    once the stub has answered it."""

    def __init__(self, manager):
        self._manager = manager
        self._info = None

    def __enter__(self):
        self._info = self._manager.__enter__()
        return self

    def __exit__(self, *exc):
        return self._manager.__exit__(*exc)

    @property
    def value(self):
        return self._info.value.request


def expect_handled(page, predicate, timeout=None):
    """Wait for the stub to have answered a matching request, not just for the
    browser to send it: the handlers record bodies, and on a slow runner the
    recording can trail the send."""
    return _Handled(page.expect_response(lambda resp: predicate(resp.request), timeout=timeout))


def wait_for_paint(page: Any) -> None:
    """Wait for two real animation frames instead of an arbitrary sleep.

    A handful of interactions (mid-drag Escape, rapid Escape presses) have
    no app-exposed DOM/network signal to wait on — the assertion that
    follows is about the ABSENCE of a console error, not a state change a
    selector can observe. Sleeping a fixed duration there is exactly the
    flake-under-load pattern this whole rewrite removes: this instead
    waits on the browser's own paint pipeline (two rAF callbacks
    guarantees at least one full frame was rendered), which naturally
    slows down under host contention the same way real waiting would,
    without picking an arbitrary millisecond budget.
    """
    page.evaluate(
        "() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the server's whole process group, SIGKILL it after a grace."""
    import os
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=15)


def _wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"vite preview exited early (code {proc.returncode}):\n{out}")
        try:
            urllib.request.urlopen(base_url, timeout=1)
            break
        except urllib.error.URLError:
            time.sleep(0.5)
    else:
        _stop_group(proc)
        raise RuntimeError(f"vite preview did not come up within {timeout}s at {base_url}")

    # Warm-up request: the port accepting connections doesn't guarantee the
    # static handler's own first-request bookkeeping (route table build,
    # first disk reads into the OS page cache) is done — under host
    # contention that first real page load can be meaningfully slower than
    # every one after it. A single throwaway fetch here, outside any test's
    # own timeout budget, absorbs that cost once instead of it landing on
    # whichever test happens to run first.
    try:
        urllib.request.urlopen(base_url, timeout=max(timeout, 10))
    except urllib.error.URLError:
        pass  # the real per-test navigation will surface a genuine failure


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
        # Own process group so the whole tree (npx wrapper + real server)
        # can be stopped; terminating the wrapper alone orphans the server.
        start_new_session=True,
    )
    base_url = f"http://localhost:{port}"
    try:
        _wait_for_server(base_url, proc)
        yield base_url
    finally:
        _stop_group(proc)


class Stub:
    """Fail-closed router for `{api_prefix}/**`.

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
        # P1 projects cutover: the GLOBAL project list, read once by the
        # root layout's bootstrap before anything scoped fires. Every
        # scoped request in the app is then built from this project's own
        # served `prefix` (`{api_prefix}/projects/default`) — matched by
        # every OTHER `.on(...)` pattern below purely by suffix, so this
        # is the only project-aware default the stub needs.
        self.on(
            "GET",
            rf"^{re.escape(api_prefix)}/projects$",
            projects_response(api_prefix),
        )
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
        self.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None, "monitoring_links": {"grafana": None, "prometheus": None, "opensearch_dashboards": None}})
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
                # OpenProcessor 3f1a11e adoption: labeled
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
            review_tabs(review_tab("regions", REGION_TAB_LABEL)),
        )
        self.on("GET", r"/bakeoff/runs(\?|$)", {"runs": []})
        # /ingest's own page reads its status table and its config on
        # mount; defaults so a route sweep through /ingest never 501s.
        self.on("GET", r"/ingest/status(\?|$)", {"total": 0, "by_source": [], "by_day": []})
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
                # v0.4.0: no detector reported and the live default policy
                # (revision 0, nothing filtered, embed every item).
                "detector": None,
                "policy": {
                    "detect": {
                        "min_confidence": None,
                        "min_box_area_frac": None,
                        "max_per_image": None,
                        "classes": None,
                        "exclude_classes": [],
                        "class_resolution": "proposal",
                    },
                    "embedding": {
                        "min_confidence": None,
                        "min_box_area_frac": None,
                        "max_per_image": None,
                        "mode": "all",
                        "classes": [],
                    },
                    "detector": None,
                    "revision": 0,
                },
            },
        )
        # K2 (docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md
        # §5.1): the root layout's `loadKeymap()` fires `GET {prefix}/keymap`
        # on EVERY route now, same "every existing test needs a default"
        # rationale as `/bakeoff/runs`/`/ingest/status` above. Defaults to a
        # 404 — the pre-W2b behaviour every test was written against — so
        # every existing test stays green without editing each one;
        # `test_keymap.py` overrides this per-test with a served document.
        self.on("GET", r"/keymap(\?|$)", (404, {"detail": "not found"}))
        # Projects P2 (§5.1): the `/p/[project]` layout reads the active
        # project's served pipeline-pause flag (`GET {prefix}/pause`) on
        # EVERY route for the switcher's chip, and `/projects` reads it
        # for every selectable row. Default: not paused.
        # `test_projects_pause.py` overrides it with a stateful stub.
        self.on("GET", r"/pause$", self._pause_state)
        # W10 (dataset import + Reprocess): /ingest, the item-detail panel
        # and the cluster toolbar probe `GET {prefix}/datasets/formats`
        # once per project. Defaults to a 404 — a backend without W10,
        # where every W10 surface is absent — so existing tests stay
        # green; test_dataset_import.py overrides it with served formats.
        self.on("GET", r"/datasets/formats(\?|$)", (404, {"detail": "Not Found"}))
        # W3 (prompt-pack CRUD): /settings and the pack pages probe
        # `GET {prefix}/prompt_packs` once per project. Defaults to a 404 —
        # a backend without W3, where every pack surface is absent — so
        # existing tests stay green; test_prompt_packs.py overrides it.
        self.on("GET", r"/prompt_packs(\?|$)", (404, {"detail": "Not Found"}))
        # W4 (region-profile CRUD): /settings and the profile pages probe
        # `GET {prefix}/region_profiles` once per project. Defaults to a
        # 404 — a backend without W4, where every profile surface is
        # absent — so existing tests stay green; test_region_profiles.py
        # overrides it.
        self.on("GET", r"/region_profiles(\?|$)", (404, {"detail": "Not Found"}))
        # W9 (VLM endpoint registry): the Settings card and /models probe the
        # GLOBAL `GET {prefix}/vlm/endpoints` once. Defaults to a 404 — a
        # backend without W9 — so existing tests stay green; the Track A
        # specs override it.
        self.on("GET", r"/vlm/endpoints(\?|$)", (404, {"detail": "Not Found"}))
        # P4 (combine projects): `/projects` probes
        # `GET {prefix}/projects/combine/<sentinel>` once. A plain 404 is the
        # "router not mounted" shape; Track B specs override it.
        self.on("GET", r"/projects/combine/[^/?]+(\?|$)", (404, {"detail": "Not Found"}))
        # OpenProcessor v0.4.0 reads
        # (docs/design/v040-backend-deltas-ui-plan-2026-10-03.md §3 item 8).
        # Default 404s so existing tests keep their zero-unhandled guarantee
        # once the tracks add these reads: `/open_vocab` is the open-vocab
        # editor's gate (404 = absent); the region stage, detections summary
        # and ingest policy reads render their error line. Each track's
        # specs override these with `stub.on(...)`.
        self.on("GET", r"/open_vocab(\?|$)", (404, {"detail": "Not Found"}))
        self.on("GET", r"/region_stage(\?|$)", (404, {"detail": "Not Found"}))
        self.on("GET", r"/detections/summary(\?|$)", (404, {"detail": "Not Found"}))
        self.on("GET", r"/ingest/policy(\?|$)", (404, {"detail": "Not Found"}))

        page.route(f"**{api_prefix}/**", self._dispatch)

    def on(self, method: str, path_regex: str, handler_or_body: Handler | HandlerResult) -> None:
        self._handlers.append((method.upper(), re.compile(path_regex), handler_or_body))

    @staticmethod
    def _pause_state(request: Any, _match: "re.Match[str]") -> HandlerResult:
        path = urlparse(request.url).path.rstrip("/")
        slug = path.split("/")[-2]
        return {"project": slug, "paused": False, "paused_by": [], "reason": None}

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
        assert self.handled, (
            "no stubbed request was ever hit — an API-prefix change or route "
            "mismatch would otherwise pass silently (see R2 in "
            "docs/design/test-audit-2026-09-24.md)"
        )


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
    log = _PageLog(page)
    yield page
    rep = getattr(request.node, "rep_call", None) or getattr(request.node, "rep_setup", None)
    failed = rep is not None and rep.failed
    if failed or getattr(request.node, "_e2e_teardown_failed", False):
        name = re.sub(r"[^\w.-]+", "_", request.node.nodeid)
        stub = getattr(request.node, "_e2e_stub", None)
        log.dump(page, FAILURE_DIR / name, stub)


class AppAssets:
    """What `serve_app_through_harness` served (URLs) and failed to fetch."""

    def __init__(self) -> None:
        self.served: list[str] = []
        self.errors: list[str] = []


def fetch_app_asset(route: Any, attempts: int = 3) -> Any:
    """`route.fetch()` for one of the app's own files, retrying a connection
    reset. `vite preview` closes idle keep-alive sockets; a request that
    reuses one in that instant fails with `read ECONNRESET`, which used to be
    aborted as a failed chunk and left the page without its first element.
    Only a reset is retried (these are idempotent GETs); any other error, and
    a reset that persists, propagates."""
    for attempt in range(attempts):
        try:
            # 3xx goes to the browser as-is, so a navigation lands on the
            # URL the server actually redirected to.
            return route.fetch(max_redirects=0)
        except Exception as exc:  # noqa: BLE001 — re-raised unless a retryable reset
            if "ECONNRESET" not in str(exc) or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def serve_app_through_harness(page: Any, app_url: str) -> AppAssets:
    """Answer every request for the app's own origin (HTML, JS chunks,
    CSS, static files) with `route.fetch()` + `route.fulfill()` instead of
    letting Chromium's network stack load it.

    Why: Chromium aborts every request still queued for a socket with
    `net::ERR_NETWORK_CHANGED` whenever the host's network configuration
    changes — including a docker container starting or stopping on this
    machine (its host-side veth gaining/losing an IPv6 address). The SPA
    boots by loading ~60 module chunks over 6 connections per origin, so a
    container event in that ~100 ms window fails a chunk import, SvelteKit
    renders "500 Internal Error", and the test times out on its first
    element. Reproduced deterministically by starting a short-lived
    container while requests are queued; see the CHANGELOG entry.
    `route.fetch()` runs in Playwright's own HTTP client, so a fulfilled
    route never touches Chromium's socket pool. The `{api_prefix}` stub,
    registered after this, still answers the API (later routes win).

    A fetch error (a down preview server, say) is recorded and the route
    aborted rather than left hanging.
    """
    assets = AppAssets()

    def handler(route: Any) -> None:
        try:
            response = fetch_app_asset(route)
            # Recorded before fulfilling: the browser's `requestfinished`
            # can be dispatched while `fulfill` is still waiting.
            assets.served.append(route.request.url)
            route.fulfill(response=response)
        except Exception as exc:  # noqa: BLE001 — must never leave the route unresolved
            assets.errors.append(f"{route.request.method} {route.request.url}: {exc!r}")
            try:
                route.abort("failed")
            except Exception:  # noqa: BLE001 — already resolved; nothing else to do
                pass

    page.route(f"{app_url.rstrip('/')}/**", handler)
    return assets


@pytest.fixture
def stub(page: Any, request: Any, app_url: str) -> Any:
    assets = serve_app_through_harness(page, app_url)
    s = Stub(page)
    s.app_assets = assets  # type: ignore[attr-defined]
    request.node._e2e_stub = s
    # Every console message (not just `error`-typed ones) — tests match on
    # console.warn output too (e.g. the malformed-annotation-profile
    # warning), same as the original scripts/playwright_*.py behavior.
    console_errors: list[str] = []
    page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}"))
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
    s.console_errors = console_errors  # type: ignore[attr-defined]
    yield s
    try:
        s.assert_fail_closed()
    except AssertionError:
        # `page` tears down after this fixture; tell it to dump diagnostics.
        request.node._e2e_teardown_failed = True
        raise
