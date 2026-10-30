"""The fail-closed `{api_prefix}` stub, its served defaults and the `stub`
fixture."""

from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.parse import urlparse

import pytest

from fixtures.wire import (
    REGION_PROFILE,
    REGION_TAB_LABEL,
    projects_response,
    review_tab,
    review_tabs,
)
from harness.assets import serve_app_through_harness
from harness.constants import API_PREFIX, TRANSPARENT_GIF

HandlerResult = Any  # dict/list (-> 200 json) | tuple[int, Any] | tuple[int, Any, str]
Handler = Callable[[Any, "re.Match[str]"], HandlerResult]

REPROCESS_VOCABULARY: dict = {
    "scopes": [
        {"id": i, "label": label, "description": f"{label}: served description"}
        for i, label in [
            ("detect", "Find objects"),
            ("open_vocab", "Find by description"),
            ("region", "Region stage"),
            ("vlm", "Vision model"),
            ("embed", "Compute vectors"),
        ]
    ],
    "filter_fields": [],
    "job_statuses": [],
    "lock_reasons": [
        {"id": i, "label": label, "description": f"{label}: served description"}
        for i, label in [
            ("human_label", "Human label"),
            ("validated", "Validated"),
            ("imported", "Imported"),
            ("test_holdout", "Test holdout"),
        ]
    ],
}


def resource_link(id, label, url, kind="service", status=None, hint="", reachable=None):
    """One served `GET /settings` `resource_links` entry."""
    return {
        "id": id, "label": label, "url": url, "kind": kind, "hint": hint, "reachable": reachable,
        "status": status or ("configured" if url else "not_configured"),
    }


# The backend's path-relative docs entries (same-origin nginx proxies).
DOCS_RESOURCE_LINKS = [
    resource_link("swagger", "API reference (Swagger UI)", "/docs", "docs"),
    resource_link("redoc", "API reference (ReDoc)", "/redoc", "docs"),
    resource_link("openapi_json", "OpenAPI JSON", "/openapi.json", "docs"),
]


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
        # /clusters/[id] lists the assist runs for the viewed cluster on every visit.
        self.on("GET", r"/assist/runs(\?|$)", {"runs": []})
        # The clusters page probes the auto-label loop panel on every visit.
        self.on("GET", r"/assist/loops(\?|$)", {"loops": []})
        # ...and reads its facet counts; a test that checks chips overrides this.
        self.on("GET", r"/clusters/(\d+)/facets$", self._empty_facets)
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
        self.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None, "resource_links": DOCS_RESOURCE_LINKS})
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
        # The deployment-configured detector/
        # segmenter/verifier vocabulary. Shapes lifted from the vendored
        # OpenAPI description (contracts/openapi/curation.json,
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
                # Labeled
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
        # Every review tab's served
        # id/label/description (contracts/openapi/curation.json,
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
            r"/ingest/sources(\?|$)",
            {
                "items": [],
                "s3_configured": False,
                "s3_allowed": [],
                "default_batch_size": 16,
                "default_concurrency": 16,
                "default_watch_interval_s": 60,
                "min_watch_interval_s": 10,
                "poll_after_s": 5,
            },
        )
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
        # The root layout's `loadKeymap()` fires `GET {prefix}/keymap`
        # on EVERY route now, same "every existing test needs a default"
        # rationale as `/bakeoff/runs`/`/ingest/status` above. Defaults to a
        # 404 — what the tests were written against — so
        # every existing test stays green without editing each one;
        # `test_keymap.py` overrides this per-test with a served document.
        self.on("GET", r"/keymap(\?|$)", (404, {"detail": "not found"}))
        # Projects P2 (§5.1): the `/p/[project]` layout reads the active
        # project's served pipeline-pause flag (`GET {prefix}/pause`) on
        # EVERY route for the switcher's chip, and `/projects` reads it
        # for every selectable row. Default: not paused.
        # `test_projects_pause.py` overrides it with a stateful stub.
        self.on("GET", r"/pause$", self._pause_state)
        # /ingest, the item-detail panel
        # and the cluster toolbar probe `GET {prefix}/datasets/formats`
        # once per project. Defaults to a 404 — a backend without
        # dataset import, where every import surface is absent — so existing tests stay
        # green; test_dataset_import.py overrides it with served formats.
        self.on("GET", r"/datasets/formats(\?|$)", (404, {"detail": "Not Found"}))
        # The Reprocess dialog and the lock badge read the served
        # `reprocess` block of `GET {prefix}/config/vocabulary` (only once
        # import is served); tests that need the full vocabulary override it.
        self.on("GET", r"/config/vocabulary(\?|$)", (200, {"reprocess": REPROCESS_VOCABULARY}))
        # /settings and the pack pages probe
        # `GET {prefix}/prompt_packs` once per project. Defaults to a 404 —
        # a backend without prompt packs, where every pack surface is absent — so
        # existing tests stay green; test_prompt_packs.py overrides it.
        self.on("GET", r"/prompt_packs(\?|$)", (404, {"detail": "Not Found"}))
        # /settings and the profile pages probe
        # `GET {prefix}/region_profiles` once per project. Defaults to a
        # 404 — a backend without region profiles, where every profile surface is
        # absent — so existing tests stay green; test_region_profiles.py
        # overrides it.
        self.on("GET", r"/region_profiles(\?|$)", (404, {"detail": "Not Found"}))
        # the Settings card and /models probe the
        # GLOBAL `GET {prefix}/vlm/endpoints` once. Defaults to a 404 — a
        # backend without the VLM registry — so existing tests stay green; the
        # VLM specs override it.
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
        # Label confirmation (#119): /settings reads the VLM scope policy on
        # mount and /audit its report and queue. Default 404s keep the zero-
        # unhandled guarantee; test_label_confirmation.py overrides them.
        self.on("GET", r"/vlm/policy(\?|$)", (404, {"detail": "Not Found"}))
        # Dashboard quick action reads the project's active VLM; a failed read
        # keeps the button as before, so the default is a 404.
        self.on("GET", r"/vlm/endpoints/active(\?|$)", (404, {"detail": "Not Found"}))
        self.on("GET", r"/audit/(report|queue)(\?|$)", (404, {"detail": "Not Found"}))

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
    def _empty_facets(_request: Any, match: "re.Match[str]") -> HandlerResult:
        return (
            200,
            {
                "cluster_id": int(match.group(1)),
                "total": 0,
                "detector_classes": [],
                "class_sources": [],
                "validation": {"validated": 0, "unvalidated": 0},
                "review_tabs": {},
                "detector_confidence": [],
                "detector_confidence_missing": 0,
            },
        )

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
