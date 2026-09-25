"""Live read-only tier: every top-level route mounts against the real
deployment without erroring.

Each route in ROUTES is opened in a real browser pointed at
`CROPWRIGHT_LIVE_URL` (nginx on :5184 by default, proxying `/curation` to
a live OpenProcessor backend). For each we assert:

  * no `pageerror` (guaranteed by `guarded_page`'s teardown — see
    conftest.py);
  * no `**/curation/**` response >= 400, except a documented allow-list
    entry (`is_allowlisted_bad_response`, conftest.py) — empty today,
    because every route this backend serves came back 200/204 in manual
    verification (`curl` against :5184/curation/{classes,methods,
    regions/statuses,regions/vocabulary,review/tabs,class_sources,
    settings,bakeoff/runs,stats/dataset,training_cohorts}`);
  * no literal "NaN" or "undefined" in the rendered body text — the
    classic symptom of an unmapped/undefined field leaking into a
    template;
  * every `<img>` whose bounding box intersects the 1280x720 viewport
    finishes loading (`naturalWidth > 0`) — a lazy offscreen image is
    allowed to still be pending;
  * (2026-09-24 visual-review follow-up) no horizontal page overflow at
    the narrow 800px viewport — `document.documentElement.scrollWidth <=
    window.innerWidth + 1`.

Every route also gets a full-page screenshot saved at both a desktop
(1600x1000) and a narrow (800x1000) viewport, unconditionally — not just
on failure — under `artifacts_local/cw-live/live-shots/<run-timestamp>/
<route-slug>-<width>.png` (`screenshot_run_dir`, conftest.py). These are
NOT self-checking: per CLAUDE.md's "Live read-only tier" section, a human
must actually open a sample of them after each run.

This is deliberately a shallow "did it mount cleanly" sweep, not a
feature check — see test_data_agreement.py and test_deep_link.py for
tests that check the UI's numbers actually match the API's.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from conftest import is_allowlisted_bad_response
from fixtures.wire import REGION_TAB_URL_ID

# Viewport widths every route is screenshotted at; height is fixed so a
# route's screenshot is comparable across runs regardless of content
# length (full_page=True still captures anything below the fold).
SCREENSHOT_VIEWPORTS: list[tuple[int, int]] = [(1600, 1000), (800, 1000)]


def _route_slug(path: str) -> str:
    """`/review?tab=model_disagreements` -> `review-tab-model_disagreements`."""
    slug = re.sub(r"[^a-zA-Z0-9_]+", "-", path).strip("-").lower()
    return slug or "root"

# (path, a selector proving the route actually mounted its real content,
# not just an empty shell / loading spinner). `{region_class}` is filled
# from the deployment's served region profile; routes that need one skip
# on a deployment without it.
ROUTES: list[tuple[str, str]] = [
    ("/dashboard", 'h1:has-text("Dashboard")'),
    ("/ingest", 'h1:has-text("Ingest")'),
    ("/clusters", 'h1:has-text("Clusters")'),
    ("/clusters?class={region_class}", 'h1:has-text("Clusters")'),
    ("/review?tab=all", '[data-testid="queue-counter"]'),
    ("/review?tab=uncertainty", '[data-testid="queue-counter"]'),
    ("/review?tab=model_disagreements", '[data-testid="queue-counter"]'),
    ("/review?tab=classifier_blind_spots", '[data-testid="queue-counter"]'),
    ("/review?tab=new_class_proposals", '[data-testid="queue-counter"]'),
    (f"/review?tab={REGION_TAB_URL_ID}", '[data-testid="queue-counter"]'),
    ("/classes", 'h1:has-text("Class management")'),
    ("/export", 'h1:has-text("Export dataset")'),
    ("/train", 'h1:has-text("Train model")'),
    ("/models", 'h1:has-text("Models")'),
    ("/bakeoff", '[data-testid="dataset-picker"]'),
    ("/settings", 'h1:has-text("Deployment defaults")'),
]


def _bad_responses(gp: Any) -> list[tuple[str, str, int]]:
    """Collected as (method, path, status) for every `**/curation/**`
    response >= 400 seen while this list is wired up by the caller."""
    return gp.bad_responses


@pytest.mark.parametrize("path,ready_selector", ROUTES, ids=[r[0] for r in ROUTES])
def test_route_mounts_cleanly(
    guarded_page: Any,
    live_url: str,
    live_region_profile: dict[str, Any] | None,
    path: str,
    ready_selector: str,
    screenshot_run_dir: Path,
) -> None:
    needs_region = "{region_class}" in path or path == f"/review?tab={REGION_TAB_URL_ID}"
    if needs_region and live_region_profile is None:
        pytest.skip("this deployment serves no region profile")
    if live_region_profile is not None:
        path = path.replace("{region_class}", live_region_profile["region_class_name"])
    gp = guarded_page
    page = gp.page

    def _record_response(response: Any) -> None:
        url = response.url
        if "/curation/" not in url:
            return
        status = response.status
        if status < 400:
            return
        method = response.request.method
        # Strip origin + query for the allow-list match.
        bare_path = url.split("://", 1)[-1].split("/", 1)[-1]
        bare_path = "/" + bare_path.split("?")[0]
        if is_allowlisted_bad_response(method, bare_path, status):
            return
        gp.bad_responses.append((method, bare_path, status))

    page.on("response", _record_response)

    page.goto(f"{live_url}{path}", wait_until="domcontentloaded")
    page.wait_for_selector(ready_selector, timeout=15_000)

    # Full-page screenshots at both viewports, saved unconditionally
    # (before any assertion below can fail) — see the module docstring
    # and CLAUDE.md's live-tier section: these must be visually reviewed
    # by a human, the sweep itself only proves the route mounted.
    # `wait_for_timeout` here is a deliberate, narrow exception to this
    # tier's "no fixed sleep" rule — there is no DOM condition to wait on
    # for "the post-resize reflow has settled" the way there is for a
    # fetch or an image load.
    default_viewport = page.viewport_size
    slug = _route_slug(path)
    narrow_overflow: bool | None = None
    for width, height in SCREENSHOT_VIEWPORTS:
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(150)
        page.screenshot(
            path=str(screenshot_run_dir / f"{slug}-{width}.png"),
            full_page=True,
        )
        if width == 800:
            narrow_overflow = page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            )
    if default_viewport is not None:
        page.set_viewport_size(default_viewport)

    assert narrow_overflow, (
        f"{path}: horizontal page overflow at 800px "
        f"(document.documentElement.scrollWidth > window.innerWidth + 1) — "
        f"see {screenshot_run_dir / f'{slug}-800.png'}"
    )

    # Give in-flight `{API_PREFIX}` fetches issued on mount a chance to
    # land and their images to start loading, without a fixed sleep or
    # `wait_until="load"` — wait for the concrete condition we actually
    # care about (every in-viewport image either loaded or still
    # legitimately pending offscreen never blocks us).
    page.wait_for_function(
        """
        () => {
          const vw = window.innerWidth, vh = window.innerHeight;
          const imgs = Array.from(document.querySelectorAll('img'));
          const inViewport = imgs.filter((img) => {
            const r = img.getBoundingClientRect();
            return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0
              && r.top < vh && r.left < vw;
          });
          return inViewport.every((img) => img.complete);
        }
        """,
        timeout=15_000,
    )

    broken = page.evaluate(
        """
        () => {
          const vw = window.innerWidth, vh = window.innerHeight;
          const imgs = Array.from(document.querySelectorAll('img'));
          return imgs
            .filter((img) => {
              const r = img.getBoundingClientRect();
              return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0
                && r.top < vh && r.left < vw;
            })
            .filter((img) => img.complete && img.naturalWidth === 0)
            .map((img) => img.src);
        }
        """
    )
    assert broken == [], f"{path}: in-viewport image(s) failed to load: {broken}"

    body_text = page.locator("body").inner_text()
    assert "NaN" not in body_text, f"{path}: literal 'NaN' rendered in body text"
    assert "undefined" not in body_text, f"{path}: literal 'undefined' rendered in body text"

    assert gp.bad_responses == [], (
        f"{path}: unexpected >=400 {{API_PREFIX}} response(s) (not on the "
        f"allow-list): {gp.bad_responses}"
    )
