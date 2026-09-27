"""Live read-only tier: `/review?crop_id=` deep links actually land.

Takes the first `verify_rejected` crop id off the live backend (skips if
there is none right now — a live dataset's cohort can be empty),
opens `/review?tab=<region tab>&region_status=verify_rejected&crop_id=<id>`
and asserts the page resolves it: the "Locating crop…" placeholder
clears, no "not in this review queue" toast appears, and the queue
counter shows a real rank (not the empty "—" placeholder).
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import api_get, page_path, wait_for_stable_text
from fixtures.wire import REGION_TAB_URL_ID


def test_deep_link_lands_on_the_requested_crop(
    guarded_page: Any, live_url: str, live_project: dict
) -> None:
    rejected = api_get(
        live_url, live_project, "/review/regions?region_status=verify_rejected&page_size=1"
    )
    items = rejected.get("items", [])
    if not items:
        pytest.skip("no verify_rejected items in the live dataset right now")
    crop_id = items[0]["crop_id"]

    page = guarded_page.page
    page.goto(
        f"{live_url}{page_path(live_project, f'/review?tab={REGION_TAB_URL_ID}&region_status=verify_rejected&crop_id={crop_id}')}",
        wait_until="domcontentloaded",
    )

    # Whether or not "Locating crop…" ever shows (a shallow rank resolves
    # near-instantly), it must not still be showing once we're done
    # waiting for the queue counter to report a real position.
    page.wait_for_selector("text=Locating crop", state="detached", timeout=20_000)

    counter = page.locator('[data-testid="queue-counter"]')
    counter.wait_for(timeout=15_000)
    page.wait_for_function(
        """
        () => {
          const el = document.querySelector('[data-testid="queue-counter"]');
          return !!el && /^#\\d+ · \\d+ loaded/.test(el.textContent.trim());
        }
        """,
        timeout=15_000,
    )
    wait_for_stable_text(page, '[data-testid="queue-counter"]')

    not_in_queue = page.get_by_text("not in this review queue", exact=False)
    assert not_in_queue.count() == 0, (
        f"deep link for crop {crop_id} reported it isn't in the queue: "
        f"{not_in_queue.first.inner_text() if not_in_queue.count() else ''}"
    )
