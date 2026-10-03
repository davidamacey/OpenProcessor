"""Two projects: switching in the top bar moves every request to the
other project's served prefix and resets the undo stack.

Flow: on `/p/default/review`, Enter labels an item (one undo entry on
`default`), then the switcher opens `beta`. From that click on, no request
touches `default`'s prefix — the review queue, the class registry and the
health read all come from `beta` — and Z finds nothing to undo instead of
reverting `default`'s write through `beta`.

Waits are real (`expect_request` / `wait_for_url` / visible text).
"""

from __future__ import annotations

import re

from conftest import ACTION_TIMEOUT_MS, expect_handled

from fixtures.wire import make_item, project, projects_response

API_PREFIX = "/curation"
DEFAULT_PREFIX = f"{API_PREFIX}/projects/default"
BETA_PREFIX = f"{API_PREFIX}/projects/beta"
GLOBAL_EXACT_PATHS = {f"{API_PREFIX}/projects", f"{API_PREFIX}/health", f"{API_PREFIX}/events"}

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget",
        "kind": "item",
        "group": "g",
        "hotkey_letter": "k",
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
]


def review_item(i: int) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="widget",
        proposed_class_id=1,
        proposed_class_name="widget",
        label_validated=False,
        label_source="model_suggestion",
    )
    item["reason"] = "uncertainty"
    return item


def _path(url: str) -> str:
    return re.sub(r"^https?://[^/]+", "", url).split("?")[0]


def test_switching_moves_requests_to_the_other_prefix_and_resets_undo(stub, page, app_url):
    label_calls: list[str] = []
    undo_calls: list[str] = []

    stub.on(
        "GET",
        rf"^{re.escape(API_PREFIX)}/projects$",
        projects_response(
            API_PREFIX,
            [
                project(API_PREFIX, "default", is_default=True, deletable=False),
                project(API_PREFIX, "beta", display_name="Beta project"),
            ],
        ),
    )
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        lambda *_: (200, {"items": [review_item(i) for i in range(3)], "total": 3, "page": 1, "page_size": 30}),
    )

    def label_handler(request, match):
        label_calls.append(match.string)
        return (200, review_item(0))

    def undo_handler(request, match):
        undo_calls.append(match.string)
        return (200, review_item(0))

    stub.on("PUT", r"/crops/([^/]+)/label$", label_handler)
    stub.on("POST", r"/crops/([^/]+)/label/undo(_batch)?$", undo_handler)

    # The page opens `default`'s event stream on load; the stub records it
    # when it answers, which on a loaded runner can trail the send. Wait for
    # that answer here so it is not counted as a post-switch request below.
    with expect_handled(page, lambda r: _path(r.url) == f"{DEFAULT_PREFIX}/events"):
        page.goto(f"{app_url}/p/default/review")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-switcher-current").inner_text() == "Default"

    # One labelled write on `default` → one undo entry.
    with expect_handled(page, lambda r: r.method == "PUT" and _path(r.url).endswith("/label")):
        page.keyboard.press("Enter")
    assert len(label_calls) == 1 and label_calls[0].startswith(DEFAULT_PREFIX), label_calls

    # Enter advanced the queue, which starts the next crop's image loads on
    # `default`. Let them land before marking the switch, so only requests
    # made after the switch are counted below.
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    # Switch to `beta` from the top bar.
    before_switch = len(stub.handled)
    page.get_by_test_id("project-switcher-trigger").click()
    page.get_by_test_id("project-switcher-menu").wait_for(timeout=ACTION_TIMEOUT_MS)
    with (
        page.expect_request(lambda r: _path(r.url) == f"{BETA_PREFIX}/review/all"),
        page.expect_request(lambda r: _path(r.url) == f"{BETA_PREFIX}/classes"),
        page.expect_request(lambda r: _path(r.url) == f"{BETA_PREFIX}/health"),
    ):
        page.get_by_test_id("project-option-beta").click()
    page.wait_for_url(re.compile(r"/p/beta/review$"), timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-switcher-current").inner_text() == "Beta project"

    # Z: the undo stack was reset on the switch — nothing to revert, and
    # no undo request on either prefix.
    page.keyboard.press("z")
    page.get_by_text("Nothing to undo.").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert undo_calls == [], f"Z must not revert the previous project's write: {undo_calls}"

    after = [p for _, p in stub.handled[before_switch:]]
    old = [p for p in after if p.startswith(f"{DEFAULT_PREFIX}/")]
    assert old == [], f"request(s) to the previous project after switching: {old}"
    stray = [p for p in after if p not in GLOBAL_EXACT_PATHS and not p.startswith(BETA_PREFIX)]
    assert stray == [], f"request(s) outside beta's prefix after switching: {stray}"

    # Browser back is a switch too: the previous project's page comes back
    # on its own prefix.
    with page.expect_request(lambda r: _path(r.url) == f"{DEFAULT_PREFIX}/review/all"):
        page.go_back()
    page.wait_for_url(re.compile(r"/p/default/review$"), timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-switcher-current").inner_text() == "Default"
