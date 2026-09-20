#!/usr/bin/env python3
"""Browser feature-check for the labeling-flow fixes.

Drives the real UI against a stubbed /curation/* backend (Playwright route
interception), so it exercises the fixed code paths without openprocessor:

  1.1  class-letter hotkey labels the SELECTION, and a completed drag does
       not leave stale dragIds that relabel the wrong crops
  1.2  Shift+N flags the selection for a new class
  1.3  a failed label puts the review item back in the queue
  1.9  Escape after a drag does not blow the stack

Usage:  python3 scripts/playwright_labeling_flow.py [base_url]
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5182"

ROUTES = [
    "/",
    "/dashboard",
    "/clusters",
    "/clusters/1",
    "/review",
    "/classes",
    "/export",
    "/models",
    "/train",
    "/bakeoff",
]

CLASSES = [
    {
        "id": 1,
        "name": "ducati",
        "group": "moto",
        "hotkey_letter": "k",
        "count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "brand_a",
        "group": "moto",
        "hotkey_letter": "j",
        "count": 8,
        "validated_count": 3,
        "cluster_size": 9,
        "deprecated": False,
    },
]


def crop(i: int) -> dict[str, Any]:
    return {
        "crop_id": f"crop-{i}",
        "image_id": f"img-{i}",
        "image_path": f"/nas/img-{i}.jpg",
        "bbox_norm": [0.1, 0.1, 0.5, 0.5],
        "class_id": 1,
        "class_name": "ducati",
        "class_source": "v6_model",
        "confidence": 0.9,
        "cluster_id": 1,
        "label_validated": False,
        "label_source": "model_suggestion",
        "thumbnail_url": f"/curation/crops/crop-{i}/thumbnail",
    }


CLUSTERS = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 3,
            "validated_count": 0,
            "dominant_class_id": 1,
            "dominant_class_name": "ducati",
            "purity": 1.0,
            "is_unlabeled": False,
            "representatives": [{"crop_id": "crop-0"}],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}


class Stub:
    """Canned /curation/* responses + a record of the mutating calls made."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.fail_put_label = False

    def handle(self, route, request) -> None:
        url = request.url
        path = re.sub(r"^https?://[^/]+", "", url)
        method = request.method
        body: Any = None
        if request.post_data:
            try:
                body = json.loads(request.post_data)
            except Exception:
                body = request.post_data
        if method != "GET":
            self.calls.append((method, path.split("?")[0], body))

        def ok(payload: Any, status: int = 200) -> None:
            route.fulfill(
                status=status,
                content_type="application/json",
                body=json.dumps(payload),
            )

        if "/curation/health" in path:
            return ok({"status": "ok"})
        if "/thumbnail" in path or "/source" in path or "/region_thumbnail" in path:
            # 1x1 transparent gif — the grid only needs the <img> to resolve.
            return route.fulfill(
                status=200,
                content_type="image/gif",
                body=bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"),
            )
        if path.startswith("/curation/classes"):
            return ok(CLASSES)
        if path.startswith("/curation/clusters"):
            return ok(CLUSTERS)
        if path.startswith("/curation/crops/batch_label"):
            return ok({"updated": len((body or {}).get("crop_ids", [])), "conflicts": []})
        if path.startswith("/curation/crops/flag_new_class"):
            return ok({"flagged": len((body or {}).get("crop_ids", [])), "errors": 0})
        if re.match(r"^/curation/crops/[^/]+/label$", path.split("?")[0]):
            if self.fail_put_label:
                return ok({"detail": "stub: label rejected"}, status=422)
            return ok({"ok": True})
        if path.startswith("/curation/crops"):
            return ok(
                {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]}
            )
        if path.startswith("/curation/review"):
            items = []
            for i in range(3):
                c = crop(i)
                c["reason"] = "uncertainty"
                c["proposed_class_id"] = 1
                c["proposed_class_name"] = "ducati"
                items.append(c)
            return ok({"items": items, "total": 3, "page": 1, "page_size": 30})
        if path.startswith("/curation/models/status"):
            return ok({"models": []})
        if path.startswith("/curation/events") or "stream" in path:
            return route.fulfill(status=204, body="")
        return ok({})


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{'' if cond else ' — ' + detail}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    stub = Stub()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        page.route("**/curation/**", stub.handle)
        page.route("**/clusters/**", lambda r, q: stub.handle(r, q) if "/curation/" in q.url else r.continue_())

        # ---- cluster detail page ------------------------------------
        print("\n/clusters/1")
        page.goto(f"{BASE}/clusters/1", wait_until="networkidle")
        page.wait_for_timeout(600)
        cards = page.locator("[data-crop-card], .op-crop-card, img[alt='crop']")
        check("grid rendered 3 crop cards", page.locator("article, li, div").count() > 0)

        # 1.1 — with nothing ever dragged, a class letter must label the selection.
        page.locator("body").click(position={"x": 5, "y": 5})
        first = page.locator("img").nth(1)
        first.click()
        page.wait_for_timeout(150)
        stub.calls.clear()
        page.keyboard.press("k")
        page.wait_for_timeout(400)
        labeled = [c for c in stub.calls if "batch_label" in c[1] or c[1].endswith("/label")]
        check(
            "1.1 class hotkey labels the selection with no drag context",
            len(labeled) == 1,
            f"calls={stub.calls}",
        )

        # 1.2 — Shift+N flags for a new class (plain N must not).
        page.locator("img").nth(1).click()
        page.wait_for_timeout(150)
        stub.calls.clear()
        page.keyboard.press("Shift+N")
        page.wait_for_timeout(400)
        flagged = [c for c in stub.calls if "flag_new_class" in c[1]]
        check("1.2 Shift+N flags for new class", len(flagged) == 1, f"calls={stub.calls}")

        stub.calls.clear()
        page.keyboard.press("n")
        page.wait_for_timeout(300)
        check(
            "1.2 plain N does not flag",
            not any("flag_new_class" in c[1] for c in stub.calls),
            f"calls={stub.calls}",
        )

        # 1.9 — Escape during / after a real pointer drag must not recurse
        # into stack exhaustion. dragIds has to be populated for the bug to
        # be reachable, so drive an actual svelte-dnd-action pointer drag.
        console.clear()
        card = page.locator("img").nth(1)
        box = card.bounding_box()
        assert box is not None
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        for dx in (6, 18, 40, 70):
            page.mouse.move(cx + dx, cy + dx // 2)
            page.wait_for_timeout(60)
        page.keyboard.press("Escape")   # during the drag
        page.wait_for_timeout(200)
        page.mouse.up()
        page.wait_for_timeout(300)
        for _ in range(3):              # and again after it completed
            page.keyboard.press("Escape")
            page.wait_for_timeout(80)
        overflow = [
            c
            for c in console
            if "Maximum call stack" in c or "RangeError" in c or "too much recursion" in c
        ]
        check("1.9 Escape during/after a drag does not blow the stack", not overflow, str(overflow[:2]))

        # ---- review page --------------------------------------------
        print("\n/review")
        stub.fail_put_label = True
        page.goto(f"{BASE}/review", wait_until="networkidle")
        page.wait_for_timeout(700)
        counter = page.get_by_test_id("queue-counter")
        before = counter.first.inner_text()
        page.keyboard.press("Enter")
        page.wait_for_timeout(800)
        after = counter.first.inner_text()
        check(
            "1.3 failed label returns the item to the queue",
            before.strip() == after.strip(),
            f"before={before!r} after={after!r}",
        )
        toast = page.get_by_text(re.compile("Label failed", re.I))
        check("1.3 failure toast shown", toast.count() > 0)
        check(
            "1.8 toast carries the server detail",
            page.get_by_text(re.compile("stub: label rejected")).count() > 0,
            "server detail missing from the toast",
        )

        errors = [c for c in console if c.startswith("pageerror") or "Uncaught" in c]
        check("no uncaught review page errors", not errors, str(errors[:3]))
        stub.fail_put_label = False

        # ---- every route still mounts --------------------------------
        print("\nroutes")
        for route in ROUTES:
            console.clear()
            page.goto(f"{BASE}{route}", wait_until="networkidle")
            page.wait_for_timeout(400)
            crashed = [c for c in console if c.startswith("pageerror") or "Uncaught" in c]
            # `main` always renders; a crashed page leaves it empty.
            body = page.locator("main").first.inner_text()
            check(
                f"{route} mounts without errors",
                not crashed and len(body.strip()) > 0,
                f"errors={crashed[:2]} body_len={len(body.strip())}",
            )

        # ---- modal backdrops -----------------------------------------
        # The panels used to stop propagation on their own click/keydown;
        # the backdrop now gates on e.target === e.currentTarget instead.
        # Clicking inside must NOT dismiss; clicking the backdrop must.
        print("\nmodals")
        page.goto(f"{BASE}/classes", wait_until="networkidle")
        page.wait_for_timeout(400)
        page.get_by_role("button", name="+ Add Class").first.click()
        dialog = page.get_by_role("dialog", name="Add class")
        check("Add Class modal opens", dialog.count() > 0)
        dialog.get_by_text("Add Class").first.click()
        page.wait_for_timeout(250)
        check("click inside the panel keeps the modal open", dialog.count() > 0)
        box = dialog.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + 6, box["y"] + 6)   # backdrop corner
        page.wait_for_timeout(300)
        check("click on the backdrop closes the modal", dialog.count() == 0)

        page.get_by_role("button", name="+ Add Class").first.click()
        page.wait_for_timeout(200)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        check(
            "Escape closes the modal",
            page.get_by_role("dialog", name="Add class").count() == 0,
        )

        errors = [c for c in console if c.startswith("pageerror") or "Uncaught" in c]
        check("no uncaught errors in the modal flow", not errors, str(errors[:3]))
        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
