"""No horizontal page overflow at 430px on any project route (#184).

jsdom has no layout, so this runs in a real browser: for each route it
asserts `documentElement.scrollWidth <= window.innerWidth` and, on failure,
names the elements with the largest right edge. Fail-closed: every request
the page makes must be stubbed (the `stub` fixture asserts it at teardown).
"""

from __future__ import annotations

import pytest

from conftest import ACTION_TIMEOUT_MS, wait_for_paint

from fixtures.wire import REGION_CLASS, REGION_TAB_URL_ID, make_item

NARROW = {"width": 430, "height": 900}

CLASSES = [
    {"class_id": 64, "class_name": "class_b", "kind": "item", "group": "moto", "hotkey_letter": None,
     "sample_count": 641, "validated_count": 0, "cluster_size": 616, "deprecated": False},
    {"class_id": 80, "class_name": REGION_CLASS, "kind": "region", "group": "widgets", "hotkey_letter": None,
     "sample_count": 40, "validated_count": 0, "cluster_size": 44, "deprecated": False},
]

# (path under /p/default, a selector proving the page mounted)
ROUTES: list[tuple[str, str]] = [
    ("/dashboard", "h1"),
    ("/ingest", "h1"),
    ("/clusters", "h1"),
    ("/clusters/64", "[data-testid='class-name']"),
    ("/review?tab=all", "[data-testid='queue-counter']"),
    ("/review?tab=uncertainty", "[data-testid='queue-counter']"),
    ("/review?tab=model_disagreements", "[data-testid='queue-counter']"),
    ("/review?tab=classifier_blind_spots", "[data-testid='queue-counter']"),
    ("/review?tab=new_class_proposals", "[data-testid='queue-counter']"),
    (f"/review?tab={REGION_TAB_URL_ID}", "[data-testid='queue-counter']"),
    ("/audit", "h1"),
    ("/classes", "h1"),
    ("/export", "h1"),
    ("/models", "h1"),
    ("/train", "h1"),
    ("/bakeoff", "h1"),
    ("/settings", "h1"),
    ("/settings/ingest-policy", "h1"),
    ("/settings/models", "h1"),
    ("/settings/models/new-endpoint", "h1"),
    ("/settings/open-vocab", "h1"),
    ("/settings/prompt-packs", "h1"),
    ("/settings/region-profiles", "h1"),
    ("/datasets", "h1"),
    ("/datasets/import", "h1"),
    ("/datasets/imports", "h1"),
]
GLOBAL_ROUTES: list[tuple[str, str]] = [
    ("/projects", "h1"),
    ("/projects/combine", "h1"),
]

PROBE = """() => {
  const vw = window.innerWidth;
  // An element inside a scroll container is clipped by it, not overflowing the page.
  const clipped = (el) => {
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      const ox = getComputedStyle(p).overflowX;
      if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') return true;
    }
    return false;
  };
  const out = [];
  for (const el of document.body.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > vw + 1 && !clipped(el)) {
      out.push({el: el.tagName + '.' + String(el.className).slice(0, 60)
        + (el.dataset.testid ? '[' + el.dataset.testid + ']' : ''), right: Math.round(r.right)});
    }
  }
  out.sort((a, b) => b.right - a.right);
  return {scrollWidth: document.documentElement.scrollWidth, innerWidth: vw, worst: out.slice(0, 5)};
}"""


def _crop(i: int) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        image_path=f"/nas/img-{i}.jpg",
        class_id=64,
        class_name="class_b",
        class_source="model",
        confidence=0.9,
        cluster_id=64,
        label_validated=False,
        label_source="model_suggestion",
        proposed_class_id=64,
        proposed_class_name="class_b",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )
    item["reason"] = "uncertainty"
    return item


CLUSTERS = {
    "items": [
        {"cluster_id": 64, "cluster_kind": "class", "size": 3, "validated_count": 0,
         "dominant_class_id": 64, "dominant_class_name": "class_b", "purity": 1.0,
         "is_unlabeled": False, "representatives": [{"crop_id": "crop-0"}],
         "n_subclusters": 0, "updated_at": None},
    ],
    "total": 1, "total_class_clusters": 1, "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}

IDLE_JOB = {
    "job_id": "job-0", "status": "idle", "stage": "", "processed": 0, "total": 0,
    "started_at": 0, "finished_at": 0, "error": None, "result": {}, "args": {},
    "eta_seconds": None, "elapsed_seconds": 0,
}


def _stub_common(stub) -> None:
    """Idle but well-formed answers for every read the swept pages make."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/regions/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/regions(\?|$)", {"items": [], "total": 0})
    stub.on("GET", r"/crops(\?|$)",
            {"total": 3, "page": 1, "page_size": 60, "crops": [_crop(i) for i in range(3)]})
    stub.on("GET", r"/models/status", {"models": []})
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)
    stub.on("GET", r"/stats/dataset(\?|$)",
            {"total_crops": 3, "validated": 0, "test_holdout": 0, "by_source": {}})
    stub.on("GET", r"/test_holdout/stats(\?|$)",
            {"total": 0, "by_class": [], "min_test_per_class": 5})
    stub.on("GET", r"/export/status(\?|$)", {"status": "idle", "last_run": None})
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})
    stub.on("GET", r"/train/status(\?|$)", {"jobs": []})
    stub.on("GET", r"/train/runs(\?|$)", {"items": [], "total": 0})
    stub.on("GET", r"/train/profiles(\?|$)", {"profiles": []})
    stub.on("GET", r"/train/presets(\?|$)", {"class_subset_presets": []})
    stub.on("GET", r"/train/gpus(\?|$)", {"options": [], "allowed_ids": [], "unrestricted": True})
    stub.on("GET", r"/training_cohorts(\?|$)", {"cohorts": []})
    stub.on("GET", r"/bakeoff/profiles(\?|$)",
            {"profiles": [], "count": 0, "default_profile": None, "default_error": None})
    stub.on("GET", r"/bakeoff/eval_datasets(\?|$)", {"datasets": [], "count": 0})
    stub.on("GET", r"/bakeoff/trained_models(\?|$)", {"models": [], "count": 0})
    stub.on("GET", r"/bakeoff/baseline_models(\?|$)", {"baselines": [], "count": 0})
    stub.on("GET", r"/ingest/region_drain(\?|$)", (404, {"detail": "Not Found"}))
    stub.on("GET", r"/scores/coverage(\?|$)", (404, {"detail": "Not Found"}))
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", (404, {"detail": "Not Found"}))

    def review(_request, _match):
        return (200, {"items": [_crop(i) for i in range(3)], "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review)


@pytest.mark.parametrize("path,ready", ROUTES + GLOBAL_ROUTES, ids=[r[0] for r in ROUTES + GLOBAL_ROUTES])
def test_no_horizontal_overflow_at_430(stub, page, app_url, path, ready):
    page.set_viewport_size(NARROW)
    _stub_common(stub)
    prefix = "" if path.startswith("/projects") else "/p/default"
    page.goto(f"{app_url}{prefix}{path}")
    page.locator(ready).first.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    m = page.evaluate(PROBE)
    assert m["scrollWidth"] <= m["innerWidth"], f"{path} overflows at 430px: {m}"
