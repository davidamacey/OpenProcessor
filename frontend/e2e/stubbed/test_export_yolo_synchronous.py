"""Regression coverage for two live /export bugs found by the
train-smoke UI smoke test (artifacts_local/cw-live/train-smoke/):

1. After a successful export, the "frozen multi-class export" card and
   the three registry download buttons (class_registry.json/data.yaml/
   manifest.json) stayed on "No frozen multi-class export yet" /
   disabled until the operator clicked Refresh — the page never
   reloaded `hasMulticlassExport` (from `GET {API_PREFIX}/export/datasets`)
   after `runExport()` resolved.

2. `POST {API_PREFIX}/export/yolo` is synchronous (the live evidence's
   `net_step3_export.json` shows the POST itself returns
   `{"status":"success", ...}` in ~10s) but the page used to assume it
   was a queued-job ack, forcing `exportState.status = 'running'` and
   polling `GET {API_PREFIX}/export/status` — so the modal showed
   "Running…" even though the backend had already said "success", and
   the toast read "Export started: success". The page must render the
   POST response's own served `status` and word the toast from it.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

CLASSES = [
    {
        "class_id": 44,
        "class_name": "mustang",
        "kind": "item",
        "group": "car",
        "hotkey_letter": "m",
        "sample_count": 161,
        "validated_count": 34,
        "cluster_size": 161,
        "deprecated": False,
    },
]

STATS_CLASSES = {
    "classes": [
        {
            "class_id": 44,
            "class_name": "mustang",
            "count": 161,
            "validated_count": 34,
            "adequacy": "warn",
            "aug_target": 500,
            "aug_gap": 466,
            "trainable": 29,
            "trainable_gap": 0,
        },
    ],
    "thresholds": {"block_below": 0, "warn_below": 5, "min_test": 5},
}

STATS_DATASET = {
    "total_crops": 200,
    "validated": 34,
    "test_holdout": 5,
    "by_source": [],
}

HOLDOUT_STATS = {
    "total": 5,
    "by_class": [{"key": 44, "doc_count": 5, "deficient": False}],
    "min_test_per_class": 5,
}

EXPORT_RESULT = {
    "status": "success",
    "export_dir": "/exports/20260924T211523Z",
    "version_tag": "",
    "manifest_path": "/exports/20260924T211523Z/manifest.json",
    "dataset_sha": "ec1226a8cf92d1c60c55a30c781b95d608587da7078d58f92e6e4f970e32111a",
    "split_counts": {"train": 0, "val": 0, "test": 174},
    "dedup": None,
    "started_at": "2026-09-24T21:15:23.518982+00:00",
    "finished_at": "2026-09-24T21:15:33.229271+00:00",
}


def test_export_completes_without_polling_and_refreshes_registry_buttons(stub, page, app_url):
    export_datasets_calls: list[int] = []
    datasets_before = {"datasets": []}
    datasets_after = {
        "datasets": [
            {
                "kind": "yolo",
                "export_dir": "/exports/20260924T211523Z",
                "version_tag": "",
                "is_current": True,
            }
        ]
    }

    def datasets_handler(request, match):
        export_datasets_calls.append(1)
        # Before the export completes: nothing on disk. After: the just-
        # finished yolo export is current. The page must re-fetch this
        # after `runExport()` resolves to flip `hasMulticlassExport`.
        if len(export_datasets_calls) == 1:
            return (200, datasets_before)
        return (200, datasets_after)

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", STATS_CLASSES)
    stub.on("GET", r"/stats/dataset(\?|$)", STATS_DATASET)
    stub.on("GET", r"/test_holdout/stats(\?|$)", HOLDOUT_STATS)
    post_calls: list[dict] = []

    def status_handler(request, match):
        # The page re-reads GET /export/status after the POST resolves
        # (loadAll) and adopts it as the modal's state. A real backend serves
        # the finished export there; an "idle" answer would replace "Export
        # complete." with "No active export." and make the wait below depend
        # on how often the browser happens to poll.
        if post_calls:
            return (
                200,
                {
                    "status": "success",
                    "last_run": EXPORT_RESULT["finished_at"],
                    "export_dir": EXPORT_RESULT["export_dir"],
                },
            )
        return (200, {"status": "idle", "last_run": None})

    stub.on("GET", r"/export/status(\?|$)", status_handler)
    stub.on("GET", r"/export/datasets(\?|$)", datasets_handler)

    def export_yolo_handler(request, match):
        post_calls.append(request.post_data_json or {})
        return (200, EXPORT_RESULT)

    stub.on("POST", r"/export/yolo$", export_yolo_handler)

    page.goto(f"{app_url}/p/default/export")
    page.get_by_text("No frozen multi-class export yet").wait_for(timeout=ACTION_TIMEOUT_MS)

    export_button = page.get_by_role("button", name="Export", exact=True)
    export_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    export_button.click()

    # The modal must never show "Running…" for a synchronous success —
    # it should go straight to "Export complete."
    page.get_by_text("Export complete.", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("Running…", exact=False).count() == 0, (
        "a synchronous success must not render the polling 'Running…' state"
    )

    page.get_by_role("button", name="Close").click()

    # Bug 1: registry buttons + banner must reflect the just-finished
    # export without the operator clicking Refresh.
    page.wait_for_selector("text=No frozen multi-class export yet", state="detached", timeout=ACTION_TIMEOUT_MS)
    registry_button = page.get_by_role("button", name="manifest.json")
    registry_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert registry_button.is_enabled(), (
        "manifest.json download must be enabled once GET /export/datasets reports a current yolo export"
    )
    assert export_datasets_calls[-1] and len(export_datasets_calls) >= 2, (
        "runExport() must re-fetch GET /export/datasets after a successful export, not just on initial load"
    )

    assert len(post_calls) == 1, f"expected exactly one POST /export/yolo: {post_calls}"

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the export flow: {errors[:3]}"


def test_export_is_blocked_by_the_served_can_export(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", STATS_CLASSES)
    stub.on("GET", r"/stats/dataset(\?|$)", STATS_DATASET)
    stub.on("GET", r"/test_holdout/stats(\?|$)", HOLDOUT_STATS)
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})
    stub.on(
        "GET",
        r"/export/status(\?|$)",
        {
            "status": "idle",
            "last_run": None,
            "can_export": False,
            "blocking_reasons": ["nothing to export: 0 items are class_validated"],
        },
    )

    page.goto(f"{app_url}/p/default/export")
    reasons = page.get_by_test_id("export-blocking-reasons")
    reasons.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "nothing to export: 0 items are class_validated" in reasons.inner_text()
    assert page.get_by_role("button", name="Export", exact=True).is_disabled()
