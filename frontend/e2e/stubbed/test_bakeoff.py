"""`/bakeoff` on the v2 comparison wire (OpenProcessor #34 §7;
docs/design/bakeoff-v2-ui-plan-2026-09-25.md): select datasets and
models, confirm, the exact POST body, poll queued -> running -> done,
then the served matrix (every tied winner bold) and the per-class table
(a class the model does not cover reads "not covered"). A previous run
whose result predates v2 (409) shows the legacy note, not an error.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import json
from urllib.parse import parse_qs, urlparse

DS_CURRENT = "export:20260924T233203Z"
DS_EXTERNAL = "external:curated/widget_set"

PROFILE = {
    "name": "generic",
    "description": "All classes present in the eval split",
    "kind": "registered",
    "default": True,
    "class_filter": [],
    "imgsz": 640,
    "conf_floor": 0.001,
    "nms_iou": 0.7,
    "op_conf": 0.25,
    "op_iou": 0.45,
    "rank_metric": "map_50_95",
    "default_backend": "ultralytics",
    "triton_model": "",
    "context_class_ids": [],
    "baselines_path": "",
}


def _dataset(ds_id, **over):
    d = {
        "id": ds_id,
        "source": "export",
        "group": None,
        "name": ds_id.split(":", 1)[1],
        "path": f"/exports/{ds_id}",
        "is_current": False,
        "dataset_kind": "multi_class",
        "nc": 5,
        "classes": [
            {"eval_class_id": 0, "name": "gear", "registry_class_id": 10, "n_objects": 5, "n_images": 5},
            {"eval_class_id": 1, "name": "bolt", "registry_class_id": 11, "n_objects": 4, "n_images": 4},
        ],
        "n_images": 9,
        "n_objects": 9,
        "n_background_images": 0,
        "frozen_test_sha": "a" * 16,
        "test_label_sha": "b" * 16,
        "sha_source": "computed",
        "dataset_sha": None,
        "exported_at": "2026-09-24T23:32:11Z",
        "unlabeled_items_on_exported_images": None,
        "frozen_ok": None,
    }
    d.update(over)
    return d


DATASETS = [
    _dataset(DS_CURRENT, is_current=True),
    _dataset(
        DS_EXTERNAL,
        source="external",
        group="curated",
        name="widget_set",
        dataset_kind="external",
        frozen_ok=True,
    ),
]


def _run(run_id, display, overlap_images):
    return {
        "run_id": run_id,
        "display_name": display,
        "model_family": "yolo26",
        "model_size": "n",
        "imgsz": 640,
        "checkpoint_path": f"/runs/{run_id}/weights/best.pt",
        "finished_at": "2026-09-24T23:51:09Z",
        "campaign_id": None,
        "train_export_id": DS_CURRENT,
        "dataset_sha": None,
        "frozen_test_sha": None,
        "class_names": ["gear", "bolt"],
        "single_cls": False,
        "trainer_map50": 0.93,
        "trainer_map50_split": "test",
        "_overlap": overlap_images,
    }


RUNS = [_run("run-a", "full model", 0), _run("run-b", "subset model", 2)]


def _metrics(v):
    return {
        "n_classes": 2, "map_50": v, "map_50_95": v, "map_75": v, "ap_small": None,
        "ap_medium": None, "ap_large": None, "precision": v, "recall": v, "f1": v,
        "mean_iou": v, "tp": 1, "fp": 0, "fn": 0,
    }


def _common(v):
    m = _metrics(v)
    for k in ("map_75", "ap_small", "ap_medium", "ap_large", "mean_iou"):
        m.pop(k)
    return m


def _per_class(cid, name, v, covered=True):
    val = v if covered else None
    return {
        "eval_class_id": cid, "name": name, "n_gt": 5, "covered": covered,
        "model_class_ids": [cid] if covered else [], "ap50": val, "ap50_95": val,
        "ap75": val, "precision": val, "recall": val, "f1": val,
        "tp": 1 if covered else None, "fp": 0 if covered else None, "fn": 0 if covered else None,
    }


def _row(model, display, rank, per_class, coverage, overlap):
    return {
        "rank": rank, "model": model, "display_name": display, "source": "run",
        "run_id": model.split(":", 1)[1], "runtime": "ultralytics", "imgsz": 640,
        "training_data": None, "overall": _metrics(0.6), "common": _common(0.6),
        "per_class": per_class, "coverage": coverage,
        "class_mapping": {"method": "registry_ids", "warnings": []},
        "train_test_overlap": overlap,
        "latency_ms": {"mean": 4.2, "p50": 4.0, "p90": 5.0, "p99": 6.0},
        "fps": 238.0, "size_mb": 5.4, "per_stratum": {},
    }


COMPARISON = {
    "schema_version": 2,
    "job_id": "job-1",
    "profile": "generic",
    "thresholds": {"conf_floor": 0.001, "nms_iou": 0.7, "op_conf": 0.25, "op_iou": 0.45},
    "dataset": {"id": DS_CURRENT, "n_images": 9, "n_objects": 9, "n_background_images": 0},
    "eval_classes": [
        {"eval_class_id": 0, "name": "gear", "n_gt": 5},
        {"eval_class_id": 1, "name": "bolt", "n_gt": 4},
    ],
    "common_classes": [0],
    "rank_by": "map_50_95",
    "rank_scope": "common",
    "models": [
        _row(
            "run:run-a", "full model", 1,
            [_per_class(0, "gear", 0.6), _per_class(1, "bolt", 0.6)],
            {"n_eval_classes": 2, "n_covered": 2, "not_covered": [],
             "unmapped_model_classes": [], "predictions_outside_scored_classes": 0},
            {"n_images": 0, "fraction": 0.0},
        ),
        _row(
            "run:run-b", "subset model", 1,
            [_per_class(0, "gear", 0.6), _per_class(1, "bolt", None, covered=False)],
            {"n_eval_classes": 2, "n_covered": 1,
             "not_covered": [{"eval_class_id": 1, "name": "bolt"}],
             "unmapped_model_classes": [{"model_class_id": 3, "name": "sprocket", "n_predictions": 7}],
             "predictions_outside_scored_classes": 0},
            {"n_images": 2, "fraction": 0.2222},
        ),
    ],
    "failed": [],
    "warnings": [],
    "n_models": 2,
}


def _cell(v, rank):
    return {"map_50": v, "map_50_95": v, "precision": v, "recall": v, "f1": v,
            "latency_ms": 4.2, "size_mb": 5.4, "coverage": 1.0, "rank": rank}


MATRIX = {
    "schema_version": 2,
    "job_id": "job-1",
    "rank_by": "map_50_95",
    "datasets": [{"id": DS_CURRENT, "frozen_test_sha": None, "test_label_sha": None,
                  "rank_scope": "common", "n_common_classes": 1}],
    "models": [
        {"model": "run:run-a", "display_name": "full model", "source": "run"},
        {"model": "run:run-b", "display_name": "subset model", "source": "run"},
    ],
    "metrics": ["map_50", "map_50_95", "precision", "recall", "f1", "latency_ms", "size_mb", "coverage"],
    "cells": {
        "run:run-a": {DS_CURRENT: _cell(0.6, 1)},
        "run:run-b": {DS_CURRENT: _cell(0.6, 1)},
    },
    "best": {DS_CURRENT: {"map_50_95": ["run:run-a", "run:run-b"]}},
}


def _trained_handler(request, _match):
    qs = parse_qs(urlparse(request.url).query)
    ds = qs.get("dataset_id", [None])[0]
    models = []
    for r in RUNS:
        m = {k: v for k, v in r.items() if not k.startswith("_")}
        if ds:
            n = r["_overlap"]
            m["for_dataset"] = {
                "dataset_id": ds, "same_export": ds == DS_CURRENT, "same_frozen_test": None,
                "n_classes_mapped": 2,
                "train_test_overlap": {"n_images": n, "fraction": n / 9},
            }
        models.append(m)
    return (200, {"models": models, "count": len(models)})


def register_discovery(stub):
    # The root layout loads the class registry on every route.
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/bakeoff/profiles(\?|$)", {
        "profiles": [PROFILE], "count": 1, "default_profile": "generic", "default_error": None,
    })
    stub.on("GET", r"/bakeoff/baseline_models(\?|$)", {"baselines": [], "count": 0})
    stub.on("GET", r"/bakeoff/eval_datasets(\?|$)", {"datasets": DATASETS, "count": len(DATASETS)})
    stub.on("GET", r"/bakeoff/trained_models(\?|$)", _trained_handler)


def test_bakeoff_select_run_poll_results(stub, page, app_url):
    register_discovery(stub)
    posted: list = []
    status_polls = {"n": 0}
    job_row = {"job_id": "job-1", "state": "queued", "profile": "generic",
               "datasets": [DS_CURRENT], "models": ["run:run-a", "run:run-b"],
               "started_at": "2026-09-25T01:02:03Z", "finished_at": None}

    def runs_handler(_request, _match):
        return (200, {"runs": [job_row] if posted else []})

    def run_post(request, _match):
        posted.append(json.loads(request.post_data))
        return (200, {
            "status": "enqueued", "job_id": "job-1", "profile": "generic",
            "datasets": [{"id": DS_CURRENT, "path": "/exports/x", "frozen_test_sha": None,
                          "test_label_sha": None, "n_eval_classes": 2}],
            "models": [{
                "model": "run:run-b", "display_name": "subset model", "source": "run",
                "class_mapping": {DS_CURRENT: {
                    "method": "run_class_remap", "model_to_eval": {"0": 0},
                    "unmapped_model_classes": [],
                    "not_covered_eval_classes": [{"eval_class_id": 1, "name": "bolt"}],
                    "warnings": [],
                }},
                "train_test_overlap": {DS_CURRENT: {"n_images": 2, "fraction": 0.2222}},
            }],
            "warnings": [],
        })

    def status_handler(_request, _match):
        status_polls["n"] += 1
        state = ["queued", "running", "done"][min(status_polls["n"] - 1, 2)]
        job_row["state"] = state
        return (200, {
            "schema_version": 2, "job_id": "job-1", "state": state, "profile": "generic",
            "datasets": [DS_CURRENT], "models": ["run:run-a", "run:run-b"],
            "started_at": "2026-09-25T01:02:03Z", "finished_at": None,
            "progress": {"done": 2 if state == "done" else 0, "total": 2},
            "completed": [], "failed": [], "error": None,
        })

    stub.on("GET", r"/bakeoff/runs(\?|$)", runs_handler)
    stub.on("POST", r"/bakeoff/run$", run_post)
    stub.on("GET", r"/bakeoff/status/job-1$", status_handler)
    stub.on("GET", r"/bakeoff/matrix/job-1$", MATRIX)
    stub.on("GET", r"/bakeoff/results/job-1$", COMPARISON)

    page.goto(f"{app_url}/p/default/bakeoff")
    page.locator('[data-testid="run-row"][data-run-id="run-b"]').wait_for(timeout=ACTION_TIMEOUT_MS)

    current = page.locator(f'[data-dataset-id="{DS_CURRENT}"] input[type="checkbox"]')
    assert current.is_checked(), "the current export is preselected"
    assert page.locator('[data-testid="dataset-current"]').count() == 1
    # Facts for the preselected dataset: only the overlapping run warns.
    page.locator('[data-testid="overlap-warning"]').first.wait_for(timeout=5000)
    warns = page.locator('[data-testid="overlap-warning"]')
    assert warns.count() == 1
    assert "2 test images" in warns.first.inner_text()

    page.locator('[data-run-id="run-a"] input[type="checkbox"]').check()
    page.locator('[data-run-id="run-b"] input[type="checkbox"]').check()
    page.get_by_test_id("run-open").click()

    dialog = page.get_by_role("dialog", name="Confirm model comparison")
    dialog.wait_for(timeout=5000)
    assert "2 models × 1 dataset = 2 evaluations" in dialog.get_by_test_id(
        "run-confirm-summary"
    ).inner_text()
    dialog.get_by_role("button", name="Run", exact=True).click()

    page.get_by_test_id("run-status").wait_for(timeout=5000)
    assert posted == [{
        "datasets": [{"id": DS_CURRENT}],
        "models": [{"source": "run", "run_id": "run-a"}, {"source": "run", "run_id": "run-b"}],
        "profile": "generic",
    }], posted

    # Enqueue-time mapping is shown while the job runs.
    assert "not covered: bolt" in page.get_by_test_id("enqueue-summary").inner_text()

    # queued -> running -> done at the 3 s poll interval.
    page.locator('[data-testid="run-state"]:has-text("done")').wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("bakeoff-matrix").wait_for(timeout=5000)
    assert status_polls["n"] >= 3

    best = page.locator('[data-testid="matrix-cell"][data-best="true"]')
    assert best.count() == 2, "both tied winners are bold"
    for i in range(best.count()):
        assert "font-bold" in (best.nth(i).get_attribute("class") or "")

    bolt = page.locator('[data-testid="per-class-table"] tr[data-class-id="1"]')
    assert "not covered" in bolt.inner_text()
    assert "sprocket (7 predictions)" in page.get_by_test_id("unmapped-classes").inner_text()

    assert not [c for c in stub.console_errors if c.startswith("pageerror")]


def test_bakeoff_previous_run_predating_v2(stub, page, app_url):
    register_discovery(stub)
    stub.on("GET", r"/bakeoff/runs(\?|$)", {"runs": [{
        "job_id": "old-job", "state": "done", "profile": None, "datasets": [DS_CURRENT],
        "models": ["run:run-a"], "started_at": "2026-09-01T00:00:00Z", "finished_at": None,
    }]})
    unsupported = (409, {"detail": "bake-off result comparison.json has an unsupported schema"})
    stub.on("GET", r"/bakeoff/matrix/old-job$", unsupported)
    stub.on("GET", r"/bakeoff/results/old-job$", unsupported)

    page.goto(f"{app_url}/p/default/bakeoff")
    page.locator('[data-job-id="old-job"]').click(timeout=ACTION_TIMEOUT_MS)
    note = page.get_by_test_id("comparison-legacy")
    note.wait_for(timeout=5000)
    assert "predates the v2 comparison format" in note.inner_text()
    assert page.get_by_test_id("comparison-error").count() == 0
    assert not [c for c in stub.console_errors if c.startswith("pageerror")]


def test_bakeoff_run_rejection_shows_served_detail(stub, page, app_url):
    register_discovery(stub)
    detail = "single_cls run over 2 classes cannot be scored per class; compare it on a single-class export"
    stub.on("POST", r"/bakeoff/run$", (422, {"detail": detail}))

    page.goto(f"{app_url}/p/default/bakeoff")
    page.locator('[data-run-id="run-a"] input[type="checkbox"]').check(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("run-open").click()
    dialog = page.get_by_role("dialog", name="Confirm model comparison")
    dialog.get_by_role("button", name="Run", exact=True).click()
    err = dialog.get_by_test_id("run-error")
    err.wait_for(timeout=5000)
    assert detail in err.inner_text()
    assert page.get_by_test_id("run-status").count() == 0
