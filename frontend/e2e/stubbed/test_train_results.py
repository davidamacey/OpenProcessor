"""Finished-run Results view on /train (past-runs table): evaluation
(labelled by its own eval.split) + lineage for a terminal run, served by
GET {API_PREFIX}/train/status/{job_id} and GET
{API_PREFIX}/train/manifest/{job_id}. Fixture data mirrors the real
served shape captured live from the 2026-09-24T23-47-55_yolo26n
train-smoke run (artifacts_local/cw-live/train-smoke/f_status.json /
f_manifest.json), hand-corrected for OpenProcessor #34 W1's
best_metric/last_metric removal: that run predates
last_epoch_metric/best_checkpoint_metric, so both serve null under the
W1 fix (never a back-filled guess) -- see
src/lib/test/fixtures/trainRun.ts's doc comment for the same call on
the frontend fixture this mirrors.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from test_train_gpus import register_train_mount

JOB_ID = "2026-09-24T23-47-55_yolo26n"

FINISHED_STATUS = {
    "job_id": JOB_ID,
    "campaign_id": None,
    "state": "finished",
    "started_at": "2026-09-24T23:47:56Z",
    "finished_at": "2026-09-24T23:51:09Z",
    "current_epoch": 20,
    "total_epochs": 20,
    "epoch_time_s": None,
    "last_epoch_metric": None,
    "best_checkpoint_metric": None,
    "mlflow_run_id": "724a9292103d4ec3b153068758be340d",
    "mlflow_run_url": "http://op-mlflow:5000/#/experiments/1/runs/724a9292103d4ec3b153068758be340d",
    "checkpoint_path": "/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/weights/best.pt",
    "gpu": [{"index": 0, "util_pct": 2, "mem_used_mb": 18155, "mem_total_mb": 49140}],
    "eval": {
        "split": "test",
        "map50": 0.9191,
        "map50_95": 0.85096,
        "per_class": [
            {
                "class_id": 0,
                "name": "miata",
                "precision": 1.0,
                "recall": 0.5758532802586369,
                "f1": 0.730846313514828,
                "ap50": 0.755,
                "support": 5,
            },
            {
                "class_id": 1,
                "name": "minicooper",
                "precision": 0.6392386650044177,
                "recall": 1.0,
                "f1": 0.7799214094339276,
                "ap50": 0.9378571428571427,
                "support": 5,
            },
        ],
        "confusion_matrix_path": "/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/confusion_matrix.png",
    },
    "compare": None,
    "error": None,
    "heartbeat_at": "2026-09-24T23:51:09Z",
}

MANIFEST = {
    "campaign_id": None,
    "code_versions": {
        "api_sha": None,
        "trainer_sha": None,
        "trainer_image_id": None,
        "ultralytics_pkg": "8.4.48",
        "ultralytics_sha": "8f5d355cd05b91503e7bf62429681dea4fa4b004",
    },
    "created_at": "2026-09-24T23:51:09.494255+00:00",
    "job_id": JOB_ID,
    "kind": "train",
    "lineage": {
        "augmentation_seed": 42,
        "class_remap": {
            "include_classes": [38, 39, 44, 52, 79],
            "names": ["miata", "minicooper", "mustang", "porsche", "vw"],
            "new_to_original": {"0": 38, "1": 39, "2": 44, "3": 52, "4": 79},
            "original_to_new": {"38": 0, "39": 1, "44": 2, "52": 3, "79": 4},
            "single_cls": False,
        },
        "dataset_sha": None,
        "deterministic": True,
        "export_dir": "/exports/20260924T233203Z",
        "include_classes": [38, 39, 44, 52, 79],
        "registry_sha": "3aee444124c8fa161030c424cc891a40139cde7b96d5cf73d0363e00d7c7f671",
        "single_cls": False,
        "training_seed": 42,
    },
    "promoted_to": None,
    "results": {
        "last_epoch_metric": None,
        "best_checkpoint_metric": None,
        "checkpoint_path": "/var/lib/openprocessor/training_runs/2026-09-24T23-47-55_yolo26n/weights/best.pt",
        "checkpoint_sha256": "cc5ffb75e020b54d87df6f534de2a7a74eaa02519c69658a9504e2fe42d15e81",
        "compare": None,
        "eval": FINISHED_STATUS["eval"],
        "final_state": "finished",
        "mlflow_run_id": "724a9292103d4ec3b153068758be340d",
        "mlflow_run_url": "http://op-mlflow:5000/#/experiments/1/runs/724a9292103d4ec3b153068758be340d",
    },
    "spec": {
        "cuda_visible_devices": "2",
        "model_family": "yolo26",
        "model_size": "n",
        "profile": "probe",
    },
}


def test_finished_run_results_render(stub, page, app_url):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [FINISHED_STATUS], "total": 1})
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", MANIFEST)

    page.goto(f"{app_url}/p/default/train")
    page.get_by_text(JOB_ID, exact=False).first.wait_for(timeout=ACTION_TIMEOUT_MS)

    results_button = page.get_by_role("button", name="Results")
    results_button.wait_for(timeout=10000)
    results_button.click()

    page.get_by_text("Loading manifest", exact=False).wait_for(state="hidden", timeout=10000)

    # Overall and per-class figures, both labelled by the served eval.split.
    page.get_by_text("test split (frozen holdout)", exact=False).first.wait_for(timeout=10000)

    # Per-class table row.
    assert page.get_by_text("miata", exact=False).count() > 0
    assert page.get_by_text("minicooper", exact=False).count() > 0

    # Lineage from the manifest.
    assert page.get_by_text("/exports/20260924T233203Z", exact=False).count() > 0
    assert (
        page.get_by_text(
            "cc5ffb75e020b54d87df6f534de2a7a74eaa02519c69658a9504e2fe42d15e81",
            exact=False,
        ).count()
        > 0
    )

    # MLflow link renders (non-null url per the backend contract).
    mlflow_link = page.locator(
        "a[href='http://op-mlflow:5000/#/experiments/1/runs/724a9292103d4ec3b153068758be340d']"
    )
    assert mlflow_link.count() > 0

    # Confusion matrix path renders as text only, never an <img>.
    assert (
        page.get_by_text("confusion_matrix.png", exact=False).count() > 0
    ), "confusion matrix server path should render as text"

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_failed_run_shows_served_error(stub, page, app_url):
    register_train_mount(stub)
    failed_job_id = "2026-09-24T20-00-00_yolo26n_failed"
    failed_status = {
        **FINISHED_STATUS,
        "job_id": failed_job_id,
        "state": "failed",
        "error": "CUDA out of memory",
        "eval": None,
        "mlflow_run_id": None,
        "mlflow_run_url": None,
    }
    stub.on("GET", r"/train/runs(\?|$)", {"items": [failed_status], "total": 1})
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", (404, {"detail": "not found"}))

    page.goto(f"{app_url}/p/default/train")
    page.get_by_text(failed_job_id, exact=False).first.wait_for(timeout=ACTION_TIMEOUT_MS)

    results_button = page.get_by_role("button", name="Results")
    results_button.wait_for(timeout=10000)
    results_button.click()

    page.get_by_text("CUDA out of memory", exact=False).wait_for(timeout=10000)

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
