"""New coverage (docs/design/test-audit-2026-09-24.md recommendation 5):
`/train` renders the GPU picker from `GET {API_PREFIX}/train/gpus`
(TrainForm.svelte). A restricted allowlist renders one radio per option,
preselecting the server's `default: true` entry.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

CLASSES = [
    {"id": 1, "name": "ducati", "group": "moto", "hotkey_letter": "k", "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

GPU_OPTIONS = {
    "options": [
        {"value": "1", "gpu_ids": [1], "label": "GPU 1 (RTX 3080 Ti)", "advisory": None, "stops_containers": [], "default": False},
        {"value": "2", "gpu_ids": [2], "label": "GPU 2 (RTX A6000)", "advisory": "stops the op-api container", "stops_containers": ["op-api"], "default": True},
    ],
    "allowed_ids": [1, 2],
    "unrestricted": False,
}


def register_train_mount(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/export/status(\?|$)", {"status": "success", "last_run": "2026-09-01T00:00:00Z", "export_dir": "/nas/exports/2026-09-01"})
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})
    stub.on("GET", r"/train/profiles(\?|$)", {"profiles": []})
    stub.on("GET", r"/train/presets(\?|$)", {"class_subset_presets": []})
    stub.on("GET", r"/train/runs(\?|$)", {"items": [], "total": 0})
    stub.on("GET", r"/train/gpus(\?|$)", GPU_OPTIONS)
    # OpenProcessor df01309: AugmentationPanel now fetches its preset list
    # from the backend instead of a hardcoded id table.
    stub.on(
        "GET",
        r"/train/augmentation_presets(\?|$)",
        {
            "presets": [
                {
                    "id": "balanced_default",
                    "label": "Balanced (default)",
                    "description": "Broad, mild coverage.",
                    "orientation_sensitive": False,
                },
            ],
            "default": "balanced_default",
        },
    )
    stub.on("GET", r"/train/status(\?|$)", (200, None))
    stub.on("GET", r"/training_cohorts(\?|$)", {"cohorts": []})
    stub.on("POST", r"/train/preflight(\?|$)", {"blocked": False, "checks": [], "summary": None})
    # Coordinator finding: the class-subset picker's "N validated crops"
    # summary now shows a served holdout figure alongside it — /train
    # fetches this on mount the same way /export already does.
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    # #36 item 8: ProbeControl fires GET /probe/status for any finished
    # run that has a checkpoint_path once its Results panel is opened
    # (adopt-in-flight check) — a default "idle" answer here means every
    # existing /train test that doesn't care about probing still passes
    # the fail-closed unhandled-request guard.
    stub.on("GET", r"/probe/status(\?|$)", {"status": "idle"})


def test_train_gpu_options(stub, page, app_url):
    register_train_mount(stub)

    page.goto(f"{app_url}/train")
    page.get_by_text("GPUs", exact=True).first.wait_for(timeout=ACTION_TIMEOUT_MS)

    radios = page.get_by_role("radio")
    expect(radios).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("GPU 1 (RTX 3080 Ti)").count() > 0
    assert page.get_by_text("GPU 2 (RTX A6000)").count() > 0

    checked = [i for i in range(radios.count()) if radios.nth(i).is_checked()]
    assert len(checked) == 1, f"exactly one GPU radio should be preselected: {checked}"
    assert radios.nth(checked[0]).get_attribute("value") == "2", (
        "the server's default:true option (GPU 2) should be preselected"
    )

    assert page.get_by_text("stops the op-api container").count() > 0, (
        "the selected option's advisory text should render"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_train_gpu_options_unrestricted(stub, page, app_url):
    register_train_mount(stub)
    stub.on("GET", r"/train/gpus(\?|$)", {"options": [], "allowed_ids": [], "unrestricted": True})

    page.goto(f"{app_url}/train")
    page.locator("input[aria-label='CUDA visible devices']").wait_for(timeout=ACTION_TIMEOUT_MS)

    assert page.locator("input[aria-label='CUDA visible devices']").count() == 1, (
        "an unrestricted backend should show the free-text CUDA-devices field"
    )
    assert page.get_by_role("radio").count() == 0

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
