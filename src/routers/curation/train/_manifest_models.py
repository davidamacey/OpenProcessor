"""Wire model for ``GET /train/manifest/{job_id}``: the trainer's run lineage envelope.

Documentation model, declared through ``responses=``: the route returns the stored dict through
``WireJSONResponse`` so an eval metric that is NaN becomes ``null`` instead of a 500. The trainer
writes this shape (``docker/trainer/job_manifest.py::build_manifest``); free-form blocks stay
``Any`` and every block allows extras, so a key the trainer adds is never hidden from clients.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra='allow')


class ManifestLineage(_Open):
    export_dir: str | None = None
    dataset_sha: str | None = None
    frozen_test_sha: str | None = None
    test_label_sha: str | None = None
    dataset_version_tag: str | None = None
    include_classes: list[Any] | None = None
    single_cls: Any = None
    class_remap: dict[str, Any] | None = None
    augmentation_seed: int | None = None
    training_seed: int | None = None
    deterministic: bool | None = None
    registry_sha: str | None = None


class ManifestCodeVersions(_Open):
    api_sha: str | None = None
    trainer_sha: str | None = None
    ultralytics_pkg: str | None = None
    ultralytics_sha: str | None = None


class ManifestSpec(_Open):
    model_family: str | None = None
    model_size: str | None = None
    profile: str | None = None
    cuda_visible_devices: str | None = None
    hyperparameters: dict[str, Any] | None = None
    augmentation: dict[str, Any] | None = None


class ManifestResults(_Open):
    final_state: str | None = None
    eval: Any = None
    compare: Any = None
    last_epoch_metric: Any = None
    best_checkpoint_metric: Any = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
    mlflow_experiment_id: str | None = None


class ManifestPromotion(_Open):
    triton_model_name: str | None = None
    promoted_at: str | None = None
    force_used: bool | None = None
    gate_report: dict[str, Any] | None = None


class RunManifest(_Open):
    """The run's lineage envelope; ``promoted_to`` is ``null`` until the run is promoted."""

    kind: str | None = None
    job_id: str | None = None
    campaign_id: str | None = None
    created_at: str | None = None
    lineage: ManifestLineage | None = None
    code_versions: ManifestCodeVersions | None = None
    spec: ManifestSpec | None = None
    results: ManifestResults | None = Field(default=None)
    promoted_to: ManifestPromotion | None = None
