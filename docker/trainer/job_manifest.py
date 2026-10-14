"""Run manifest: the lineage envelope written next to a finished job's status file.

See :mod:`job_protocol` for the overall file protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logutil import get_logger


if TYPE_CHECKING:
    from job_spec import JobSpec
    from job_status import StatusState


logger = get_logger('trainer.job_manifest')


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _checkpoint_sha256(checkpoint_path: Path) -> str | None:
    """Return the sha256 hex digest of ``checkpoint_path``, or ``None``."""
    if not checkpoint_path.is_file():
        return None
    try:
        h = hashlib.sha256()
        with checkpoint_path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                h.update(chunk)
    except OSError as exc:
        logger.warning('checkpoint sha256 failed', path=str(checkpoint_path), error=str(exc))
        return None
    return h.hexdigest()


def _ultralytics_version() -> str | None:
    """Best-effort read of the installed ultralytics package version."""
    try:
        import importlib.metadata as md

        return md.version('ultralytics')
    except Exception:  # PackageNotFoundError family
        return None


def build_lineage(spec: JobSpec) -> dict[str, Any]:
    """Byte-identical lineage dict shared by the run manifest and the
    MLflow tags/params/registry metadata (``mlflow_callbacks.py``).

    Reads straight off ``spec.raw`` -- the job.json the API wrote via
    ``src.services.training.jobs.write_job`` -- so both consumers see the
    exact same values with no second source of truth to drift out of sync.
    ``trainer_sha`` prefers this container's own baked ``OP_BUILD_SHA``
    (the trainer image actually running this code) over the API's stamped
    ``trainer_image_revision`` (the trainer image the API *observed* at
    submit time, via the docker socket) -- the two agree unless the
    trainer image was rebuilt/redeployed between submit and run.
    """
    raw = spec.raw
    return {
        'dataset_sha': raw.get('dataset_sha'),
        'frozen_test_sha': raw.get('frozen_test_sha'),
        'test_label_sha': raw.get('test_label_sha'),
        'dataset_version_tag': raw.get('dataset_version_tag'),
        'api_sha': raw.get('api_sha'),
        'trainer_sha': os.environ.get('OP_BUILD_SHA') or raw.get('trainer_image_revision') or None,
        'trainer_image_id': raw.get('trainer_image_id'),
    }


def write_manifest(
    spec: JobSpec, state: StatusState, class_remap: dict[str, Any] | None = None
) -> None:
    """Write ``<job_id>.manifest.json`` -- the run's full lineage envelope.

    Captures dataset SHA, class remap, code versions and eval results so a
    finished run can be reproduced and audited. Read by
    ``GET {api_prefix}/train/manifest/{job_id}`` and by promote (which resolves
    ``lineage.class_remap`` from here first). Writes are atomic and
    best-effort: a failure logs and moves on rather than blocking the trainer's
    terminal-status write.

    ``class_remap`` is passed in rather than read here: it lives under the
    job's scratch dir, which the caller deletes right after this runs, and this
    module has no business knowing where dataset preparation put it (see
    :func:`dataset_prep.read_class_remap`). ``None`` is correct for a
    whole-export run.
    """
    try:
        checkpoint_sha = (
            _checkpoint_sha256(Path(state.checkpoint_path)) if state.checkpoint_path else None
        )
        lineage = build_lineage(spec)
        manifest: dict[str, Any] = {
            'kind': 'train',
            'job_id': spec.job_id,
            'campaign_id': spec.campaign_id,
            'created_at': datetime.now(tz=UTC).isoformat(),
            'lineage': {
                'export_dir': str(spec.dataset_export_dir),
                'dataset_sha': lineage['dataset_sha'],
                'frozen_test_sha': lineage['frozen_test_sha'],
                'test_label_sha': lineage['test_label_sha'],
                'dataset_version_tag': lineage['dataset_version_tag'],
                'include_classes': spec.include_classes,
                'single_cls': spec.single_cls,
                'class_remap': class_remap,
                'augmentation_seed': (spec.augmentation or {}).get('seed', 42),
                # Actual values handed to model.train(), not just what was
                # requested -- falls back to the requested/default value only
                # if the run failed before train_kwargs was assembled.
                'training_seed': (
                    state.actual_seed
                    if state.actual_seed is not None
                    else int(spec.hyperparameters.get('seed') or 42)
                ),
                'deterministic': (
                    state.actual_deterministic if state.actual_deterministic is not None else True
                ),
                # Registry snapshot the API pinned at submit time -- see
                # src/services/training/jobs.py::write_job.
                'registry_sha': spec.raw.get('registry_sha'),
            },
            'code_versions': {
                'api_sha': lineage['api_sha'],
                'trainer_sha': lineage['trainer_sha'],
                'trainer_image_id': lineage['trainer_image_id'],
                'ultralytics_pkg': _ultralytics_version(),
                'ultralytics_sha': os.environ.get('ULTRALYTICS_SHA'),
            },
            'spec': {
                'model_family': spec.model_family,
                'model_size': spec.model_size,
                'profile': spec.profile,
                'cuda_visible_devices': spec.cuda_visible_devices,
                'hyperparameters': spec.hyperparameters,
                # ``spec.augmentation`` is ALWAYS a dict by this point (the
                # job.json parser coerces a missing/null value to ``{}``), so
                # recording it verbatim would lose the distinction between "no
                # augmentation requested" and "an empty-but-present block". A
                # reproduce flow resubmits this value as-is, and
                # ``AugmentationSpec`` parses a bare ``{}`` into a fully
                # populated, ``enabled=True``-by-default spec -- so a run that
                # never used augmentation would silently get it turned ON.
                # Recording ``None`` for the genuinely-empty case keeps
                # reproduce identical in behavior, not just in JSON shape.
                'augmentation': spec.augmentation or None,
            },
            'results': {
                'final_state': state.state,
                'eval': state.eval,
                'compare': state.compare,
                'last_epoch_metric': state.last_epoch_metric,
                'best_checkpoint_metric': state.best_checkpoint_metric,
                'checkpoint_path': state.checkpoint_path,
                'checkpoint_sha256': checkpoint_sha,
                'mlflow_run_id': state.mlflow_run_id,
                'mlflow_run_url': state.mlflow_run_url,
                'mlflow_experiment_id': state.mlflow_experiment_id,
            },
            # Stamped by POST {api_prefix}/train/promote/{job_id}.
            'promoted_to': None,
        }
        tmp = spec.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(spec.manifest_path)
    except Exception as exc:  # manifest must never block the status write
        logger.warning('manifest write failed', job_id=spec.job_id, error=str(exc))
