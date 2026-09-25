"""Custom MLflow logging hooks for Ultralytics training.

Ultralytics' built-in MLflow integration auto-logs scalar metrics + ``best.pt``
+ run params when ``MLFLOW_TRACKING_URI`` is set. This module adds:

* ``on_train_start``   -- class distribution params + ``class_distribution.json``
                          artifact, 16-image augmented preview montage, run tags.
* ``on_fit_epoch_end`` -- per-class mAP@0.5 and mAP@0.5:0.95.
* ``on_val_end``       -- confusion-matrix artifact every N epochs.
* ``on_train_end``     -- register ``best.pt`` in the MLflow Model Registry
                          under :data:`DEFAULT_MODEL_REGISTRY_NAME` (override
                          per-deployment with ``OP_MLFLOW_MODEL_NAME``) in stage
                          ``Staging``. Production promotion stays manual.

All hooks are best-effort: if MLflow is unreachable mid-run we log + continue
rather than break the training loop. Multi-day runs must not crash because
metrics shipping hiccupped.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


log = logging.getLogger('mlflow_cb')


# Deliberately generic: this repo ships no domain model name. A deployment
# points its runs at its own registered model via OP_MLFLOW_MODEL_NAME.
DEFAULT_MODEL_REGISTRY_NAME = os.environ.get('OP_MLFLOW_MODEL_NAME', 'openprocessor_detector')

# How often (in epochs) to ship the confusion matrix as an artifact.
CONFUSION_MATRIX_EVERY_N_EPOCHS = int(os.environ.get('OP_MLFLOW_CONFUSION_MATRIX_EVERY', '25'))


def _safe_mlflow() -> Any | None:
    """Import mlflow lazily so a missing package doesn't kill training if the
    operator wants to run without a tracking server."""
    try:
        import mlflow
    except Exception as exc:  # MLflow is optional at runtime
        log.warning('mlflow unavailable: %s', exc)
        return None
    return mlflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_labels(data_cfg: dict[str, Any], data_yaml_path: Path) -> dict[str, int]:
    """Count class IDs across train + val label files.

    Returns ``{class_name: count}`` (zeros included).
    """
    names_raw = data_cfg.get('names') or {}
    if isinstance(names_raw, list):
        names: dict[int, str] = dict(enumerate(names_raw))
    else:
        names = {int(k): v for k, v in names_raw.items()}

    counts: Counter[int] = Counter()
    base = data_yaml_path.parent
    for split_key in ('train', 'val'):
        split_path = data_cfg.get(split_key)
        if not split_path:
            continue
        # data.yaml may give a directory of images -- labels live alongside as
        # 'labels/{split}/*.txt' relative to the dataset root (Ultralytics
        # layout).
        candidates = [
            base / 'labels' / split_key,
            (base / split_path).resolve().parent.parent / 'labels' / split_key,
        ]
        for label_dir in candidates:
            if not label_dir.is_dir():
                continue
            for txt in label_dir.glob('*.txt'):
                # Best-effort: a malformed label file shouldn't kill the whole
                # class-distribution dump.
                with contextlib.suppress(Exception):
                    for line in txt.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        counts[int(parts[0])] += 1
            break
    return {names.get(i, f'class_{i}'): counts.get(i, 0) for i in sorted(names)}


def _augment_preview_montage(trainer: Any) -> bytes | None:
    """Build a 4x4 grid of augmented training samples as JPEG bytes.

    Returns ``None`` if anything goes wrong -- the montage is a nice-to-have,
    never a blocker.
    """
    try:
        from PIL import Image

        loader = getattr(trainer, 'train_loader', None) or getattr(trainer, 'trainloader', None)
        if loader is None:
            return None
        batch = next(iter(loader))
        imgs = batch['img'] if isinstance(batch, dict) else batch[0]
        # imgs shape: (B, 3, H, W) uint8 or float
        sel = imgs[: min(16, imgs.shape[0])]
        if hasattr(sel, 'detach'):
            sel = sel.detach().cpu().numpy()
        if sel.dtype.kind == 'f':
            sel = (sel * 255.0).clip(0, 255).astype('uint8')
        # CHW -> HWC
        sel = sel.transpose(0, 2, 3, 1)
        h, w = sel.shape[1:3]
        grid_n = 4
        out = Image.new('RGB', (w * grid_n, h * grid_n), (0, 0, 0))
        for i in range(min(16, sel.shape[0])):
            tile = Image.fromarray(sel[i])
            out.paste(tile, ((i % grid_n) * w, (i // grid_n) * h))
        buf = io.BytesIO()
        out.save(buf, format='JPEG', quality=80)
    except Exception as exc:  # montage is cosmetic
        log.warning('augment preview montage failed: %s', exc)
        return None
    return buf.getvalue()


def _get_class_names(trainer: Any) -> dict[int, str]:
    names = getattr(trainer.model, 'names', None) or getattr(trainer, 'names', None) or {}
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------


def register_callbacks(  # four closures + wiring, cohesive by design
    model: Any,
    *,
    run_name: str,
    profile: str,
    seed: int,
    lineage: dict[str, Any],
    data_cfg: dict[str, Any],
    data_yaml_path: Path,
    model_registry_name: str = DEFAULT_MODEL_REGISTRY_NAME,
) -> None:
    """Wire all four hook handlers into the Ultralytics model.

    The closure captures the run metadata so handlers are pure-functional with
    respect to the training loop. ``lineage`` (``job_protocol.build_lineage``)
    is the single source of dataset/build identity -- MLflow tags, params and
    registry metadata read from it so they are byte-identical to the run
    manifest's own ``lineage``/``code_versions`` blocks.
    """
    state: dict[str, Any] = {
        'class_dist': None,
        'names_by_id': None,
        'registry_name': model_registry_name,
        'every_n_epochs_for_cm': CONFUSION_MATRIX_EVERY_N_EPOCHS,
    }

    def _tag_run(mlflow: Any) -> None:
        try:
            tags = {
                'git_sha': lineage.get('trainer_sha') or 'unknown',
                'dataset_sha': str(lineage.get('dataset_sha') or ''),
                'dataset_version': str(lineage.get('dataset_version_tag') or ''),
                'frozen_test_sha': str(lineage.get('frozen_test_sha') or ''),
                'test_label_sha': str(lineage.get('test_label_sha') or ''),
                'api_sha': str(lineage.get('api_sha') or ''),
                'docker_digest_trainer': str(lineage.get('trainer_image_id') or ''),
                'profile': profile,
                'seed': str(seed),
                'run_name': run_name,
            }
            mlflow.set_tags(tags)
            mlflow.log_param('profile', profile)
            mlflow.log_param('seed', seed)
            mlflow.log_param('dataset_sha', tags['dataset_sha'])
            mlflow.log_param('dataset_version', tags['dataset_version'])
        except Exception as exc:  # tracking hiccup must not kill the run
            log.warning('tag run failed: %s', exc)

    # -----------------------------------------------------------------
    # on_train_start
    # -----------------------------------------------------------------
    def on_train_start(trainer: Any) -> None:
        mlflow = _safe_mlflow()
        if mlflow is None:
            return
        try:
            _tag_run(mlflow)
            class_dist = _count_labels(data_cfg, data_yaml_path)
            state['class_dist'] = class_dist
            state['names_by_id'] = _get_class_names(trainer)
            mlflow.log_param('class_count', len(class_dist))
            mlflow.log_dict(class_dist, 'class_distribution.json')
            for cname, count in class_dist.items():
                # Param names must be safe -- Ultralytics may also log params.
                safe = 'count_' + cname.replace('/', '_').replace(' ', '_')[:240]
                # Params can't be re-logged with different values; suppress.
                with contextlib.suppress(Exception):
                    mlflow.log_param(safe, count)
            montage = _augment_preview_montage(trainer)
            if montage:
                tmp = Path(trainer.save_dir) / 'augment_preview.jpg'
                tmp.write_bytes(montage)
                mlflow.log_artifact(str(tmp))
        except Exception:  # never break the training loop
            log.warning('on_train_start failed:\n%s', traceback.format_exc())

    # -----------------------------------------------------------------
    # on_fit_epoch_end -- per-class mAP
    # -----------------------------------------------------------------
    def on_fit_epoch_end(trainer: Any) -> None:
        mlflow = _safe_mlflow()
        if mlflow is None:
            return
        try:
            box = None
            validator = getattr(trainer, 'validator', None)
            if validator is not None:
                box = (
                    getattr(validator.metrics, 'box', None)
                    if getattr(validator, 'metrics', None)
                    else None
                )
            if box is None:
                # ultralytics >=8.3 exposes a .box.maps array indexed by class
                return
            maps_50_95 = getattr(box, 'maps', None)  # array, mAP50-95 per class
            ap50 = getattr(box, 'ap50', None)  # array, mAP50 per class
            names_by_id = state['names_by_id'] or _get_class_names(trainer)
            epoch = int(getattr(trainer, 'epoch', 0))
            for arr, metric_suffix in ((maps_50_95, 'map50_95'), (ap50, 'map50')):
                if arr is None:
                    continue
                for idx, val in enumerate(arr):
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        continue
                    cname = names_by_id.get(idx, f'class_{idx}')
                    safe = cname.replace('/', '_').replace(' ', '_')[:240]
                    with contextlib.suppress(Exception):
                        mlflow.log_metric(f'class_{safe}_{metric_suffix}', float(val), step=epoch)
        except Exception:  # never break the training loop
            log.warning('on_fit_epoch_end failed:\n%s', traceback.format_exc())

    # -----------------------------------------------------------------
    # on_val_end -- confusion matrix every N epochs
    # -----------------------------------------------------------------
    def on_val_end(validator: Any) -> None:
        mlflow = _safe_mlflow()
        if mlflow is None:
            return
        try:
            trainer = getattr(validator, 'trainer', None)
            if trainer is None:
                return
            epoch = int(getattr(trainer, 'epoch', 0))
            if epoch % state['every_n_epochs_for_cm'] != 0 or epoch == 0:
                return
            cm_path = Path(validator.save_dir) / 'confusion_matrix.png'
            if cm_path.exists():
                target_name = f'confusion_matrix_epoch_{epoch}.png'
                tmp = cm_path.with_name(target_name)
                tmp.write_bytes(cm_path.read_bytes())
                mlflow.log_artifact(str(tmp))
        except Exception:  # never break the training loop
            log.warning('on_val_end failed:\n%s', traceback.format_exc())

    # -----------------------------------------------------------------
    # on_train_end -- register best.pt in MLflow Model Registry
    # -----------------------------------------------------------------
    def on_train_end(trainer: Any) -> None:
        mlflow = _safe_mlflow()
        if mlflow is None:
            return
        try:
            best = Path(trainer.save_dir) / 'weights' / 'best.pt'
            if not best.exists():
                log.warning('best.pt missing at %s; skipping registry push', best)
                return
            class_dist = state['class_dist'] or {}
            names_by_id = state['names_by_id'] or {}
            metadata = {
                'dataset_sha': str(lineage.get('dataset_sha') or ''),
                'dataset_version': str(lineage.get('dataset_version_tag') or ''),
                'class_count': len(names_by_id),
                'class_names': json.dumps([names_by_id[i] for i in sorted(names_by_id)]),
                'frozen_test_sha': str(lineage.get('frozen_test_sha') or ''),
                'test_label_sha': str(lineage.get('test_label_sha') or ''),
                'api_sha': str(lineage.get('api_sha') or ''),
                'git_sha': str(lineage.get('trainer_sha') or ''),
                'docker_digest_trainer': str(lineage.get('trainer_image_id') or ''),
                'profile': profile,
                'seed': seed,
                'run_name': run_name,
                'class_distribution': class_dist,
            }
            meta_path = Path(trainer.save_dir) / 'run_metadata.json'
            meta_path.write_text(json.dumps(metadata, indent=2))
            mlflow.log_artifact(str(meta_path))
            # Log best.pt as a generic artifact (Ultralytics' auto-MLflow
            # already logs it once; redundancy is cheap and we want it under a
            # stable path the Model Registry can point at).
            mlflow.log_artifact(str(best), artifact_path='weights')

            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id is None:
                log.warning('no active MLflow run; cannot register model')
                return
            model_uri = f'runs:/{run_id}/weights/best.pt'
            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                # Ensure the registered model exists.
                try:
                    client.get_registered_model(state['registry_name'])
                except Exception:  # "not found" has no stable type here
                    client.create_registered_model(state['registry_name'])
                mv = client.create_model_version(
                    name=state['registry_name'],
                    source=model_uri,
                    run_id=run_id,
                    tags={k: str(v) for k, v in metadata.items() if not isinstance(v, dict)},
                )
                # Auto-promote to Staging only. Production is a manual step.
                client.transition_model_version_stage(
                    name=state['registry_name'],
                    version=mv.version,
                    stage='Staging',
                    archive_existing_versions=False,
                )
                log.info(
                    'registered %s v%s in stage Staging',
                    state['registry_name'],
                    mv.version,
                )
            except Exception as exc:  # registry push is best-effort
                log.warning('model registry push failed: %s', exc)
        except Exception:  # never break the training loop
            log.warning('on_train_end failed:\n%s', traceback.format_exc())

    # Wire up
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    model.add_callback('on_val_end', on_val_end)
    model.add_callback('on_train_end', on_train_end)
    log.info('registered MLflow callbacks for run=%s', run_name)
