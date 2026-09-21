"""Materialize the dataset a single training run will train against.

Three responsibilities, all driven entirely by ``job.json``:

* **subset rewrite** -- filter + renumber labels to the requested class subset
  (:mod:`subset_dataset`), and propagate the resulting ``class_remap.json`` to
  the two places promote looks for it;
* **orientation-sensitive classes** -- resolve which class ids must suppress
  horizontal flip, by *name*, against the data.yaml the run actually uses;
* **stage-1 augmentation** -- write an augmented copy of the train split
  (:mod:`augment`) and patch ``train:`` to point at it.

The frozen export on disk is never modified; everything is written under the
job's scratch directory.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import subset_dataset as subset_mod
from logutil import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from job_protocol import JobSpec


logger = get_logger('trainer.dataset_prep')


# Class names whose content is orientation-sensitive (text, digits, signage).
# Horizontal flip is disabled for any run that trains one of them -- a mirrored
# plate/sign is not a valid example. Per-job override: ``job.json``'s
# ``augmentation.text_class_names``.
def _default_text_class_names() -> list[str]:
    raw = os.environ.get('OP_TRAIN_TEXT_CLASS_NAMES', '')
    return [tok.strip() for tok in raw.split(',') if tok.strip()]


# ---------------------------------------------------------------------------
# Dataset preparation: subset rewrite, text classes, augmentation
# ---------------------------------------------------------------------------


def _data_yaml_names(data_yaml_path: Path) -> dict[int, str]:
    """Return ``{class_id: name}`` from a data.yaml, ``{}`` on any failure."""
    try:
        import yaml

        with data_yaml_path.open('r', encoding='utf-8') as fh:
            doc = yaml.safe_load(fh) or {}
    except (ImportError, OSError, ValueError) as exc:
        logger.warning('data.yaml names read failed', path=str(data_yaml_path), error=str(exc))
        return {}
    names_raw = doc.get('names')
    if isinstance(names_raw, list):
        return dict(enumerate(str(n) for n in names_raw))
    if isinstance(names_raw, dict):
        return {int(k): str(v) for k, v in names_raw.items()}
    return {}


def resolve_text_classes(spec: JobSpec, data_yaml_path: Path) -> set[int]:
    """Resolve orientation-sensitive class ids for the *final* training id space.

    Names are matched against the data.yaml the run will actually train on,
    which sidesteps the registry-id / dense-export-id / subset-renumbered-id
    confusion entirely: whatever `data.yaml` says is the id space Ultralytics
    (and therefore the augmenter) sees.

    Sources, in order: ``job.json``'s ``augmentation.text_class_names``, then
    the ``OP_TRAIN_TEXT_CLASS_NAMES`` deployment default. Matching is
    case-insensitive. Unknown names are ignored (a subset run legitimately
    drops classes).
    """
    requested = spec.augmentation.get('text_class_names')
    names_wanted = (
        [str(n) for n in requested] if isinstance(requested, list) else _default_text_class_names()
    )
    if not names_wanted:
        return set()
    wanted_lower = {n.strip().lower() for n in names_wanted if str(n).strip()}
    return {
        cid
        for cid, name in _data_yaml_names(data_yaml_path).items()
        if name.strip().lower() in wanted_lower
    }


def _resolve_aug_config(spec: JobSpec, text_classes: set[int]) -> Any:
    """Build an :class:`augment.AugConfig` from the job's augmentation block."""
    import augment as augment_mod

    aug = spec.augmentation
    overrides_raw = aug.get('albumentations') or {}
    if not isinstance(overrides_raw, dict):
        overrides_raw = {}
    per_class_raw = aug.get('per_class_multiplier') or {}
    return augment_mod.AugConfig(
        enabled=bool(aug.get('enabled', False)),
        preset=str(aug.get('preset') or 'balanced_default'),
        multiplier=int(aug.get('multiplier') or 1),
        per_class_multiplier={int(k): int(v) for k, v in per_class_raw.items()},
        text_classes=text_classes,
        overrides=overrides_raw,
        seed=int(spec.hyperparameters.get('seed') or 42),
    )


def prepare_dataset(spec: JobSpec) -> tuple[Path, set[int]]:
    """Materialize the data.yaml this run trains against.

    Returns ``(data_yaml_path, text_classes)``. Three cases:

    * subset run -> rewrite labels under ``<tmp>/subset`` (renumbered to
      ``0..N-1``) and use that data.yaml;
    * whole-export run -> the export's own data.yaml, untouched;
    * either, plus ``augmentation.enabled`` -> an augmented copy of the train
      split is written and a patched data.yaml points ``train:`` at it. The
      original data.yaml on disk is never modified.
    """
    tmp_root = spec.tmp_root
    tmp_root.mkdir(parents=True, exist_ok=True)

    if spec.include_classes:
        print(f'[trainer] subset rewrite -> {spec.subset_dir}')
        subset_result = subset_mod.build_subset_view(
            export_dir=spec.dataset_export_dir,
            include_classes=spec.include_classes,
            single_cls=spec.single_cls,
            out_dir=spec.subset_dir,
        )
        data_yaml_path = subset_result.data_yaml
        print(
            f'[trainer]   kept_rows={subset_result.kept_rows}'
            f' dropped_rows={subset_result.dropped_rows}'
        )
    else:
        data_yaml_path = spec.dataset_export_dir / 'data.yaml'
        if not data_yaml_path.is_file():
            msg = f'export missing data.yaml at {data_yaml_path}'
            raise FileNotFoundError(msg)

    text_classes = resolve_text_classes(spec, data_yaml_path)
    if text_classes:
        print(f'[trainer]   orientation-sensitive class ids (hflip off): {sorted(text_classes)}')

    if spec.augmentation.get('enabled'):
        data_yaml_path = _build_augmented_view(spec, data_yaml_path, text_classes)

    return data_yaml_path, text_classes


def _build_augmented_view(spec: JobSpec, data_yaml_path: Path, text_classes: set[int]) -> Path:
    """Write the augmented train split and return a patched data.yaml path."""
    # Imported here, not at module scope: albumentations is a heavyweight
    # optional leg, and the watcher must still boot (and this module must still
    # import for unit tests) without it.
    import augment as augment_mod
    import yaml

    aug_cfg = _resolve_aug_config(spec, text_classes)
    aug_out = spec.tmp_root / 'aug_train'
    print(
        f'[trainer] augmentation: preset={aug_cfg.preset}'
        f' multiplier={aug_cfg.multiplier} -> {aug_out}'
    )
    # A subset run augments the *rewritten* tree (renumbered labels); a
    # whole-export run augments the export itself.
    aug_in = spec.subset_dir if spec.include_classes else spec.dataset_export_dir
    if not (aug_in / 'images' / 'train').is_dir():
        aug_in = spec.dataset_export_dir
    aug_result = augment_mod.build_augmented_dataset(in_dir=aug_in, out_dir=aug_out, config=aug_cfg)

    with data_yaml_path.open('r', encoding='utf-8') as fh:
        yaml_doc = yaml.safe_load(fh) or {}
    yaml_doc['train'] = str(aug_result.out_images_dir.resolve())
    patched_path = spec.tmp_root / 'data.aug.yaml'
    with patched_path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(yaml_doc, fh, sort_keys=False)
    print(
        f'[trainer]   images_written={aug_result.images_written}'
        f' boxes_written={aug_result.boxes_written}'
    )
    return patched_path


# ---------------------------------------------------------------------------
# class_remap propagation (trainer half of the promote fix)
# ---------------------------------------------------------------------------


def read_class_remap(spec: JobSpec) -> dict[str, Any] | None:
    """Recover ``class_remap.json`` for a subset-trained run.

    :func:`subset_dataset.build_subset_view` writes it under the per-job tmp
    dir, which ``run_job``'s ``finally`` block rmtree's -- so the manifest
    writer (which runs BEFORE cleanup) is the only window to capture it into
    ``manifest.lineage.class_remap``.
    """
    candidate = spec.subset_dir / 'class_remap.json'
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('class_remap read failed', job_id=spec.job_id, error=str(exc))
        return None
    return payload if isinstance(payload, dict) else None


def copy_class_remap_to_weights_dir(spec: JobSpec, save_dir: Path) -> bool:
    """Copy ``class_remap.json`` from the tmp subset dir next to ``best.pt``.

    Without this the file only ever lives under the per-job tmp dir, which is
    rmtree'd at job end -- leaving promote with a single source (the manifest's
    ``lineage.class_remap``, captured pre-cleanup). Writing it into
    ``<save_dir>/weights/`` gives
    :func:`src.services.training.triton_promote.resolve_class_remap` its second,
    on-disk source.

    Atomic tmp-file-then-rename. Returns ``True`` on success. A failure here is
    reported loudly (``status.class_remap_copy_failed``), never a silent skip:
    a missing remap for a subset run makes promote serve the full registry's
    ``labels.txt`` against a subset-trained model.

    A whole-export run never writes this file; that is not an error, so the
    return value is ``True`` only when the run genuinely isn't a subset run.
    """
    src = spec.subset_dir / 'class_remap.json'
    if not src.is_file():
        return not spec.is_subset_run
    try:
        weights_dir = save_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        dest = weights_dir / 'class_remap.json'
        tmp = dest.with_suffix(dest.suffix + '.tmp')
        tmp.write_bytes(src.read_bytes())
        tmp.replace(dest)
    except OSError as exc:
        logger.error(
            'class_remap.json copy to weights dir failed',
            job_id=spec.job_id,
            src=str(src),
            error=str(exc),
        )
        return False
    return True
