"""Albumentations-based on-disk augmentation for the trainer container.

Stage-1 augmentation: produces a multiplied + photometrically/geometrically
perturbed copy of the train split *before* training begins. Ultralytics still
runs its own per-batch augmentation (mosaic, mixup, copy-paste, HSV jitter) on
top -- the two layers compound, giving the model exposure to weather, glare,
sensor noise, and viewpoint variation that Ultralytics' built-ins don't cover.

Validation and test splits are NEVER augmented -- they must stay identical
across runs so metrics are comparable.

Public API:

* :func:`build_augmented_dataset` -- read train images + labels, write augmented
  copies to ``out_dir/{images,labels}``.
* :func:`compute_auto_balance` -- compute per-class multipliers to lift
  under-represented classes toward a target count.
* :data:`PRESETS` -- dict of preset name -> factory ``(hflip) -> A.Compose``,
  built from the preset catalog (``augmentation_presets``: a flat copy of
  ``src/services/training/augmentation_presets.py``, the one list of preset
  ids the API validates and serves). Each catalog id needs a
  ``preset_<id>`` factory here.

Preset names are deliberately domain-neutral scene/condition descriptors: the
job spec (``job.json``'s ``augmentation.preset``, see
``src.services.training.jobs.AugmentationSpec``) picks one, and a deployment
that needs a different mix supplies ``augmentation.albumentations`` overrides
rather than forking this file.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Albumentations is a hard dep of the trainer image. Importing at module load
# makes a broken install show up when the augmentation leg is first touched
# rather than mid-run. ``trainer.py`` imports this module lazily so the watcher
# still boots (and unit-tests still import) without the wheel present.
# Lower-case alias 'A' is the upstream project's documented convention; ruff
# N812 is silenced.
import albumentations as A  # noqa: N812
import cv2
from augmentation_presets import (
    AUGMENTATION_PRESETS,
    DEFAULT_AUGMENTATION_PRESET,
    ORIENTATION_SENSITIVE_PRESET_IDS,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


log = logging.getLogger('augment')


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class AugConfig:
    """Resolved augmentation config for a single training run."""

    enabled: bool = False
    preset: str = DEFAULT_AUGMENTATION_PRESET
    multiplier: int = 1
    per_class_multiplier: dict[int, int] = field(default_factory=dict)
    # Class ids (in the *final* training id space) whose content is
    # orientation-sensitive -- text, digits, signage. Any of these present
    # disables horizontal flip for the whole run, since a mirrored plate /
    # sign / serial number is not a valid training example.
    text_classes: set[int] = field(default_factory=set)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 42


@dataclass
class AugResult:
    """Return value of :func:`build_augmented_dataset`."""

    out_images_dir: Path
    out_labels_dir: Path
    images_written: int
    boxes_written: int
    boxes_dropped_invisible: int
    config_path: Path
    samples_log: list[Path]
    per_class_image_counts: dict[int, int]


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


def _bbox_params() -> A.BboxParams:
    """YOLO-format BboxParams with ``min_visibility=0.3``."""
    return A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels'])


def preset_none(hflip: bool = True) -> A.Compose:
    """Identity transform (still re-encodes through Albumentations for
    consistency, but doesn't modify pixels or boxes)."""
    _ = hflip
    return A.Compose([A.NoOp()], bbox_params=_bbox_params())


def preset_balanced_default(hflip: bool = True) -> A.Compose:
    """Broad, mild coverage suitable as a default."""
    transforms: list[Any] = [
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), shear=(-3, 3), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.RandomShadow(p=0.2),
    ]
    if hflip:
        transforms.append(A.HorizontalFlip(p=0.3))
    return A.Compose(transforms, bbox_params=_bbox_params())


def preset_outdoor_scene(hflip: bool = True) -> A.Compose:
    """Outdoor daytime capture -- weather and glare emphasis."""
    transforms: list[Any] = [
        A.Rotate(limit=5, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.RandomRain(p=0.2),
        A.RandomFog(p=0.15),
        A.RandomSunFlare(p=0.15),
    ]
    if hflip:
        transforms.append(A.HorizontalFlip(p=0.3))
    return A.Compose(transforms, bbox_params=_bbox_params())


def preset_heavy_tilt(hflip: bool = True) -> A.Compose:
    """Heavy rotation/shear for subjects that routinely appear off-axis."""
    transforms: list[Any] = [
        A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(scale=(0.85, 1.15), shear=(-10, 10), p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomShadow(p=0.3),
    ]
    if hflip:
        transforms.append(A.HorizontalFlip(p=0.3))
    return A.Compose(transforms, bbox_params=_bbox_params())


def preset_low_light(hflip: bool = True) -> A.Compose:
    """Night / low-light camera datasets."""
    transforms: list[Any] = [
        A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.RandomGamma(gamma_limit=(50, 150), p=0.6),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        A.ImageCompression(quality_range=(50, 80), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.1), contrast_limit=0.3, p=0.6),
    ]
    if hflip:
        transforms.append(A.HorizontalFlip(p=0.3))
    return A.Compose(transforms, bbox_params=_bbox_params())


def preset_text_targets(hflip: bool = False) -> A.Compose:
    """Small, text-bearing targets (plates, signage, labels, serials).

    ``hflip`` is accepted for signature symmetry and always ignored: mirroring
    a text-bearing target produces an invalid training example.
    """
    _ = hflip
    transforms: list[Any] = [
        A.Perspective(scale=(0.05, 0.10), p=0.4),
        A.Rotate(limit=8, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.ImageCompression(quality_range=(60, 90), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            p=0.3,
        ),
    ]
    return A.Compose(transforms, bbox_params=_bbox_params())


def preset_text_targets_aggressive(hflip: bool = False) -> A.Compose:
    """Aggressive variant of :func:`preset_text_targets`, for single-class runs."""
    _ = hflip
    transforms: list[Any] = [
        A.Perspective(scale=(0.10, 0.15), p=0.5),
        A.Affine(rotate=(-20, 20), scale=(0.85, 1.15), p=0.6),
        A.MotionBlur(blur_limit=9, p=0.4),
        A.ISONoise(p=0.4),
        A.ImageCompression(quality_range=(50, 85), p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.05, 0.20),
            hole_width_range=(0.05, 0.20),
            p=0.4,
        ),
    ]
    return A.Compose(transforms, bbox_params=_bbox_params())


# A catalog id with no ``preset_<id>`` factory fails here, at import, rather
# than mid-run.
PRESETS: dict[str, Callable[[bool], A.Compose]] = {
    p.id: globals()[f'preset_{p.id}'] for p in AUGMENTATION_PRESETS
}

# Presets that are inherently orientation-sensitive: horizontal flip stays off
# for these regardless of which classes the run includes.
NO_HFLIP_PRESETS = ORIENTATION_SENSITIVE_PRESET_IDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_label_file(path: Path) -> tuple[list[list[float]], list[int]]:
    """Read a YOLO label file. Returns ``(bboxes, class_labels)``.

    bboxes are normalized [cx, cy, w, h] floats; class_labels are int class IDs.
    """
    bboxes: list[list[float]] = []
    cids: list[int] = []
    if not path.is_file():
        return bboxes, cids
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        bboxes.append([cx, cy, w, h])
        cids.append(cid)
    return bboxes, cids


def _write_label_file(path: Path, bboxes: list[list[float]], cids: list[int]) -> None:
    lines = [
        f'{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}'
        for cid, b in zip(cids, bboxes, strict=False)
    ]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _multiplier_for_image(
    classes_in_image: set[int],
    per_class_multiplier: dict[int, int],
    default_multiplier: int,
) -> int:
    """Pick the max multiplier across the classes present in this image."""
    if not classes_in_image:
        return default_multiplier
    relevant = [per_class_multiplier[c] for c in classes_in_image if c in per_class_multiplier]
    if not relevant:
        return default_multiplier
    return max([default_multiplier, *relevant])


def _resolve_pipeline(config: AugConfig) -> A.Compose:
    """Build the Compose pipeline from preset + overrides.

    Overrides take the form ``{TransformName: {kwarg: val}}`` and *add* that
    transform onto the preset. Unknown overrides are logged and ignored.
    """
    preset_factory = PRESETS.get(config.preset)
    if preset_factory is None:
        msg = f'unknown augmentation preset: {config.preset!r}'
        raise ValueError(msg)

    # Disable HFlip when orientation-sensitive classes are in the label set OR
    # the preset is itself orientation-sensitive.
    hflip_safe = not config.text_classes and config.preset not in NO_HFLIP_PRESETS
    pipeline = preset_factory(hflip_safe)

    if not config.overrides:
        return pipeline

    extra: list[Any] = []
    for name, kwargs in config.overrides.items():
        cls = getattr(A, name, None)
        if cls is None:
            log.warning('augment: unknown override transform %r -- skipping', name)
            continue
        try:
            extra.append(cls(**kwargs))
        except Exception as exc:  # a bad override must not kill the run
            log.warning('augment: bad kwargs for %s (%r): %s -- skipping', name, kwargs, exc)
    if not extra:
        return pipeline
    return A.Compose([*pipeline.transforms, *extra], bbox_params=_bbox_params())


# ---------------------------------------------------------------------------
# Public -- auto-balance
# ---------------------------------------------------------------------------


def compute_auto_balance(
    class_counts: dict[int, int],
    target_count: int,
    max_multiplier: int,
) -> dict[int, int]:
    """Compute per-class multipliers that lift rare classes toward a target.

    For each class with ``count < target_count``, set its multiplier to
    ``min(max_multiplier, ceil(target_count / count))``. Classes already at or
    above the target get multiplier 1. ``count == 0`` classes are skipped (no
    augmentation can resurrect a missing class).
    """
    if target_count <= 0:
        return {}
    out: dict[int, int] = {}
    for cid, count in class_counts.items():
        if count <= 0:
            continue
        if count >= target_count:
            out[cid] = 1
            continue
        out[cid] = min(max_multiplier, math.ceil(target_count / count))
    return out


# ---------------------------------------------------------------------------
# Public -- build_augmented_dataset
# ---------------------------------------------------------------------------


def build_augmented_dataset(
    in_dir: Path,
    out_dir: Path,
    config: AugConfig,
) -> AugResult:
    """Stream train images + labels through the pipeline and write copies.

    Args:
        in_dir: Directory containing ``images/train/`` and ``labels/train/``
            (Ultralytics layout). Only the train split is augmented.
        out_dir: Destination root. Created if missing. Output:
            ``out_dir/images/`` and ``out_dir/labels/``.
        config: :class:`AugConfig`. ``config.text_classes`` auto-disables HFlip
            when any class in the label set is orientation-sensitive.

    Returns:
        :class:`AugResult` describing the rewrite.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    src_images = in_dir / 'images' / 'train'
    src_labels = in_dir / 'labels' / 'train'
    if not src_images.is_dir():
        msg = f'missing train images dir: {src_images}'
        raise FileNotFoundError(msg)
    if not src_labels.is_dir():
        msg = f'missing train labels dir: {src_labels}'
        raise FileNotFoundError(msg)

    out_images = out_dir / 'images'
    out_labels = out_dir / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    pipeline = _resolve_pipeline(config)

    rng = random.Random(config.seed)  # augmentation jitter, not crypto

    images_written = 0
    boxes_written = 0
    boxes_dropped = 0
    samples_log: list[Path] = []
    per_class_counts: dict[int, int] = {}

    # Up to 12 sample images, spread across classes where possible, logged as
    # MLflow artifacts so an operator can eyeball what the pipeline produced.
    samples_target = 12
    classes_seen_in_samples: set[int] = set()

    for img_path in _iter_images(src_images):
        label_path = src_labels / f'{img_path.stem}.txt'
        bboxes, cids = _parse_label_file(label_path)
        classes_in_image = set(cids)
        n_copies = _multiplier_for_image(
            classes_in_image, config.per_class_multiplier, config.multiplier
        )
        if n_copies <= 0:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            log.warning('augment: cv2 failed to load %s -- skipping', img_path)
            continue

        for copy_idx in range(n_copies):
            seed = rng.randint(0, 2**31 - 1)
            random.seed(seed)
            # Albumentations 2.x picks up the global random state; we reset
            # before each call so the same input image + same `seed` is
            # deterministic between invocations.
            try:
                result = pipeline(image=img, bboxes=bboxes, class_labels=cids)
            except Exception as exc:  # one bad image must not kill the run
                log.warning('augment: pipeline failed for %s copy %d: %s', img_path, copy_idx, exc)
                continue

            aug_img = result['image']
            aug_bboxes = list(result['bboxes'])
            aug_cids = list(result['class_labels'])

            # Albumentations may drop boxes whose visibility falls below
            # min_visibility -- track that for the manifest.
            boxes_dropped += max(0, len(bboxes) - len(aug_bboxes))

            stem = f'{img_path.stem}_aug{copy_idx:02d}'
            out_img = out_images / f'{stem}{img_path.suffix.lower() or ".jpg"}'
            out_lbl = out_labels / f'{stem}.txt'
            cv2.imwrite(str(out_img), aug_img)
            _write_label_file(out_lbl, [list(b) for b in aug_bboxes], aug_cids)

            images_written += 1
            boxes_written += len(aug_bboxes)
            for cid in aug_cids:
                per_class_counts[cid] = per_class_counts.get(cid, 0) + 1

            if len(samples_log) < samples_target:
                novel_class = bool(set(aug_cids) - classes_seen_in_samples)
                if novel_class or len(samples_log) < samples_target // 2:
                    samples_log.append(out_img)
                    classes_seen_in_samples.update(aug_cids)

    if images_written == 0:
        msg = (
            'augment: 0 images written -- check multiplier, per_class_multiplier, '
            'and that the train split has any labeled images'
        )
        raise RuntimeError(msg)

    # Persist resolved config alongside the data so MLflow can log it.
    config_payload = {
        'enabled': config.enabled,
        'preset': config.preset,
        'multiplier': config.multiplier,
        'per_class_multiplier': {str(k): v for k, v in config.per_class_multiplier.items()},
        'text_classes': sorted(config.text_classes),
        'overrides': config.overrides,
        'seed': config.seed,
        'results': {
            'images_written': images_written,
            'boxes_written': boxes_written,
            'boxes_dropped_invisible': boxes_dropped,
            'per_class_image_counts': {str(k): v for k, v in per_class_counts.items()},
        },
    }
    config_path = out_dir / 'augmentation_config.json'
    config_path.write_text(json.dumps(config_payload, indent=2), encoding='utf-8')

    log.info(
        'augment: images_written=%d boxes_written=%d boxes_dropped=%d preset=%s',
        images_written,
        boxes_written,
        boxes_dropped,
        config.preset,
    )
    return AugResult(
        out_images_dir=out_images,
        out_labels_dir=out_labels,
        images_written=images_written,
        boxes_written=boxes_written,
        boxes_dropped_invisible=boxes_dropped,
        config_path=config_path,
        samples_log=samples_log,
        per_class_image_counts=per_class_counts,
    )


def _iter_images(src_images: Path) -> Iterator[Path]:
    for ext in IMG_EXTS:
        yield from src_images.glob(f'*{ext}')


# ---------------------------------------------------------------------------
# MLflow hook (called by trainer.py after augmentation completes)
# ---------------------------------------------------------------------------


def log_augmentation_to_mlflow(result: AugResult) -> None:
    """Log the augmentation config + sample images to the active MLflow run.

    Best-effort: failures are swallowed so a flaky tracking server can't kill a
    training run.
    """
    try:
        import mlflow
    except Exception as exc:  # MLflow is optional at runtime
        log.warning('augment: mlflow unavailable, skipping log: %s', exc)
        return
    try:
        mlflow.log_artifact(str(result.config_path), artifact_path='augmentation')
        for sample in result.samples_log:
            mlflow.log_artifact(str(sample), artifact_path='augmentation/samples')
    except Exception as exc:  # tracking hiccup must not kill the run
        log.warning('augment: mlflow log failed: %s', exc)
