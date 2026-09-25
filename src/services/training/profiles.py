"""YOLO26 training profiles + class-subset presets.

This module is the **single source of truth** for the
hyperparameter-profile table that a frontend renders on the train form
and the trainer container consumes via ``job.json``. Keeping it on the
backend means a profile tweak ships in one place, and the frontend
picks it up via ``GET /curation/train/profiles``.

Why YOLO26 only:
    YOLO26 is the official Ultralytics recipe (YOLO Vision 2025) and the
    only family the trainer container supports.

Critical hyperparameter notes:
    - ``optimizer="MuSGD"`` is **required**. Submitting ``optimizer="auto"``
      silently overrides MuSGD with AdamW for runs <10k iterations
      (Ultralytics issue #23696). The router rejects that combo before
      writing ``job.json``.
    - The profile's ``model_size`` is a *default* -- the user can pick a
      different size on the form (e.g. medium profile, ``s`` weights).
    - YOLO26 is anchor-free, so no anchor config.
    - ``patience`` is a cap on the early-stopping window, tighter for
      small models (which converge faster) than for large ones. Set
      ``patience=0`` to disable early stopping.

Class-subset presets: ``get_class_subset_presets()``
is generic -- ``all`` is always served; ``all_except_region`` /
``region_only`` only appear when the active region profile names a
``region_class_name``; a deployment appends its own presets via
``OP_TRAIN_PRESETS_PATH`` (a JSON list of the same shape).
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# Hyperparameter profile table -- YOLO26 (Ultralytics YOLO Vision 2025 recipe)
# =============================================================================
#
# Each entry is a dict of Ultralytics-compatible keyword args. The form on
# the frontend renders this table and lets the user diff against any
# field. ``model_size`` here is the *default* but is independently
# overridable on the form.

PROFILES_YOLO26: dict[str, dict[str, Any]] = {
    'probe': {
        'model_size': 'n',
        'epochs': 20,
        'patience': 10,
        'batch': 64,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.005,
        'momentum': 0.947,
        'weight_decay': 0.00064,
    },
    'nano': {
        'model_size': 'n',
        'epochs': 245,
        'patience': 20,
        'batch': 64,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.0054,
        'momentum': 0.947,
        'weight_decay': 0.00064,
        'mosaic': 0.909,
        'mixup': 0.012,
        'copy_paste': 0.075,
    },
    'small': {
        'model_size': 's',
        'epochs': 70,
        'patience': 20,
        'batch': 48,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.00038,
        'momentum': 0.948,
        'weight_decay': 0.00027,
        'mosaic': 0.992,
        'mixup': 0.05,
        'copy_paste': 0.404,
    },
    'medium': {
        'model_size': 'm',
        'epochs': 80,
        'patience': 30,
        'batch': 32,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.00038,
        'momentum': 0.948,
        'weight_decay': 0.00027,
        'mosaic': 0.992,
        'mixup': 0.427,
        'copy_paste': 0.304,
    },
    'large': {
        'model_size': 'l',
        'epochs': 60,
        'patience': 40,
        'batch': 16,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.00038,
        'momentum': 0.948,
        'weight_decay': 0.00027,
        'mosaic': 0.992,
        'mixup': 0.427,
        'copy_paste': 0.404,
    },
    'xlarge': {
        'model_size': 'x',
        'epochs': 40,
        'patience': 50,
        'batch': 8,
        'imgsz': 640,
        'optimizer': 'MuSGD',
        'lr0': 0.00038,
        'momentum': 0.948,
        'weight_decay': 0.00027,
        'mosaic': 0.992,
        'mixup': 0.427,
        'copy_paste': 0.404,
    },
}


# Free-form descriptions surfaced next to each profile button on the form.
PROFILE_DESCRIPTIONS: dict[str, str] = {
    'probe': 'Quick smoke run (~20 epochs, nano weights) — verifies a job submits and trains',
    'nano': 'Long nano run (245 epochs) for production-grade fastest model',
    'small': '70 epochs, small weights — balanced accuracy / speed',
    'medium': '80 epochs, medium weights — recommended default for most jobs',
    'large': '60 epochs, large weights — slower, higher accuracy',
    'xlarge': '40 epochs, xlarge weights — best accuracy, longest wall-time',
}


# Optimizers that must be rejected at submission time. ``auto`` silently
# trades MuSGD for AdamW on short runs (Ultralytics #23696); we reject it
# rather than let the user trip the bug.
RESERVED_OPTIMIZERS_YOLO26: frozenset[str] = frozenset({'auto'})


def get_profiles() -> list[dict[str, Any]]:
    """Return the profile table as a list of dicts (frontend friendly).

    Each entry has the shape:
        {
          "name": "medium",
          "description": "80 epochs, medium weights — ...",
          "defaults": { ... hyperparameters ... }
        }
    """
    return [
        {
            'name': name,
            'description': PROFILE_DESCRIPTIONS.get(name, ''),
            'defaults': dict(defaults),
        }
        for name, defaults in PROFILES_YOLO26.items()
    ]


# =============================================================================
# Class-subset presets
# =============================================================================
#
# Presets are surfaced via ``GET /curation/train/presets``. The frontend
# renders them as one-click buttons in the class picker. The ``selector``
# field tells the API how to materialize the class list at preflight
# time:
#
#   - ``all``                   -> every class in the registry
#   - ``all_except: [name, ...]`` -> registry minus the named classes
#   - ``names: [name, ...]``      -> just the named classes
#   - ``groups: [name, ...]``     -> every class whose ``group`` is in the list
#
# The actual class_id list is resolved at preflight time (the registry
# is dynamic). Hard-coding ids here would silently drift.

_REQUIRED_PRESET_KEYS = frozenset({'name', 'label', 'description', 'selector'})
_OPTIONAL_PRESET_KEYS = frozenset({'single_cls_default'})


def _validate_preset_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        msg = f'OP_TRAIN_PRESETS_PATH entry must be an object, got {entry!r}'
        raise ValueError(msg)
    missing = _REQUIRED_PRESET_KEYS - set(entry)
    if missing:
        msg = (
            f'OP_TRAIN_PRESETS_PATH preset {entry!r} is missing required key(s): {sorted(missing)}'
        )
        raise ValueError(msg)
    unknown = set(entry) - _REQUIRED_PRESET_KEYS - _OPTIONAL_PRESET_KEYS
    if unknown:
        msg = f'OP_TRAIN_PRESETS_PATH preset {entry!r} has unknown key(s): {sorted(unknown)}'
        raise ValueError(msg)
    return dict(entry)


def _extra_presets_from_env() -> list[dict[str, Any]]:
    """Deployment-supplied presets from ``OP_TRAIN_PRESETS_PATH`` (a JSON
    list of preset objects, same shape as the built-ins). A malformed
    file raises rather than being silently skipped."""
    import json
    import os
    from pathlib import Path

    path = os.environ.get('OP_TRAIN_PRESETS_PATH', '').strip()
    if not path:
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        msg = f'OP_TRAIN_PRESETS_PATH={path!r} could not be read as JSON: {exc}'
        raise ValueError(msg) from exc
    if not isinstance(raw, list):
        msg = f'OP_TRAIN_PRESETS_PATH={path!r} must contain a JSON list'
        raise ValueError(msg)
    return [_validate_preset_entry(entry) for entry in raw]


def get_class_subset_presets() -> list[dict[str, Any]]:
    """Return the class-subset preset table for the frontend.

    Always includes ``all`` (every class in the registry). When the
    active region profile (``src.services.detection.profile_registry``)
    names a non-empty ``region_class_name``, also includes
    ``all_except_region`` and ``region_only`` (the single-class-detector
    preset, ``single_cls_default: True``). Deployment-specific presets
    from ``OP_TRAIN_PRESETS_PATH`` are appended last.
    """
    from src.services.detection.profile_registry import get_active_region_profile

    presets: list[dict[str, Any]] = [
        {
            'name': 'all',
            'label': 'All classes',
            'description': 'Every class in the registry.',
            'selector': {'kind': 'all'},
        }
    ]

    profile = get_active_region_profile()
    region_class_name = profile.region_class_name if profile else ''
    if region_class_name:
        label = profile.display_name or region_class_name if profile else region_class_name
        presets.append(
            {
                'name': 'all_except_region',
                'label': f'All except {label}',
                'description': f'Every class except {region_class_name}.',
                'selector': {'kind': 'all_except', 'names': [region_class_name]},
            }
        )
        presets.append(
            {
                'name': 'region_only',
                'label': label,
                'description': f'Single-class {region_class_name} detector.',
                'selector': {'kind': 'names', 'names': [region_class_name]},
                'single_cls_default': True,
            }
        )

    presets.extend(_extra_presets_from_env())
    return presets
