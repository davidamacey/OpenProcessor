"""YOLO26 training profiles + class-subset presets.

Ported from a private reference vehicle/license-plate curation stack's
training pipeline (confirmed generic including the hyperparameter
*values* -- see ``docs/design/curation_design_rationale.md`` for the
genericization approach). This module is the **single source of truth** for the
hyperparameter-profile table that a frontend renders on the train form
and the trainer container consumes via ``job.json``. Keeping it on the
backend means a profile tweak ships in one place, and the frontend
picks it up via ``GET /curation/train/profiles``.

Why YOLO26 only:
    YOLO26 is the official Ultralytics recipe (YOLO Vision 2025) and the
    only family the reference trainer container supports.

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

NOTE (oss port): ``CLASS_SUBSET_PRESETS`` below still names the
reference dataset's vehicle/license-plate class vocabulary
(``license_plate``, ``cars``, ``cruisers`` ...). Per the port directive
for this chunk the whole file lands verbatim, values included; a
dataset-agnostic preset table is a follow-up, not part of this port.
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
# Presets are server-side constants surfaced via ``GET /curation/train/presets``.
# The frontend renders them as one-click buttons in the class picker. The
# ``selector`` field tells the API how to materialize the class list at
# preflight time:
#
#   - ``all``                   -> every class in the registry
#   - ``all_except: [name, ...]`` -> registry minus the named classes
#   - ``names: [name, ...]``      -> just the named classes
#   - ``groups: [name, ...]``     -> every class whose ``group`` is in the list
#
# The actual class_id list is resolved at preflight time (the registry
# is dynamic). Hard-coding ids here would silently drift.

CLASS_SUBSET_PRESETS: list[dict[str, Any]] = [
    {
        'name': 'all_vehicles',
        'label': 'All vehicles',
        'description': 'Every class except license_plate. Default detection model.',
        'selector': {'kind': 'all_except', 'names': ['license_plate']},
    },
    {
        'name': 'plates_only',
        'label': 'Plates only',
        'description': 'Single-class plate detector for OCR pre-step.',
        'selector': {'kind': 'names', 'names': ['license_plate']},
        'single_cls_default': True,
    },
    {
        'name': 'vehicles_and_plates',
        'label': 'Vehicles + plates',
        'description': 'Every class. Multi-class production model (recommended default).',
        'selector': {'kind': 'all'},
    },
    {
        'name': 'cars_only',
        'label': 'Cars only',
        'description': 'Specialized cars-only model (group=cars).',
        'selector': {'kind': 'groups', 'groups': ['cars']},
    },
    {
        'name': 'bikes_only',
        'label': 'Bikes only',
        'description': 'Specialized bikes-only model (cruisers, sportbikes, dirtbikes).',
        'selector': {'kind': 'groups', 'groups': ['cruisers', 'sportbikes', 'dirtbikes']},
    },
]


def get_class_subset_presets() -> list[dict[str, Any]]:
    """Return the class-subset preset table for the frontend."""
    return [dict(p) for p in CLASS_SUBSET_PRESETS]
