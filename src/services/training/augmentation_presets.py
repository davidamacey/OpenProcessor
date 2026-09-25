"""The augmentation preset catalog — the one list of preset ids.

Both sides of the job-file protocol read this file:

* the API validates ``augmentation.preset`` against it (training preflight
  and ``/train/start``, before any GPU claim) and serves it on
  ``GET /train/augmentation_presets``;
* the trainer image copies it flat next to ``augment.py``
  (``docker/trainer/Dockerfile``), which builds ``PRESETS`` from these ids
  and looks up a ``preset_<id>`` factory for each.

Standard library only: the trainer image ships this single file without
the rest of ``src/``. Adding a preset = one entry here plus its
``preset_<id>`` factory in ``docker/trainer/augment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationPreset:
    """One selectable preset."""

    id: str
    label: str
    description: str
    # Horizontal flip stays off for the whole run with this preset: a
    # mirrored text-bearing target is not a valid training example.
    orientation_sensitive: bool = False


AUGMENTATION_PRESETS: tuple[AugmentationPreset, ...] = (
    AugmentationPreset(
        'none',
        'None',
        'No pixel or box changes; train images are copied through as-is.',
    ),
    AugmentationPreset(
        'balanced_default',
        'Balanced (default)',
        'Broad, mild coverage: small rotation and affine shifts, brightness, '
        'contrast and colour jitter, occasional shadows.',
    ),
    AugmentationPreset(
        'outdoor_scene',
        'Outdoor scene',
        'Outdoor daytime capture: perspective, strong brightness/contrast, rain, fog '
        'and sun glare.',
    ),
    AugmentationPreset(
        'heavy_tilt',
        'Heavy tilt',
        'Strong rotation (up to 25 degrees), shear and scale for subjects that often '
        'appear off-axis.',
    ),
    AugmentationPreset(
        'low_light',
        'Low light',
        'Night and low-light cameras: gamma, sensor noise, compression and darkening.',
    ),
    AugmentationPreset(
        'text_targets',
        'Text targets',
        'Small text-bearing targets (signage, labels, serial numbers): perspective, '
        'motion blur, compression, occlusion. Never flips horizontally.',
        orientation_sensitive=True,
    ),
    AugmentationPreset(
        'text_targets_aggressive',
        'Text targets (aggressive)',
        'Stronger version of Text targets for single-class runs. Never flips horizontally.',
        orientation_sensitive=True,
    ),
)

DEFAULT_AUGMENTATION_PRESET = 'balanced_default'

PRESET_IDS: tuple[str, ...] = tuple(p.id for p in AUGMENTATION_PRESETS)

ORIENTATION_SENSITIVE_PRESET_IDS: frozenset[str] = frozenset(
    p.id for p in AUGMENTATION_PRESETS if p.orientation_sensitive
)


def unknown_preset_error(preset: str) -> str | None:
    """``None`` for a known preset id, else the error message naming the valid ones."""
    if preset in PRESET_IDS:
        return None
    return f'unknown augmentation preset {preset!r}; valid presets: {", ".join(PRESET_IDS)}'


__all__ = [
    'AUGMENTATION_PRESETS',
    'DEFAULT_AUGMENTATION_PRESET',
    'ORIENTATION_SENSITIVE_PRESET_IDS',
    'PRESET_IDS',
    'AugmentationPreset',
    'unknown_preset_error',
]
