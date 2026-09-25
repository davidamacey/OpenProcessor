"""What a bake-off measures, as data: :class:`BakeoffProfile`.

Same pattern as :class:`src.config.DetectionProfile`: a frozen dataclass
describing one deployment's parameters, a ``from_env`` classmethod for
per-field overrides, and a small registry of named profiles. The harness
code (``run``, ``datasets``, ``bakeoff_runner``) reads everything
deployment-specific from a profile -- which eval classes to score, the
context classes a coarse->fine cascade crops on, the Triton model under
test, backend defaults, metric thresholds and plugin modules -- so
benchmarking a different domain means writing a profile, not editing the
harness. Every class present in the eval split is scored unless
``class_filter`` narrows it.

It lives next to the harness rather than under ``src/config`` on purpose:
the harness is meant to run in a standalone evaluator/trainer image, and
this module (like the rest of the harness core) is stdlib-only.

Resolution order for :func:`resolve_profile` (``spec`` is a CLI
``--profile`` value or a job spec's ``profile`` field):

1. ``None`` -> the ``OP_BAKEOFF_PROFILE`` env var if set (re-resolved as
   a spec), else :meth:`BakeoffProfile.from_env` -- the neutral
   :data:`GENERIC_PROFILE` plus any ``OP_BAKEOFF_PROFILE_*`` overrides.
2. A registered profile name (``generic``, or anything registered via
   :func:`register_profile`).
3. A path to a profile JSON file (e.g. an opt-in example profile under the
   repo's ``examples/`` tree; in containers it must be mounted there).
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


ENV_PREFIX = 'OP_BAKEOFF_PROFILE_'
GENERIC_DESCRIPTION = 'All classes present in the eval split'

# Built-in detector backends (see backends/factory.py). A profile's
# backend_modules can register more.
BACKENDS: tuple[str, ...] = ('ultralytics', 'triton', 'two-stage', 'onnxruntime', 'coreml')

# Metrics a comparison may be ranked by (keys of a comparison row's
# ``overall`` / ``common`` block).
RANKABLE_METRICS: tuple[str, ...] = ('map_50_95', 'map_50', 'precision', 'recall', 'f1')


@dataclass(frozen=True)
class BakeoffProfile:
    """Eval-class scope, cascade context, model identity and metric config.

    Attributes:
        name: Profile identifier (reports, registry key).
        description: One-line human description.
        class_filter: Eval class NAMES to score; empty = every class present
            in the eval split's test labels.
        class_names: Label space written by dataset CONVERTERS into
            ``data.yaml`` (index = class id); empty = ``('object',)``.
            Not used for scoring (eval datasets carry their own names).
        imgsz: Default detector input size (training runs override it from
            their own manifest).
        conf_floor: Detection floor (low, so COCOeval sees the full PR curve).
        nms_iou: NMS IoU during inference.
        op_conf: Operating-point confidence for P/R/F1.
        op_iou: Operating-point match IoU.
        rank_metric: Metric the per-dataset comparison is ranked by.
        default_backend: Backend used when ``--backend`` is omitted.
        triton_model: Triton model scored by ``--backend triton``.
            Deliberately empty: there is no sensible default model, so the
            triton backend requires it from the profile or ``--triton-model``.
        context_class_ids: Class ids the coarse stage keeps in ``--mode
            crop`` / ``two-stage`` (the "parent" objects the target is
            cropped from). Empty keeps every coarse-stage class.
        context_weights: Weights for the coarse-stage detector.
        context_imgsz: Coarse-stage input size.
        context_conf: Coarse-stage confidence floor.
        converter_modules: Importable modules that register extra dataset
            converters (see ``datasets.register_converter``).
        backend_modules: Importable modules that register extra detector
            backends.
        baselines_path: Optional baseline-model registry JSON for this profile.
    """

    name: str
    description: str = ''
    class_filter: tuple[str, ...] = ()
    class_names: tuple[str, ...] = ()
    imgsz: int = 640
    conf_floor: float = 0.001
    nms_iou: float = 0.7
    op_conf: float = 0.25
    op_iou: float = 0.45
    rank_metric: str = 'map_50_95'
    default_backend: str = 'ultralytics'
    triton_model: str = ''
    context_class_ids: tuple[int, ...] = ()
    context_weights: str = './weights/yolo11n.pt'
    context_imgsz: int = 960
    context_conf: float = 0.25
    converter_modules: tuple[str, ...] = ()
    backend_modules: tuple[str, ...] = ()
    baselines_path: str = ''

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('BakeoffProfile.name must be non-empty')
        # A plugin backend is registered only once its module is imported, so
        # the default can only be checked against the built-ins when the
        # profile brings no backend modules.
        if not self.backend_modules and self.default_backend not in BACKENDS:
            raise ValueError(
                f'default_backend {self.default_backend!r} not one of {", ".join(BACKENDS)}'
            )
        if self.rank_metric not in RANKABLE_METRICS:
            raise ValueError(
                f'rank_metric {self.rank_metric!r} not one of {", ".join(RANKABLE_METRICS)}'
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BakeoffProfile:
        """Build from a JSON-shaped mapping; unknown keys are an error."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in data if k not in known and not k.startswith('_'))
        if unknown:
            raise ValueError(f'unknown BakeoffProfile field(s): {", ".join(unknown)}')
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if isinstance(value, list):
                value = tuple(value)
            kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: Path) -> BakeoffProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_env(cls, prefix: str = ENV_PREFIX, *, name: str = 'generic') -> BakeoffProfile:
        """Build from ``{prefix}*`` env vars over the dataclass defaults.

        Mirrors :meth:`src.config.DetectionProfile.from_env`: ints/floats
        via their constructors, tuples from comma-separated lists (``int``
        elements for ``context_class_ids``; ``str`` for ``class_filter``,
        ``class_names``, ``converter_modules``, ``backend_modules``),
        everything else as a string.
        """
        overrides: dict[str, Any] = {'name': os.environ.get(f'{prefix}NAME', name)}
        if overrides['name'] == 'generic':
            overrides['description'] = GENERIC_DESCRIPTION
        for f in fields(cls):
            if f.name == 'name':
                continue
            raw = os.environ.get(f'{prefix}{f.name.upper()}')
            if raw is None:
                continue
            default = f.default
            if f.name == 'context_class_ids':
                overrides[f.name] = tuple(int(p) for p in raw.split(',') if p.strip())
            elif isinstance(default, tuple):
                overrides[f.name] = tuple(p.strip() for p in raw.split(',') if p.strip())
            elif isinstance(default, int):
                overrides[f.name] = int(raw)
            elif isinstance(default, float):
                overrides[f.name] = float(raw)
            else:
                overrides[f.name] = raw
        return cls(**overrides)


GENERIC_PROFILE = BakeoffProfile(name='generic', description=GENERIC_DESCRIPTION)

_REGISTRY: dict[str, BakeoffProfile] = {GENERIC_PROFILE.name: GENERIC_PROFILE}


def register_profile(profile: BakeoffProfile, *, replace: bool = False) -> None:
    """Register ``profile`` under its name (error on a clash unless ``replace``)."""
    if profile.name in _REGISTRY and not replace:
        raise ValueError(f'bake-off profile {profile.name!r} already registered')
    _REGISTRY[profile.name] = profile


def registered_profiles() -> dict[str, BakeoffProfile]:
    return dict(_REGISTRY)


def resolve_profile(spec: str | None = None) -> BakeoffProfile:
    """Resolve a profile spec (see module docstring for the order)."""
    if spec is None or not str(spec).strip():
        env_spec = os.environ.get('OP_BAKEOFF_PROFILE', '').strip()
        return resolve_profile(env_spec) if env_spec else BakeoffProfile.from_env()
    spec = str(spec).strip()
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    path = Path(spec)
    if path.suffix == '.json' and path.is_file():
        return BakeoffProfile.from_json_file(path)
    known = sorted(_REGISTRY)
    raise ValueError(
        f'unknown bake-off profile {spec!r}: not a registered name ({", ".join(known)}) '
        'or an existing profile .json file'
    )


def load_converter_plugins(profile: BakeoffProfile) -> None:
    """Import the profile's converter modules so they self-register."""
    for module in profile.converter_modules:
        importlib.import_module(module)


def resolve_baselines_path(profile: BakeoffProfile, default: Path) -> Path:
    """The profile's baseline registry (relative paths resolve against the repo root).

    In a container the repo root is ``/app``: an example profile's relative
    ``baselines_path`` (e.g. ``examples/...``) needs ``examples/`` mounted at
    ``/app/examples``.
    """
    if not profile.baselines_path:
        return default
    path = Path(profile.baselines_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path


__all__ = [
    'BACKENDS',
    'GENERIC_PROFILE',
    'RANKABLE_METRICS',
    'BakeoffProfile',
    'load_converter_plugins',
    'register_profile',
    'registered_profiles',
    'resolve_baselines_path',
    'resolve_profile',
]
