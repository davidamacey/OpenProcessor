"""The open-vocabulary prompt set: typed decode of a stored ``open_vocab`` body.

One decoder for every consumer (validator, schema, the runner, the per-image
test route), so the field names, defaults and types live here only. A body
that does not decode raises :class:`ValueError` naming the offending path; a
typo must fail loudly, never fall back to a default.

A target's ``class_name`` is a registry class NAME; an empty one is discovery
mode (the hit is stored as an unlabeled proposal named by its prompt).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from src.services.detection.open_vocab_select import TargetRules


DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_AREA_FRAC = 0.0005
DEFAULT_MAX_AREA_FRAC = 0.9
DEFAULT_MAX_INSTANCES = 20
DEFAULT_IMAGE_MAX_SIDE = 1024
DEFAULT_DEDUP_IOU = 0.5
DEFAULT_MAX_ENABLED_TARGETS = 8
#: Hard ceiling on ``max_enabled_targets``: cost is linear in targets.
MAX_ENABLED_TARGETS_CEILING = 32


@dataclass(frozen=True)
class OpenVocabTarget:
    prompt: str = ''
    class_name: str = ''
    min_score: float = DEFAULT_MIN_SCORE
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC
    max_instances: int = DEFAULT_MAX_INSTANCES
    parent_classes: tuple[str, ...] = ()
    enabled: bool = True
    mask: bool = True

    def rules(self) -> TargetRules:
        return TargetRules(
            prompt=self.prompt,
            class_name=self.class_name,
            min_score=self.min_score,
            min_area_frac=self.min_area_frac,
            max_area_frac=self.max_area_frac,
            max_instances=self.max_instances,
        )


@dataclass(frozen=True)
class HitRateGate:
    enabled: bool = False
    window: int = 20
    miss_threshold: int = 15
    sample_floor: float = 0.1


@dataclass(frozen=True)
class GatingConfig:
    tier2_vlm_precheck: bool = False
    tier3_hit_rate: HitRateGate = field(default_factory=HitRateGate)


@dataclass(frozen=True)
class OpenVocabSet:
    name: str
    display_name: str = ''
    targets: tuple[OpenVocabTarget, ...] = ()
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    dedup_iou: float = DEFAULT_DEDUP_IOU
    run_on_ingest: bool = False
    max_enabled_targets: int = DEFAULT_MAX_ENABLED_TARGETS
    gating: GatingConfig = field(default_factory=GatingConfig)

    @property
    def enabled_targets(self) -> tuple[OpenVocabTarget, ...]:
        return tuple(t for t in self.targets if t.enabled)


_SCALARS: dict[str, type] = {'str': str, 'int': int, 'float': float, 'bool': bool}


def _scalar(value: Any, annotation: str, path: str) -> Any:
    expected = _SCALARS[annotation]
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if type(value) is not expected:
        msg = f'{path}: expected {expected.__name__}, got {type(value).__name__}'
        raise ValueError(msg)
    return value


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f'{path}: expected a list of strings'
        raise ValueError(msg)
    return tuple(value)


def _build[T](cls: type[T], data: Any, path: str) -> T:
    if not isinstance(data, dict):
        msg = f'{path}: expected an object'
        raise ValueError(msg)
    by_name = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(by_name))
    if unknown:
        msg = f'{path}: unknown field {unknown[0]!r}'
        raise ValueError(msg)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        where = f'{path}.{key}'
        if key == 'targets':
            if not isinstance(value, list):
                msg = f'{where}: expected a list'
                raise ValueError(msg)
            kwargs[key] = tuple(
                _build(OpenVocabTarget, t, f'{where}[{i}]') for i, t in enumerate(value)
            )
        elif key == 'gating':
            kwargs[key] = _build(GatingConfig, value, where)
        elif key == 'tier3_hit_rate':
            kwargs[key] = _build(HitRateGate, value, where)
        elif key == 'parent_classes':
            kwargs[key] = _string_list(value, where)
        else:
            kwargs[key] = _scalar(value, str(by_name[key].type), where)
    return cls(**kwargs)


def decode_open_vocab_set(name: str, body: dict[str, Any]) -> OpenVocabSet:
    """The :class:`OpenVocabSet` a stored body describes (``ValueError`` if it
    has an unknown field or a wrong type)."""
    if not isinstance(body, dict):
        msg = 'open_vocab body must be an object'
        raise ValueError(msg)
    return _build(OpenVocabSet, {**body, 'name': name}, 'body')


__all__ = [
    'DEFAULT_DEDUP_IOU',
    'DEFAULT_IMAGE_MAX_SIDE',
    'DEFAULT_MAX_AREA_FRAC',
    'DEFAULT_MAX_ENABLED_TARGETS',
    'DEFAULT_MAX_INSTANCES',
    'DEFAULT_MIN_AREA_FRAC',
    'DEFAULT_MIN_SCORE',
    'MAX_ENABLED_TARGETS_CEILING',
    'GatingConfig',
    'HitRateGate',
    'OpenVocabSet',
    'OpenVocabTarget',
    'decode_open_vocab_set',
]
