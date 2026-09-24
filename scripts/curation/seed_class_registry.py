#!/usr/bin/env python3
"""Seed or validate ``class_registry.json`` against a detector's class order.

Every YOLO ``.txt`` label, every detector output and every item document
stores a bare integer class id; the registry is the only place that id
turns into a name. If the registry's id order ever diverges from the
order the detector was trained with, re-importing labels (or trusting
detector output) silently files objects under the wrong class. This
script makes the detector's own embedded class list the source of truth:

* **Seed** — no registry yet: write one whose ids ``0..N-1`` are exactly
  the model's ``names`` order.
* **Extend** — registry exists: verify ids ``0..N-1`` still agree with
  the model, then *append* any model classes the registry does not have
  yet (and any ``--extra-class`` entries). Existing entries are never
  renamed, reordered or removed, and new ids always come from
  ``max(existing id) + 1``, so a deprecated id is never reused.
* **Check** (``--check``) — write nothing; exit 1 on any drift between
  model and registry. Use it in CI / before a label import / after a
  detector promotion.

Class names are read from the ``names`` metadata an Ultralytics ONNX
export embeds (``{0: 'cat', 1: 'dog'}``), or from a YOLO dataset
``data.yaml`` (``names:`` dict or list) — pass either as ``--model``.

Drift findings:

* ``name_mismatch`` — registry id *i* has a different name than model id
  *i*. Always fatal: fixing it means retraining or a deliberate registry
  migration, never an automatic rename.
* ``deprecated_in_model`` — the model still predicts an id the registry
  deprecated. Fatal.
* ``missing_in_registry`` — the model has ids the registry lacks. Fatal
  under ``--check``; appended in write mode.
* ``append_blocked`` — a missing model id cannot be appended at its own
  id (registry-only classes already hold the next ids). Fatal.
* ``duplicate_model_name`` — the model lists one name twice. Fatal.
* ``registry_only`` — registry ids beyond the model's range (classes
  added after the model was trained). Informational; fatal with
  ``--strict``.

Usage::

    # First-time seed from a detector export (registry path from OP_REGISTRY_PATH)
    python3 scripts/curation/seed_class_registry.py --model detector.onnx

    # Seed from a dataset's data.yaml and append one post-model class
    python3 scripts/curation/seed_class_registry.py --model data.yaml \\
        --registry data/class_registry.json --extra-class background_clutter:special

    # CI / pre-import drift gate
    python3 scripts/curation/seed_class_registry.py --model detector.onnx --check
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.clients.curation_opensearch import ClassRegistry, ClassRegistryFile, RegistryClassEntry
from src.config.curation import CurationConfig


logger = logging.getLogger('seed_class_registry')

FATAL_KINDS = frozenset(
    {'name_mismatch', 'deprecated_in_model', 'duplicate_model_name', 'append_blocked'}
)


class ClassNamesError(RuntimeError):
    """The model/dataset does not carry a usable class-name list."""


# =============================================================================
# Reading class names
# =============================================================================


def _names_to_list(raw: Any, source: str) -> list[str]:
    """Normalize a ``names`` dict (int or digit-string keys) or list to an ordered list."""
    if isinstance(raw, list | tuple):
        return [str(n) for n in raw]
    if not isinstance(raw, dict):
        raise ClassNamesError(
            f'{source}: expected dict or list for names, got {type(raw).__name__}'
        )
    try:
        by_id = {int(k): str(v) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ClassNamesError(f'{source}: non-integer class id in names: {exc}') from exc
    ids = sorted(by_id)
    if ids != list(range(len(ids))):
        raise ClassNamesError(f'{source}: class ids are not contiguous from 0: {ids[:10]}')
    return [by_id[i] for i in ids]


def read_onnx_class_names(model_path: Path) -> list[str]:
    """Class names from an ONNX model's ``names`` metadata, in id order."""
    import onnx  # lazy: only needed for the ONNX source

    model = onnx.load(str(model_path), load_external_data=False)
    meta = {p.key: p.value for p in model.metadata_props}
    if 'names' not in meta:
        raise ClassNamesError(f'{model_path}: no "names" metadata — cannot derive class order')
    # Ultralytics writes a Python-repr dict (single quotes); literal_eval
    # also accepts JSON-style double quotes.
    try:
        raw = ast.literal_eval(meta['names'])
    except (ValueError, SyntaxError) as exc:
        raise ClassNamesError(f'{model_path}: unparseable names metadata: {exc}') from exc
    return _names_to_list(raw, str(model_path))


def read_yaml_class_names(yaml_path: Path) -> list[str]:
    """Class names from a YOLO dataset ``data.yaml`` (``names:`` dict or list)."""
    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    if 'names' not in data:
        raise ClassNamesError(f'{yaml_path}: no "names" key')
    names = _names_to_list(data['names'], str(yaml_path))
    nc = data.get('nc')
    if nc is not None and int(nc) != len(names):
        raise ClassNamesError(f'{yaml_path}: nc={nc} but {len(names)} names')
    return names


def read_class_names(path: Path) -> list[str]:
    if path.suffix.lower() in {'.yaml', '.yml'}:
        return read_yaml_class_names(path)
    return read_onnx_class_names(path)


# =============================================================================
# Reconciliation
# =============================================================================


@dataclass
class Finding:
    kind: str
    class_id: int
    detail: str

    @property
    def fatal(self) -> bool:
        return self.kind in FATAL_KINDS


@dataclass
class ReconcileResult:
    findings: list[Finding] = field(default_factory=list)
    appended: list[RegistryClassEntry] = field(default_factory=list)
    registry: ClassRegistryFile | None = None

    @property
    def has_fatal(self) -> bool:
        return any(f.fatal for f in self.findings)


def parse_extra_class(spec: str) -> tuple[str, str]:
    """``name`` or ``name:group`` -> ``(name, group)``."""
    name, _, group = spec.partition(':')
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f'empty class name in --extra-class {spec!r}')
    return name, (group.strip() or 'unknown')


def reconcile(
    model_names: list[str],
    existing: ClassRegistryFile | None,
    *,
    group_map: dict[str, str] | None = None,
    extra_classes: list[tuple[str, str]] | None = None,
) -> ReconcileResult:
    """Compare model class order to the registry and compute append-only additions.

    Never mutates ``existing``; ``result.registry`` is a new object with
    the appended entries (or ``None`` when a fatal finding blocks writing).
    """
    group_map = group_map or {}
    result = ReconcileResult()

    seen: dict[str, int] = {}
    for i, name in enumerate(model_names):
        if name in seen:
            result.findings.append(
                Finding('duplicate_model_name', i, f'{name!r} also at id {seen[name]}')
            )
        seen[name] = i

    registry = (
        existing.model_copy(deep=True) if existing is not None else ClassRegistryFile(classes=[])
    )
    by_id = {c.class_id: c for c in registry.classes}

    missing: list[int] = []
    for i, name in enumerate(model_names):
        entry = by_id.get(i)
        if entry is None:
            missing.append(i)
        elif entry.class_name != name:
            result.findings.append(
                Finding(
                    'name_mismatch',
                    i,
                    f'registry has {entry.class_name!r}, model has {name!r}',
                )
            )
        elif entry.deprecated:
            result.findings.append(
                Finding('deprecated_in_model', i, f'{name!r} is deprecated in the registry')
            )

    for c in sorted(registry.classes, key=lambda c: c.class_id):
        if c.class_id >= len(model_names):
            result.findings.append(
                Finding(
                    'registry_only',
                    c.class_id,
                    f'{c.class_name!r} is not a model output'
                    + (' (deprecated)' if c.deprecated else ''),
                )
            )

    next_id = max((c.class_id for c in registry.classes), default=-1) + 1
    for i in missing:
        result.findings.append(
            Finding('missing_in_registry', i, f'model class {model_names[i]!r} not in registry')
        )
        if i != next_id:
            # A lower id is free only if the registry has a hole, and a
            # higher one means registry-only classes already occupy the
            # model's next ids. Either way appending would break
            # "registry id == model id".
            result.findings.append(
                Finding(
                    'append_blocked',
                    i,
                    f'cannot append {model_names[i]!r} at id {i}: next free registry id is '
                    f'{next_id}',
                )
            )
            continue
        entry = RegistryClassEntry(
            class_id=i,
            class_name=model_names[i],
            group=group_map.get(model_names[i], 'unknown'),
        )
        registry.classes.append(entry)
        result.appended.append(entry)
        next_id += 1

    active_names = {c.class_name for c in registry.classes if not c.deprecated}
    for name, group in extra_classes or []:
        if name in active_names:
            continue
        entry = RegistryClassEntry(class_id=next_id, class_name=name, group=group)
        registry.classes.append(entry)
        result.appended.append(entry)
        active_names.add(name)
        next_id += 1

    result.registry = None if result.has_fatal else registry
    return result


def load_group_map(path: Path | None) -> dict[str, str]:
    """``{"group": ["class", ...]}`` JSON -> ``{"class": "group"}``."""
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding='utf-8'))
    out: dict[str, str] = {}
    for group, names in raw.items():
        for name in names:
            out[str(name)] = str(group)
    return out


# =============================================================================
# CLI
# =============================================================================


def run(args: argparse.Namespace) -> int:
    registry_path: Path = args.registry or CurationConfig.from_env().class_registry_path
    try:
        model_names = read_class_names(args.model)
    except (ClassNamesError, OSError) as exc:
        logger.error('%s', exc)
        return 2
    logger.info('read %d class names from %s', len(model_names), args.model)

    store = ClassRegistry(registry_path)
    existing = store.load() if registry_path.exists() else None
    result = reconcile(
        model_names,
        existing,
        group_map=load_group_map(args.group_map),
        extra_classes=args.extra_class,
    )

    for f in result.findings:
        level = logging.ERROR if f.fatal else logging.WARNING
        if f.kind == 'registry_only' and not args.strict:
            level = logging.INFO
        logger.log(level, '[%s] id=%d %s', f.kind, f.class_id, f.detail)

    strict_fail = args.strict and any(f.kind == 'registry_only' for f in result.findings)

    if args.check:
        drift = result.has_fatal or strict_fail or bool(result.appended)
        if drift:
            logger.error('registry %s is NOT in sync with %s', registry_path, args.model)
            return 1
        logger.info('registry %s is in sync with %s', registry_path, args.model)
        return 0

    if result.has_fatal or strict_fail:
        logger.error('refusing to write %s: fix the drift above first', registry_path)
        return 1
    if not result.appended:
        logger.info('registry %s already covers every model class; nothing to write', registry_path)
        return 0
    for e in result.appended:
        logger.info('append id=%d %r (group=%s)', e.class_id, e.class_name, e.group)
    if args.dry_run:
        logger.info(
            'dry-run: would write %d new class(es) to %s', len(result.appended), registry_path
        )
        return 0

    assert result.registry is not None
    # ClassRegistry's writer snapshots the previous file and does the
    # tmp+fsync+rename dance, same as every registry mutation in the API.
    store._atomic_write(result.registry)
    logger.info('wrote %s (%d classes)', registry_path, len(result.registry.classes))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Detector ONNX (reads "names" metadata) or a YOLO data.yaml',
    )
    p.add_argument(
        '--registry',
        type=Path,
        default=None,
        help='class_registry.json path (default: OP_REGISTRY_PATH / CurationConfig)',
    )
    p.add_argument(
        '--group-map',
        type=Path,
        default=None,
        help='Optional JSON {"group": ["class", ...]} used for groups of appended classes',
    )
    p.add_argument(
        '--extra-class',
        action='append',
        type=parse_extra_class,
        default=[],
        metavar='NAME[:GROUP]',
        help='Append a class that is not a model output (repeatable; no-op if present)',
    )
    p.add_argument('--check', action='store_true', help='Validate only; exit 1 on drift')
    p.add_argument(
        '--strict',
        action='store_true',
        help='Also treat registry ids beyond the model range as drift',
    )
    p.add_argument('--dry-run', action='store_true', help='Report additions without writing')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
